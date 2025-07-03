import json
import requests
import time
import random
import argparse
import os
import threading
import queue
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import pyarrow.parquet as pq
import os
import glob

class ShareGPTReGenerator:
    """Class for regenerating ShareGPT dataset responses"""

    def __init__(self, vllm_url: str, output_dir: str, 
                 temperature: float = 0.7, max_tokens: int = 2048,
                 num_threads: int = 4, request_timeout: int = 60,
                 model_name: str = "llama-model"):
        """
        Initialize ShareGPT data regenerator

        Args:
            vllm_url: vLLM service API address
            output_dir: Output dataset directory
            temperature: Text generation temperature parameter
            max_tokens: Maximum tokens per generation
            num_threads: Number of parallel processing threads
            request_timeout: API request timeout in seconds
        """
        self.vllm_url = vllm_url
        self.output_dir = output_dir
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_threads = num_threads
        self.request_timeout = request_timeout
        self.model_name = model_name

        # For thread-safe operations
        self.result_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.api_rate_limit_lock = threading.Lock()
        self.last_api_call = 0
        self.min_api_interval = 0  # Minimum API call interval in seconds

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    def load_ultra_dataset(self, directory_path:str) -> List[Dict]:
        # Get all parquet file paths in the specified directory
        parquet_files = glob.glob(os.path.join(directory_path, "*.parquet"))

        result_list = []

        # Read each parquet file and add to the list
        for file_path in parquet_files:
            table = pq.read_table(file_path)
            # Convert Table to dictionary list
            records = table.to_pydict()

            # Convert to record list
            num_rows = len(next(iter(records.values())))
            row_list = [
                {key: records[key][i] for key in records} 
                for i in range(num_rows)
            ]

            result_list.extend(row_list)

        return result_list

    def load_sharegpt_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load ShareGPT dataset

        Args:
            dataset_path: ShareGPT dataset file path, format is jsonl or json

        Returns:
            Loaded dataset
        """
        data = []
        if dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        elif dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")

        return data

    def call_vllm_api(self, prompt: str) -> str:
        """
        Call vLLM API to generate response

        Args:
            prompt: Input prompt

        Returns:
            Generated response
        """
        try:
            # Implement simple rate limiting
            with self.api_rate_limit_lock:
                current_time = time.time()
                time_since_last_call = current_time - self.last_api_call
                if time_since_last_call < self.min_api_interval:
                    time.sleep(self.min_api_interval - time_since_last_call)
                self.last_api_call = time.time()

            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            response = requests.post(self.vllm_url, json=payload, timeout=self.request_timeout)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
            #print(result["choices"][0]["message"]["content"])
            # vLLM API response format may vary depending on your deployment, please adjust as needed
            # if "text" in result:
            #     return result["text"]
            # elif "choices" in result and len(result["choices"]) > 0:
            #     return result["choices"][0]["text"]
            # else:
            #     print(f"Unknown API response format: {result}")
            #     return ""

        except Exception as e:
            print(f"Error calling vLLM API: {e}")
            return ""
    def process_conversation(self, conversation: Dict, idx: int) -> Dict:
        """
        Process a single conversation, regenerate assistant responses

        Args:
            conversation: Original conversation data
            idx: Conversation index

        Returns:
            Updated conversation data
        """
        conversations_list = conversation.get("conversations", [])

        if not conversations_list:
            # If using another format
            if "items" in conversation:
                conversations_list = conversation["items"]
            else:
                conversations_list = conversation.get("messages", [])

        # Regenerated conversations
        new_conversations = []
        context = ""

        print(conversations_list)

        for i, msg in enumerate(conversations_list):
            role = msg.get("from") or msg.get("role")
            content = msg.get("value") or msg.get("content")

            # If it's a human message, keep unchanged and add to context
            if role in ["human", "user"]:
                new_conversations.append({
                    "from": "human",
                    "value": content
                })
                context += f"Human: {content}\n\n"

            # If it's an assistant message, regenerate using vLLM
            elif role in ["gpt", "assistant"]:
                # Prepare prompt including previous context
                prompt = f"{context}Assistant:"

                # Call vLLM to generate new response
                new_response = self.call_vllm_api(prompt)

                new_conversations.append({
                    "from": "gpt",
                    "value": new_response
                })

                # Update context, add new response
                context += f"Assistant: {new_response}\n\n"

        # Save regenerated conversation
        new_data = {
            "id": conversation.get("id", f"regenerated_{idx}"),
            "conversations": new_conversations
        }

        return new_data

    def worker_task(self, task_queue: queue.Queue, results: List[Dict], 
                   progress_bar: tqdm, processed_count: List[int]):
        """
        Worker thread task function

        Args:
            task_queue: Task queue
            results: Results list
            progress_bar: Progress bar
            processed_count: Processed count
        """
        while True:
            try:
                task = task_queue.get(block=False)
                if task is None:
                    break

                idx, conversation = task
                result = self.process_conversation(conversation, idx)

                # Add result to results list (thread-safe)
                with self.result_lock:
                    results.append(result)

                # Update progress (thread-safe)
                with self.progress_lock:
                    processed_count[0] += 1
                    progress_bar.update(1)

                task_queue.task_done()

            except queue.Empty:
                break
            except Exception as e:
                print(f"Error processing conversation: {e}")
                task_queue.task_done()

    def regenerate_responses_parallel(self, conversations: List[Dict], start_index: int = 0, 
                                    end_index: Optional[int] = None, batch_save: int = 10) -> List[Dict]:
        """
        Use multithreading to regenerate assistant responses in conversations in parallel

        Args:
            conversations: Original conversation data list
            start_index: Starting processing index
            end_index: Ending processing index (exclusive)
            batch_save: Save after processing how many conversations

        Returns:
            Updated conversation data list
        """
        if end_index is None:
            end_index = len(conversations)

        total_conversations = end_index - start_index
        print(f"Starting parallel data processing, total {total_conversations} conversations, using {self.num_threads} threads")

        # Initialize results list and progress bar
        results = []
        progress_bar = tqdm(total=total_conversations)
        processed_count = [0]  # Use list to pass reference between functions

        # Create task queue
        task_queue = queue.Queue()
        for idx in range(start_index, end_index):
            task_queue.put((idx, conversations[idx]))

        # Start worker threads
        threads = []
        for _ in range(min(self.num_threads, total_conversations)):
            thread = threading.Thread(
                target=self.worker_task,
                args=(task_queue, results, progress_bar, processed_count)
            )
            thread.start()
            threads.append(thread)

        # Periodically check if batch save is needed
        last_save_count = 0
        while processed_count[0] < total_conversations:
            time.sleep(5)  # Check every 5 seconds

            # If enough new conversations have been processed, save batch
            with self.result_lock:
                current_results = results.copy()

            if len(current_results) >= last_save_count + batch_save:
                self.save_batch(
                    current_results[last_save_count:],
                    start_index + last_save_count,
                    start_index + len(current_results) - 1
                )
                last_save_count = len(current_results)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        progress_bar.close()
        print(f"Parallel processing completed, processed {len(results)} conversations")

        # Ensure results are sorted by index
        results.sort(key=lambda x: int(x["id"].split("_")[-1]) if "_" in x["id"] else 0)

        # Save remaining results
        if last_save_count < len(results):
            self.save_batch(
                results[last_save_count:],
                start_index + last_save_count,
                start_index + len(results) - 1
            )

        return results

    def save_batch(self, data: List[Dict], start_idx: int, end_idx: int):
        """
        Save batch processed data

        Args:
            data: Data to save
            start_idx: Batch start index
            end_idx: Batch end index
        """
        batch_file = os.path.join(self.output_dir, f"regenerated_{start_idx}_to_{end_idx}.json")
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Also save as jsonl format
        batch_jsonl = os.path.join(self.output_dir, f"regenerated_{start_idx}_to_{end_idx}.jsonl")
        with open(batch_jsonl, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Saved batch {start_idx} to {end_idx} complete, total {len(data)} conversations")
    def regenerate_responses_with_threadpool(self, conversations: List[Dict], start_index: int = 0,
                                    end_index: Optional[int] = None, batch_save: int = 10) -> List[Dict]:
            """
            Use thread pool to regenerate assistant responses in conversations

            Args:
                conversations: Original conversation data list
                start_index: Starting processing index
                end_index: Ending processing index (exclusive)
                batch_save: Save after processing how many conversations

            Returns:
                Updated conversation data list
            """
            if end_index is None:
                end_index = len(conversations)

            total_conversations = end_index - start_index
            print(f"Starting thread pool data processing, total {total_conversations} conversations, using {self.num_threads} threads")

            # Prepare tasks
            tasks = [(idx, conversations[idx]) for idx in range(start_index, end_index)]
            results = []

            # Use thread pool to execute tasks
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit all tasks
                future_to_idx = {executor.submit(self.process_conversation, conv, idx): (idx, i) 
                                for i, (idx, conv) in enumerate(tasks)}

                # Process completed task results
                for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(tasks)):
                    idx, task_idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results.append(result)

                        # Save after every batch_save completed tasks
                        if len(results) % batch_save == 0:
                            # Get current batch index range
                            completed_indices = sorted([future_to_idx[f][0] for f in future_to_idx if f.done()])
                            if completed_indices:
                                batch_start = min(completed_indices)
                                batch_end = max(completed_indices)
                                # Save current batch
                                current_batch = [r for r in results if int(r["id"].split("_")[-1]) <= batch_end 
                                                and int(r["id"].split("_")[-1]) >= batch_start]
                                if current_batch:
                                    self.save_batch(current_batch, batch_start, batch_end)

                    except Exception as e:
                        print(f"Error processing task {idx}: {e}")

            # Ensure results are sorted by index
            results.sort(key=lambda x: int(x["id"].split("_")[-1]) if "_" in x["id"] else 0)

            # Save remaining results
            if results:
                remaining_start = start_index + (len(results) // batch_save) * batch_save
                remaining_end = start_index + len(results) - 1
                remaining_batch = results[remaining_start - start_index:]
                if remaining_batch:
                    self.save_batch(remaining_batch, remaining_start, remaining_end)

            return results    
    def regenerate_responses(self, conversations: List[Dict], start_index: int = 0, 
                             end_index: Optional[int] = None, batch_save: int = 10) -> List[Dict]:
        """
        Single-threaded regeneration of assistant responses in conversations (keep this method as alternative)

        Args:
            conversations: Original conversation data list
            start_index: Starting processing index
            end_index: Ending processing index (exclusive)
            batch_save: Save after processing how many conversations

        Returns:
            Updated conversation data list
        """
        if end_index is None:
            end_index = len(conversations)

        print(f"Starting data processing, total {end_index - start_index} conversations")
        regenerated_data = []

        for idx in tqdm(range(start_index, end_index)):
            conversation = conversations[idx]
            new_data = self.process_conversation(conversation, idx)
            regenerated_data.append(new_data)

            # Save after processing a certain amount of data
            if (idx - start_index + 1) % batch_save == 0:
                self.save_batch(regenerated_data, start_index, idx)

        # Save remaining data
        if regenerated_data and (end_index - start_index) % batch_save != 0:
            self.save_batch(regenerated_data, start_index, end_index - 1)

        return regenerated_data
    def merge_batches(self):
        """Merge all batch files into one complete dataset file"""
        all_data = []
        batch_files = [f for f in os.listdir(self.output_dir) if f.startswith("regenerated_") and f.endswith(".json")]

        for batch_file in sorted(batch_files, key=lambda x: int(x.split('_')[1])):
            with open(os.path.join(self.output_dir, batch_file), 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                all_data.extend(batch_data)

        # Save complete dataset
        complete_file = os.path.join(self.output_dir, "regenerated_complete.json")
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

        # Save jsonl format
        complete_jsonl = os.path.join(self.output_dir, "regenerated_complete.jsonl")
        with open(complete_jsonl, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Merge completed, total {len(all_data)} conversations")

def main():
    parser = argparse.ArgumentParser(description='Regenerate ShareGPT dataset responses using vLLM')
    parser.add_argument('--dataset', type=str, required=True, help='ShareGPT dataset file path')
    parser.add_argument('--vllm_url', type=str, required=True, help='vLLM service API address')
    parser.add_argument('--output_dir', type=str, default='./regenerated_data', help='Output directory')
    parser.add_argument('--temperature', type=float, default=0.7, help='Text generation temperature parameter')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum tokens per generation')
    parser.add_argument('--start_index', type=int, default=0, help='Starting processing index')
    parser.add_argument('--end_index', type=int, default=None, help='Ending processing index (exclusive)')
    parser.add_argument('--batch_save', type=int, default=10, help='Save after processing how many conversations')
    parser.add_argument('--threads', type=int, default=4, help='Number of parallel processing threads')
    parser.add_argument('--timeout', type=int, default=60, help='API request timeout in seconds')
    parser.add_argument('--mode', type=str, default='threadpool', 
                        choices=['single', 'parallel', 'threadpool'],
                        help='Processing mode: single-threaded(single), custom multi-threaded(parallel), thread pool(threadpool)')
    parser.add_argument('--model_name', type=str, default='llama-model', help='Model name')

    args = parser.parse_args()

    regenerator = ShareGPTReGenerator(
        vllm_url=args.vllm_url,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_threads=args.threads,
        request_timeout=args.timeout,
        model_name=args.model_name
    )
    if 'ultra' in args.dataset:
        dataset = regenerator.load_ultra_dataset(args.dataset)
        print(f"Loading Ultra dataset: {args.dataset}")
    else:
    # Load dataset
        print(f"Loading ShareGPT dataset: {args.dataset}")
        dataset = regenerator.load_sharegpt_dataset(args.dataset)
    print(f"Dataset loading completed, total {len(dataset)} conversations")

    # Process data according to selected mode
    if args.mode == 'single':
        # Single-threaded processing
        regenerator.regenerate_responses(
            conversations=dataset,
            start_index=args.start_index,
            end_index=args.end_index,  
            batch_save=args.batch_save
        )
    elif args.mode == 'parallel':
        # Custom multi-threaded processing
        regenerator.regenerate_responses_parallel(
            conversations=dataset,
            start_index=args.start_index,
            end_index=args.end_index,  
            batch_save=args.batch_save
        )
    else:  # threadpool
        # Thread pool processing (default)
        regenerator.regenerate_responses_with_threadpool(
            conversations=dataset,
            start_index=args.start_index,
            end_index=args.end_index,  
            batch_save=args.batch_save
        )

    # Merge batch files
    regenerator.merge_batches()
if __name__ == "__main__":
    main()
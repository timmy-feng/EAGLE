#!/bin/bash

DATASET_PATH="${DATASET_PATH:-./data/ultrachat/data/}"
OUTPUT_PATH="${OUTPUT_PATH:-./data/Llama-3.2-1B-Instruct/}"
VLLM_URL="${VLLM_URL:-http://0.0.0.0:8000}"
MODEL_NAME="${MODEL_NAME:-unsloth/Llama-3.2-1B-Instruct}"

python3 scripts/gen_sharegpt.py --dataset $DATASET_PATH --vllm_url $VLLM_URL/v1/chat/completions --output_dir $OUTPUT_PATH --threads 2048 --batch_save 20000 --model_name $MODEL_NAME --end_index 10
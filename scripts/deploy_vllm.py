import threading

import modal
import modal.experimental
import requests
import time

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

vllm_image = vllm_image.env({"VLLM_USE_V1": "1"})

FAST_BOOT = True

app = modal.App("eagle-vllm-server")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.cls(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    min_containers=1,
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    experimental_options={"flash": True},
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
class FlashTest:
    @modal.enter()
    def enter(self):
        self.server_thread = threading.Thread(target=serve, daemon=True)
        self.server_thread.start()

        # wait for port
        while True:
            try:
                requests.get(f"http://localhost:{VLLM_PORT}/metrics")
                break
            except requests.exceptions.RequestException:
                time.sleep(1)

        self.flash_manager = modal.experimental.flash_forward(VLLM_PORT)

    @modal.exit()
    def exit(self):
        self.flash_manager.stop()


def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]

    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)

import modal
import os
import subprocess
import sys

GPU_COUNT = 1

# Create Modal app
app = modal.App("eagle-training")

# Create a volume for saving model checkpoints
volume = modal.Volume.from_name("eagle-checkpoints", create_if_missing=True)

# Define the Docker image with all necessary dependencies
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu20.04", add_python="3.11")
    .apt_install(["git"])
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "torch==2.6.0",
        "transformers>=4.47.0",
        "accelerate==0.26.0",
        "deepspeed==0.16.9",
        "sentencepiece==0.1.99",
        "datasets",
        "wandb"
    )
    # .pip_install(  # add flash-attn
    #     "flash-attn==2.7.4.post1", extra_options="--no-build-isolation"
    # )
    .add_local_dir("eagle/traineagle3", remote_path="/eagle")
    .add_local_dir("weights", remote_path="/weights")
    .add_local_dir("data", remote_path="/data")
)

@app.function(
    image=image,
    cpu=64,
    gpu=f"H100:{GPU_COUNT}",
    timeout=86400,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("wandb-secret")]  # For wandb logging

)
def train_model(
    basepath: str,
    trainpath: str,
    testpath: str,
    configpath: str,
    savedir: str,
    deepspeed_config: str
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(GPU_COUNT)
    os.environ["RANK"] = "0"
    
    training_script = "/eagle/main.py"
    
    cmd = [
        "deepspeed",
        f"--num_gpus={GPU_COUNT}",
        "--master_port=29500",
        training_script,
        f"--basepath={basepath}",
        f"--trainpath={trainpath}",
        f"--testpath={testpath}",
        f"--savedir={savedir}",
        f"--configpath={configpath}",
        f"--deepspeed_config={deepspeed_config}"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    os.makedirs(savedir, exist_ok=True)
    
    subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)

@app.local_entrypoint()
def main(
    basepath: str,
    trainpath: str,
    testpath: str,
    configpath: str,
    savedir: str = "/checkpoints/eagle",
    deepspeed_config: str = "/eagle/ds_config.json"
):
    print("Starting training with:")
    print(f"  Base model path: {basepath}")
    print(f"  Training data: {trainpath}")
    print(f"  Test data: {testpath}")
    print(f"  Save directory: {savedir}")
    print(f"  DeepSpeed config: {deepspeed_config}")
    
    result = train_model.remote(
        basepath=basepath,
        trainpath=trainpath,
        testpath=testpath,
        configpath=configpath,
        savedir=savedir,
        deepspeed_config=deepspeed_config
    )
    
    print("Training result:", result)
    return result
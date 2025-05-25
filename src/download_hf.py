from huggingface_hub import snapshot_download, login
import os
import shutil

hf_token = os.getenv("HF_TOKEN")

login(token=hf_token)
hf_model_path = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
model_name = hf_model_path.split("/")[-1]
local_model_dir = f"../models/{model_name}"

# Function to remove the model
def remove_model(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Model directory {directory} has been removed.")
    else:
        print(f"Model directory {directory} does not exist.")


def load_model(repo_id, local_dir):
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"Model {repo_id} has been downloaded to {local_dir}.")

# Example usage
load_model(hf_model_path, local_model_dir)

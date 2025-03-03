import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                  local_dir="./haggingfaceModels",
                  allow_patterns=["*.json", "*.safetensors"])

# huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir ./haggingfaceModels
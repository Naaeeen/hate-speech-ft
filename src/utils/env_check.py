import os
import torch
import transformers
import datasets
import peft

print("Environment check")
print("-" * 40)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Transformers version:", transformers.__version__)
print("Datasets version:", datasets.__version__)
print("PEFT version:", peft.__version__)
print("HF_HOME:", os.environ.get("HF_HOME"))
print("HF_HUB_CACHE:", os.environ.get("HF_HUB_CACHE"))
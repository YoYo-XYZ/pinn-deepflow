from src.deepflow.Utility import get_device
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"DeepFlow device: {get_device()}")

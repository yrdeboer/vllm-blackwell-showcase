import torch
print(f"Active Torch: {torch.__version__}")
print(f"CUDA-version used by Torch: {torch.version.cuda}")
print(f"Blackwell (sm_120) support: {'Yes' if torch.cuda.get_device_capability(0) >= (12, 0) else 'No'}")

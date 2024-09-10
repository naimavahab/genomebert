import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version used by PyTorch: {torch.version.cuda}")
num_gpus = torch.cuda.device_count()

print(f"Number of GPUs available: {num_gpus}")

for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

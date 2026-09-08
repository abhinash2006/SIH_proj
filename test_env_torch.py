import sys

try:
    import torch
    print(f"[SUCCESS] PyTorch Version: {torch.__version__}")
    print(f"[SUCCESS] CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[SUCCESS] Device Count: {torch.cuda.device_count()}")
        print(f"[SUCCESS] Device Name: {torch.cuda.get_device_name(0)}")
        print(f"[SUCCESS] CUDA Capability: {torch.cuda.get_device_capability(0)}")
        print(f"[SUCCESS] Allocated VRAM: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB")
except ImportError as e:
    print(f"[ERROR] torch import failed: {e}")
    sys.exit(1)

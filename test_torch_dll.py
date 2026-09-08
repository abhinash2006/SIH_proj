import os
import sys

torch_lib = r"D:\Vggt\drone_vggt_env\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    print(f"Adding DLL directory: {torch_lib}")
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(torch_lib)

print("Attempting to import torch...")
try:
    import torch
    print("[SUCCESS] PyTorch Version:", torch.__version__)
    print("[SUCCESS] CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[SUCCESS] Device Name:", torch.cuda.get_device_name(0))
        print("[SUCCESS] Memory Allocated:", torch.cuda.memory_allocated(0))
        # Simple tensor operation
        x = torch.randn(3, 3, device="cuda")
        print("[SUCCESS] CUDA Tensor:", (x * 2).sum().item())
except Exception as e:
    print("[ERROR] Failed to import/run torch:", e)
    import traceback
    traceback.print_exc()

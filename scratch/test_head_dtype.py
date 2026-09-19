import os, sys
from pathlib import Path
torch_lib = r"D:\Vggt\drone_vggt_env\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib) and hasattr(os, "add_dll_directory"):
    try: os.add_dll_directory(torch_lib)
    except Exception: pass
import torch

sys.path.insert(0, str(Path("vggt_repo").resolve()))
from vggt.heads.camera_head import CameraHead

head = CameraHead(dim_in=2048).cuda()
print("Default head param dtype:", next(head.parameters()).dtype)

x_bf16 = torch.randn(1, 4, 10, 2048, device="cuda", dtype=torch.bfloat16)
x_fp32 = torch.randn(1, 4, 10, 2048, device="cuda", dtype=torch.float32)

print("\nTesting head(Float32) with autocast(False)...")
try:
    with torch.cuda.amp.autocast(enabled=False):
        out = head([x_fp32])
    print("[SUCCESS] head works with Float32 input!")
except Exception as e:
    print("[ERROR Float32]:", e)

print("\nTesting head(BFloat16) with autocast(False)...")
try:
    with torch.cuda.amp.autocast(enabled=False):
        out = head([x_bf16])
    print("[SUCCESS] head works with BFloat16 input!")
except Exception as e:
    print("[ERROR BFloat16]:", e)

print("\nTesting head.bfloat16() with BFloat16 input...")
head_bf16 = CameraHead(dim_in=2048).cuda().bfloat16()
try:
    with torch.cuda.amp.autocast(enabled=False):
        out = head_bf16([x_bf16])
    print("[SUCCESS] head_bf16 works with BFloat16 input!")
except Exception as e:
    print("[ERROR head_bf16]:", e)

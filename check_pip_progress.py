import os
import glob
import sys

site_packages = r"d:\Vggt\drone_vggt_env\Lib\site-packages"
if os.path.exists(site_packages):
    items = os.listdir(site_packages)
    print("Items in site-packages:", len(items))
    for it in items[:15]:
        print(" -", it)

# Check for torch
try:
    import torch
    print("\n>>> PyTorch is INSTALLED!")
    print("Version:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except ImportError:
    print("\n>>> PyTorch is not yet fully installed.")

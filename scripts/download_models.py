"""
Safe Model Downloader for UAV 3D Reconstruction System.
Strictly adheres to Model Download Rule 31:
1. Checks whether model checkpoint already exists.
2. Checks available GPU VRAM.
3. Checks available disk storage.
4. Announces exact model and download source before starting.
5. Prefers official model repositories.
6. Prevents duplicate downloads.
"""

import os
import sys
import shutil
import logging
from pathlib import Path
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("ModelDownloader")


def get_system_resources():
    """Checks disk space and GPU memory."""
    # Disk space on current drive
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)

    gpu_info = "CPU only"
    vram_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_info = f"{torch.cuda.get_device_name(0)} ({vram_gb:.2f} GB VRAM)"
    except Exception:
        pass

    return free_gb, vram_gb, gpu_info


MODELS = {
    "yolo11n": {
        "dest": "yolo11n.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "size_mb": 5.6,
        "required_vram_gb": 0.5,
    },
    "vggt_1b": {
        "dest": "models/VGGT-1B/model.pt",
        "url": "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
        "size_mb": 4800,
        "required_vram_gb": 3.8,
    },
    "depth_anything_v2": {
        "dest": "models/depth_anything_v2/depth_anything_v2_vits.pth",
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
        "size_mb": 99,
        "required_vram_gb": 0.5,
    },
    "sam2_1_small": {
        "dest": "models/sam2/sam2.1_hiera_small.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "size_mb": 184,
        "required_vram_gb": 1.5,
    },
}


def download_model(model_key: str):
    if model_key not in MODELS:
        print(f"[ERROR] Unknown model '{model_key}'. Available: {list(MODELS.keys())}")
        return

    info = MODELS[model_key]
    dest_path = Path(info["dest"])

    # 1. Check whether it already exists
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / (1024**2)
        print(f"[FOUND] Model '{model_key}' already exists at '{dest_path}' ({size_mb:.1f} MB). Skipping duplicate download.")
        return

    # 2 & 3. Check disk space and GPU memory
    free_gb, vram_gb, gpu_info = get_system_resources()
    needed_gb = info["size_mb"] / 1024.0

    print(f"\n[RESOURCE CHECK]")
    print(f"  * Available Disk Space: {free_gb:.2f} GB (Required: {needed_gb:.2f} GB)")
    print(f"  * Detected Hardware   : {gpu_info}")

    if free_gb < (needed_gb + 2.0):
        print(f"[ERROR] Insufficient disk space! {free_gb:.2f} GB available, need at least {needed_gb + 2.0:.2f} GB safety buffer.")
        return

    # 4 & 5. Announce model, official source, and proceed
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[DOWNLOAD] Downloading '{model_key}' from official repository:")
    print(f"  * Source URL : {info['url']}")
    print(f"  * Destination: {dest_path}")
    print(f"  * Size       : ~{info['size_mb']} MB")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded / total_size * 100)
            mb = downloaded / (1024**2)
            tot_mb = total_size / (1024**2)
            sys.stdout.write(f"\r  * Progress: {pct:.1f}% ({mb:.1f}/{tot_mb:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(info["url"], str(dest_path), reporthook=report_progress)
    print(f"\n[SUCCESS] Model '{model_key}' downloaded successfully to '{dest_path}'!\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        download_model(target)
    else:
        print("Usage: python scripts/download_models.py <model_name>")
        print("Available models:")
        for k, v in MODELS.items():
            status = "INSTALLED" if os.path.exists(v["dest"]) else "NOT FOUND"
            print(f"  - {k:<20} [{status:<9}] -> {v['dest']}")

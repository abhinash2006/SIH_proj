import os
import sys
import urllib.request
from pathlib import Path

CHECKPOINT_URL = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
TARGET_DIR = Path("models/depth_anything_v2")
TARGET_FILE = TARGET_DIR / "depth_anything_v2_vits.pth"

def download_checkpoint():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET_FILE.exists() and TARGET_FILE.stat().st_size > 10_000_000:
        print(f"[INFO] Depth Anything V2 checkpoint already exists at {TARGET_FILE} ({TARGET_FILE.stat().st_size / 1e6:.1f} MB)")
        return str(TARGET_FILE)

    print(f"[INFO] Downloading Depth Anything V2 Small checkpoint from:\n  {CHECKPOINT_URL}")
    def progress_hook(count, block_size, total_size):
        percent = min(100, int(count * block_size * 100 / total_size))
        sys.stdout.write(f"\rDownloading: {percent}% [{count * block_size / 1e6:.1f} / {total_size / 1e6:.1f} MB]")
        sys.stdout.flush()

    urllib.request.urlretrieve(CHECKPOINT_URL, TARGET_FILE, reporthook=progress_hook)
    print(f"\n[SUCCESS] Downloaded checkpoint to {TARGET_FILE} ({TARGET_FILE.stat().st_size / 1e6:.1f} MB)")
    return str(TARGET_FILE)

if __name__ == "__main__":
    download_checkpoint()

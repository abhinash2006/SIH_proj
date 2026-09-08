import os
import sys
import time
import requests
from tqdm import tqdm

torch_lib = r"D:\Vggt\drone_vggt_env\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib) and hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(torch_lib)

import torch
from vggt.models.vggt import VGGT

MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "VGGT-1B", "model.pt")
LOCAL_MODEL_PATH = os.path.abspath(LOCAL_MODEL_PATH)

def download_file_with_resume(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".tmp"
    
    initial_bytes = 0
    if os.path.exists(temp_path):
        initial_bytes = os.path.getsize(temp_path)
        print(f"[INFO] Resuming download from {initial_bytes / (1024**2):.2f} MB")
        
    headers = {}
    if initial_bytes > 0:
        headers['Range'] = f'bytes={initial_bytes}-'
        
    response = requests.get(url, headers=headers, stream=True, timeout=30)
    
    if response.status_code == 416: # Range not satisfiable
        # File is already complete or invalid range
        if os.path.exists(temp_path):
            os.replace(temp_path, dest_path)
            return
    elif response.status_code not in (200, 206):
        response.raise_for_status()
        
    total_size = int(response.headers.get('content-length', 0)) + initial_bytes
    mode = 'ab' if initial_bytes > 0 and response.status_code == 206 else 'wb'
    if mode == 'wb':
        initial_bytes = 0
        
    print(f"[INFO] Downloading VGGT-1B weights (~{total_size / (1024**3):.2f} GB) to: {dest_path}")
    
    chunk_size = 1024 * 1024 # 1 MB
    with open(temp_path, mode) as f:
        with tqdm(total=total_size, initial=initial_bytes, unit='B', unit_scale=True, desc="VGGT-1B Weights") as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
    os.replace(temp_path, dest_path)
    print(f"[SUCCESS] Download completed successfully: {dest_path}")

def verify_model(weights_path=LOCAL_MODEL_PATH):
    print(f"[INFO] Verifying model loading from: {weights_path}...")
    try:
        model = VGGT()
        print("[INFO] Loading state_dict on CPU...")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[SUCCESS] Model loaded successfully! (Missing keys: {len(missing)}, Unexpected: {len(unexpected)})")
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"[SUCCESS] Total parameter count: {num_params:,}")
        return True
    except Exception as e:
        print(f"[ERROR] Model verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if os.path.exists(LOCAL_MODEL_PATH) and os.path.getsize(LOCAL_MODEL_PATH) > 1024 * 1024 * 100:
        print(f"[INFO] Local checkpoint already exists at {LOCAL_MODEL_PATH} ({os.path.getsize(LOCAL_MODEL_PATH)/(1024**3):.2f} GB)")
    else:
        try:
            download_file_with_resume(MODEL_URL, LOCAL_MODEL_PATH)
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            sys.exit(1)
            
    verified = verify_model(LOCAL_MODEL_PATH)
    if not verified:
        sys.exit(1)

if __name__ == "__main__":
    main()

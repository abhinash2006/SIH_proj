import sys
import os
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("YOLOEnvironmentCheck")

def check_environment():
    print("=" * 60)
    print("      ULTRALYTICS YOLO ENVIRONMENT DIAGNOSTIC REPORT")
    print("=" * 60)
    
    python_ver = sys.version.split()[0]
    print(f"Python Version      : {python_ver}")
    
    torch_ver = torch.__version__
    print(f"PyTorch Version     : {torch_ver}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available      : {cuda_available}")
    
    if cuda_available:
        cuda_ver = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"CUDA Version        : {cuda_ver}")
        print(f"GPU Device Name     : {gpu_name}")
        print(f"Total VRAM          : {vram_gb:.2f} GB")
    else:
        print("CUDA Version        : N/A (CPU Mode)")
        print("GPU Device Name     : N/A (CPU Mode)")
        print("Total VRAM          : N/A (CPU Mode)")
        
    print("-" * 60)
    
    try:
        import ultralytics
        ultralytics_ver = ultralytics.__version__
        print(f"Ultralytics Package : INSTALLED (Version {ultralytics_ver})")
    except ImportError as e:
        print(f"Ultralytics Package : NOT INSTALLED ({e})")
        return False

    # Attempt loading preferred model (yolo26n.pt), fallback to yolo11n.pt
    from ultralytics import YOLO
    
    models_to_test = ["yolo26n.pt", "yolo11n.pt", "yolov8n.pt"]
    loaded_model_name = None
    model_obj = None
    
    for model_candidate in models_to_test:
        try:
            device = "cuda" if cuda_available else "cpu"
            print(f"Testing model load  : '{model_candidate}' on device '{device}'...")
            model_obj = YOLO(model_candidate)
            model_obj.to(device)
            loaded_model_name = model_candidate
            print(f"Model Load Status   : SUCCESS ({model_candidate} loaded on {device.upper()})")
            break
        except Exception as err:
            print(f"Model Load Status   : FAILED ({model_candidate}: {err})")

    if model_obj is not None:
        # Perform quick synthetic inference verification
        import numpy as np
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model_obj.predict(dummy_img, verbose=False)
        print(f"Diagnostic Predict  : SUCCESS ({len(results)} output frame result)")
        print("=" * 60)
        print("ENVIRONMENT STATUS  : PASSED (Ultralytics YOLO Ready)")
        print("=" * 60)
        return True
    else:
        print("=" * 60)
        print("ENVIRONMENT STATUS  : FAILED (Could not load any YOLO candidate model)")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)

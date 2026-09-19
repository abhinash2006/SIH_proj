import os, sys
from pathlib import Path
torch_lib = r"D:\Vggt\drone_vggt_env\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib) and hasattr(os, "add_dll_directory"):
    try: os.add_dll_directory(torch_lib)
    except Exception: pass
import torch
sys.path.insert(0, ".")
from src.vggt_inference import VGGTInferenceEngine

engine = VGGTInferenceEngine()
engine.load_model()

# Test with 2 dummy image tensors
dummy_imgs = torch.randn(1, 2, 3, 392, 392, device=engine.device)
print(f"Running inference on {dummy_imgs.shape} with mixed_precision={engine.use_mixed_precision}, dtype={engine.dtype}...")

try:
    with torch.cuda.amp.autocast(enabled=engine.use_mixed_precision, dtype=engine.dtype):
        preds = engine.model(dummy_imgs)
    print("[SUCCESS] engine.model forward pass succeeded!")
    print("Pred keys:", preds.keys())
except Exception as e:
    print("[ERROR in engine.model]:", e)
    import traceback
    traceback.print_exc()

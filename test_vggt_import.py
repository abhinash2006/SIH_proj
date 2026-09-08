import os
import sys

torch_lib = r"D:\Vggt\drone_vggt_env\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib) and hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(torch_lib)

import torch
import numpy as np

print("[INFO] Testing VGGT import...")
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

print("[SUCCESS] VGGT classes imported successfully!")
print("[INFO] Instantiating VGGT architecture...")
model = VGGT()
num_params = sum(p.numel() for p in model.parameters())
print(f"[SUCCESS] VGGT instantiated successfully! Total parameters: {num_params:,} (~{num_params/1e6:.1f}M)")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Testing geometry unprojection with shape (S=2, H=518, W=518, 1)...")
dummy_depth = np.ones((2, 518, 518, 1), dtype=np.float32)
dummy_extri = np.tile(np.eye(4)[:3][None, ...], (2, 1, 1))
dummy_intri = np.tile(np.array([[500., 0., 259.], [0., 500., 259.], [0., 0., 1.]])[None, ...], (2, 1, 1))
pts = unproject_depth_map_to_point_map(dummy_depth, dummy_extri, dummy_intri)
print(f"[SUCCESS] Geometry unprojection works! Output shape: {pts.shape} (Expected: (2, 518, 518, 3))")
print("[SUCCESS] All basic VGGT modules and geometry functions verified!")

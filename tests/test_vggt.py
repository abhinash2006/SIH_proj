import os
import pytest
import numpy as np

torch_lib = r"D:\Vggt\drone_vggt_env\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib) and hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(torch_lib)

import torch
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map

def test_vggt_architecture():
    model = VGGT(enable_camera=True, enable_point=True, enable_depth=True, enable_track=False)
    assert model.camera_head is not None
    assert model.depth_head is not None
    assert model.point_head is not None
    
def test_geometry_unprojection():
    S, H, W = 2, 64, 64
    depth = np.ones((S, H, W, 1), dtype=np.float32) * 5.0
    extrinsics = np.tile(np.eye(4)[:3][None, ...], (S, 1, 1))
    intrinsics = np.tile(np.array([[100., 0., 32.], [0., 100., 32.], [0., 0., 1.]])[None, ...], (S, 1, 1))
    
    pts = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)
    assert pts.shape == (S, H, W, 3)
    # Check depth consistency (z ~ 5.0 in camera coords)
    assert np.allclose(pts[:, :, :, 2], 5.0, atol=1e-3)

import os
import pytest
import numpy as np

def test_depth_anything_v2_import():
    from src.depth_anything_v2 import DepthAnythingV2Estimator
    estimator = DepthAnythingV2Estimator(model_name="small", encoder="vits")
    assert estimator.encoder == "vits"

def test_depth_anything_v2_synthetic_predict():
    from src.depth_anything_v2 import DepthAnythingV2Estimator
    estimator = DepthAnythingV2Estimator()
    dummy_img = (np.random.rand(200, 300, 3) * 255).astype(np.uint8)
    depth_map = estimator.predict_image(dummy_img, input_size=256)
    
    assert depth_map.shape == (200, 300)
    assert depth_map.dtype == np.float32
    assert np.all(depth_map >= 0)

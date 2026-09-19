import numpy as np
import pytest
from src.depth.depth_anything import DepthAnythingV2Estimator, DepthResult


def test_depth_anything_estimator_fallback():
    estimator = DepthAnythingV2Estimator(checkpoint_path="non_existent.pth")
    dummy_img = np.full((120, 160, 3), 128, dtype=np.uint8)

    result: DepthResult = estimator.estimate(dummy_img)
    assert isinstance(result, DepthResult)
    assert result.depth_map.shape == (120, 160)
    assert result.confidence.shape == (120, 160)
    assert result.scale_type == "relative"
    assert result.processing_time_ms > 0.0

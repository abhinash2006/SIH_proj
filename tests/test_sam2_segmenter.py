import os
import cv2
import numpy as np
import pytest
from src.segmentation.sam2_segmenter import SAM2Segmenter, MaskResult
from src.detection.dynamic_object_filter import DynamicObjectFilter


def test_sam2_segmenter_fallback():
    # Test fallback bounding box mask generator when SAM 2 checkpoint does not exist
    segmenter = SAM2Segmenter(checkpoint_path="non_existent_sam2.pt", use_fallback_mask=True)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [[10.0, 10.0, 50.0, 50.0]]

    result: MaskResult = segmenter.generate_mask(img, boxes)
    assert result.source == "fallback_bbox"
    assert result.mask.shape == (100, 100)
    assert result.dynamic_pixel_count > 0
    assert result.dynamic_pixel_ratio > 0.0


def test_dynamic_object_filter_disabled(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    dyn_filter = DynamicObjectFilter(mode="disabled")
    res = dyn_filter.process_frame(str(img_path), str(tmp_path / "masks"))

    assert res.dynamic_detected is False
    assert res.dynamic_pixel_ratio == 0.0
    assert os.path.exists(res.mask_path)

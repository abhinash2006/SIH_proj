import os
import cv2
import pytest
import numpy as np
from src.frame_quality import FrameQualityAnalyzer
from src.frame_selector import FrameSelector

def test_sharpness_and_exposure_metrics(tmp_path):
    # Sharp checkered image with balanced exposure (mean ~125)
    sharp_img = np.full((100, 100, 3), 70, dtype=np.uint8)
    sharp_img[::4, ::4] = 190
    sharp_img[1::4, 1::4] = 190
    sharp_path = tmp_path / "sharp.jpg"
    cv2.imwrite(str(sharp_path), sharp_img)
    
    # Blurry image
    blurry_img = cv2.GaussianBlur(sharp_img, (15, 15), 5)
    blurry_path = tmp_path / "blurry.jpg"
    cv2.imwrite(str(blurry_path), blurry_img)
    
    res_sharp = FrameQualityAnalyzer.analyze_frame(sharp_path)
    res_blurry = FrameQualityAnalyzer.analyze_frame(blurry_path)
    
    assert res_sharp["sharpness"] > res_blurry["sharpness"]
    assert res_sharp["quality_score"] > res_blurry["quality_score"]

def test_duplicate_filtering(tmp_path):
    img_dir = tmp_path / "dup_test"
    img_dir.mkdir()
    
    # Create 4 identical images + 1 different image
    base_img = np.full((100, 100, 3), 120, dtype=np.uint8)
    diff_img = np.full((100, 100, 3), 40, dtype=np.uint8)
    cv2.circle(diff_img, (50, 50), 20, (255, 255, 255), -1)
    
    paths = []
    for i in range(4):
        p = img_dir / f"frame_{i}.jpg"
        cv2.imwrite(str(p), base_img)
        paths.append(str(p))
        
    p_diff = img_dir / "frame_diff.jpg"
    cv2.imwrite(str(p_diff), diff_img)
    paths.append(str(p_diff))
    
    selector = FrameSelector({"frame_selection": {"similarity_threshold": 0.95, "sharpness_threshold": 0.0}})
    results = selector.select_frames(paths)
    
    # Redundant identical frames should be compressed
    assert results["selected_count"] < len(paths)

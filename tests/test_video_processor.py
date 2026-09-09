import os
import cv2
import pytest
import numpy as np
from pathlib import Path
from src.video_processor import VideoProcessor

def test_image_folder_processing(tmp_path):
    # Create dummy image folder
    img_dir = tmp_path / "test_images"
    img_dir.mkdir()
    
    for i in range(5):
        dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i}.jpg"), dummy_img)
        
    assert VideoProcessor.is_image_folder(img_dir) is True
    
    out_dir = tmp_path / "extracted"
    extracted = VideoProcessor.extract_frames(img_dir, out_dir, max_frames=3)
    assert len(extracted) == 3
    for p in extracted:
        assert os.path.exists(p)

def test_video_metadata_synthetic(tmp_path):
    # Create a small synthetic video
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (120, 80))
    for _ in range(20):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    assert VideoProcessor.is_video_file(video_path) is True
    meta = VideoProcessor.get_video_metadata(video_path)
    assert meta["frame_count"] == 20
    assert meta["width"] == 120
    assert meta["height"] == 80
    assert meta["fps"] == 10.0

def test_video_extract_frames(tmp_path):
    video_path = tmp_path / "test_video_extract.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (120, 80))
    for _ in range(15):
        frame = np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    out_dir = tmp_path / "extracted_vid"
    extracted = VideoProcessor.extract_frames(video_path, out_dir, target_fps=2.0)
    assert isinstance(extracted, list)
    assert len(extracted) > 0
    for p in extracted:
        assert os.path.exists(p)

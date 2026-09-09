import pytest
import numpy as np
from pathlib import Path
from src.yolo_detector import YOLODetector

def test_yolo_detector_load_model():
    detector = YOLODetector(model_name="yolo26n.pt", confidence=0.35)
    model = detector.load_model()
    assert model is not None
    detector.release()

def test_yolo_detector_synthetic_inference():
    detector = YOLODetector(model_name="yolo26n.pt", confidence=0.35)
    
    # Create synthetic blank RGB image
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Inference on blank image should return clean empty list
    dets = detector.detect_image(dummy_img, frame_id="frame_0000.jpg", frame_idx=0)
    assert isinstance(dets, list)
    assert len(dets) == 0
    detector.release()

def test_yolo_detector_original_coordinate_scaling():
    detector = YOLODetector(model_name="yolo26n.pt", confidence=0.10)
    
    # Create synthetic image with bright rectangle simulating an object
    img = np.zeros((1500, 2000, 3), dtype=np.uint8)
    img[600:800, 900:1100] = 255
    
    dets = detector.detect_image(img, frame_id="frame_0001.jpg", frame_idx=1)
    
    for det in dets:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        assert 0 <= x1 <= 2000
        assert 0 <= x2 <= 2000
        assert 0 <= y1 <= 1500
        assert 0 <= y2 <= 1500
        assert det["image_width"] == 2000
        assert det["image_height"] == 1500
        assert "raw_detector_confidence" in det
        assert "validated_detection_confidence" in det

    detector.release()

def test_yolo_tiled_inference_tiling_generation():
    detector = YOLODetector(tiled_inference=True, tile_size=1024, tile_overlap=0.20)
    tiles = detector._generate_tiles(W=2000, H=1500, tile_sz=1024, overlap=0.20)
    
    assert len(tiles) > 1
    for tx1, ty1, tx2, ty2 in tiles:
        assert 0 <= tx1 < tx2 <= 2000
        assert 0 <= ty1 < ty2 <= 1500
    detector.release()

def test_yolo_track_frames_multi_frame():
    detector = YOLODetector(model_name="yolo26n.pt", confidence=0.08)
    
    # Create two synthetic frames: empty
    f1 = np.zeros((720, 1280, 3), dtype=np.uint8)
    f2 = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    res = detector.track_frames([f1, f2], frame_ids=["frame_0000.jpg", "frame_0001.jpg"])
    assert len(res) == 2
    assert isinstance(res[0], list)
    assert isinstance(res[1], list)
    detector.release()


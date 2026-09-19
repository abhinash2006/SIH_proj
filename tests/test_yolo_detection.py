import os
import cv2
import numpy as np
import pytest
from src.detection.yolo_detector import YOLODetector, DetectionResult


@pytest.fixture
def test_image(tmp_path):
    img_path = tmp_path / "test_frame.jpg"
    img = np.full((320, 320, 3), 100, dtype=np.uint8)
    # Draw simple shapes
    cv2.circle(img, (160, 160), 40, (0, 0, 255), -1)
    cv2.imwrite(str(img_path), img)
    return str(img_path)


def test_yolo_detector_init():
    detector = YOLODetector(model_path="yolo11n.pt")
    assert "person" in detector.dynamic_classes
    assert "car" in detector.dynamic_classes
    assert "truck" in detector.dynamic_classes


def test_yolo_detector_inference(test_image):
    if not os.path.exists("yolo11n.pt") and not os.path.exists("yolov8n.pt"):
        pytest.skip("YOLO weights not present in workspace")

    weight = "yolo11n.pt" if os.path.exists("yolo11n.pt") else "yolov8n.pt"
    detector = YOLODetector(model_path=weight, device="cpu")
    result: DetectionResult = detector.detect(test_image)

    assert isinstance(result, DetectionResult)
    assert result.processing_time_ms > 0.0
    assert isinstance(result.detections, list)

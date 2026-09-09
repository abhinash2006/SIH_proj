import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import cv2
from PIL import Image

from src.yolo_detector import YOLODetector

logger = logging.getLogger("DisasterObjectDetection")

class DisasterObjectDetector:
    """
    Object & Person Detector for drone aerial disaster inspection using Ultralytics YOLO models (yolo26n.pt / yolo11n.pt).
    Delegates perception to isolated YOLODetector.
    """

    def __init__(self, model_size: str = "n", conf_threshold: float = 0.35, tile_inference: bool = True, model_name: str = "yolo26n.pt"):
        self.conf_threshold = conf_threshold
        self.model_size = model_size
        self.tile_inference = tile_inference
        if "yolo" in model_name:
            target_model = model_name
        else:
            target_model = f"yolov8{model_size}.pt"
        self.detector = YOLODetector(
            model_name=target_model,
            confidence=conf_threshold,
            tiled_inference=tile_inference
        )

    def load_model(self):
        return self.detector.load_model()

    def detect_objects_in_frame(
        self,
        image_input: Union[str, Path, np.ndarray],
        frame_idx: int = 0
    ) -> List[Dict[str, Any]]:
        frame_id = f"frame_{frame_idx:04d}.jpg"
        return self.detector.detect_image(image_input, frame_id=frame_id, frame_idx=frame_idx, conf=self.conf_threshold)

    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        detector = YOLODetector()
        return detector.annotate_frame(image, detections)

    def _nms_detections(self, detections: List[Dict[str, Any]], iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
        # Format mock detections for YOLODetector NMS if needed
        for d in detections:
            if "bbox_xyxy" not in d and "bbox" in d:
                d["bbox_xyxy"] = d["bbox"]
            if "raw_detector_confidence" not in d and "confidence" in d:
                d["raw_detector_confidence"] = d["confidence"]
        return self.detector._apply_nms(detections, iou_thresh=iou_thresh)



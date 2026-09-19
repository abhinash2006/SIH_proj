"""
YOLO Object Detector for UAV Aerial Imagery.
Identifies potential dynamic objects (people, vehicles, animals) while preserving static infrastructure.
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_DYNAMIC_CLASSES = [
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "animal",
    "dog",
    "cat",
    "horse",
    "sheep",
    "cow",
    "boat",
    "airplane",
]


@dataclass
class Detection:
    box: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    is_dynamic: bool


@dataclass
class DetectionResult:
    frame_path: str
    detections: List[Detection]
    dynamic_count: int
    processing_time_ms: float


class YOLODetector:
    """
    Ultralytics YOLO detector wrapper with dynamic class filtering.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        dynamic_classes: Optional[List[str]] = None,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.dynamic_classes = set(c.lower() for c in (dynamic_classes or DEFAULT_DYNAMIC_CLASSES))
        self.device = device
        self.model = None

    def load_model(self):
        """Loads YOLO model on requested device."""
        if self.model is not None:
            return self.model

        try:
            from ultralytics import YOLO
            import torch

            # Determine actual device
            dev = self.device
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"[YOLO] Loading model '{self.model_path}' on device '{dev}'...")
            self.model = YOLO(self.model_path)
            self.target_device = dev
            logger.info("[YOLO] Model loaded successfully.")
            return self.model
        except Exception as e:
            logger.error(f"[YOLO] Failed to load YOLO model: {e}")
            raise

    def detect(self, image_input: Union[str, np.ndarray]) -> DetectionResult:
        """
        Runs object detection on image path or numpy array.
        """
        import time

        self.load_model()

        start_t = time.perf_counter()
        results = self.model.predict(
            source=image_input,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.target_device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        detections: List[Detection] = []
        dynamic_count = 0
        frame_name = str(image_input) if isinstance(image_input, (str, Path)) else "numpy_array"

        if results and len(results) > 0:
            res = results[0]
            names = res.names  # dict id -> name
            boxes = res.boxes

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = str(names.get(cls_id, f"class_{cls_id}")).lower()

                is_dyn = cls_name in self.dynamic_classes
                if is_dyn:
                    dynamic_count += 1

                detections.append(
                    Detection(
                        box=[round(c, 2) for c in xyxy],
                        confidence=round(conf, 4),
                        class_id=cls_id,
                        class_name=cls_name,
                        is_dynamic=is_dyn,
                    )
                )

        return DetectionResult(
            frame_path=frame_name,
            detections=detections,
            dynamic_count=dynamic_count,
            processing_time_ms=round(elapsed_ms, 2),
        )

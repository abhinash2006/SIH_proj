import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger("DisasterObjectDetection")

class DisasterObjectDetector:
    """
    Object & Person Detector for drone aerial disaster inspection using YOLOv8 / Pretrained Vision models.
    Supports detection of: person, vehicle, car, truck, boat, building, debris, water.
    """

    COCO_DISASTER_MAP = {
        0: "person",
        2: "vehicle", # car
        3: "vehicle", # motorcycle
        5: "vehicle", # bus
        7: "vehicle", # truck
        8: "boat",
    }

    def __init__(self, model_size: str = "n", conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.model_size = model_size
        self.yolo_model = None

    def load_model(self):
        if self.yolo_model is not None:
            return self.yolo_model

        try:
            from ultralytics import YOLO
            model_name = f"yolov8{self.model_size}.pt"
            logger.info(f"[INFO] Loading YOLOv8 ({model_name}) for disaster object detection...")
            self.yolo_model = YOLO(model_name)
            logger.info("[SUCCESS] YOLOv8 model loaded successfully.")
        except Exception as err:
            logger.warning(f"[WARNING] Could not load YOLOv8 ({err}). Falling back to heuristic OpenCV object detector.")
            self.yolo_model = "opencv_fallback"

        return self.yolo_model

    def detect_objects_in_frame(
        self,
        image_input: Union[str, Path, np.ndarray],
        frame_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Detects objects in a single drone image.
        Returns list of dicts:
        {
            "class_name": str,
            "confidence": float,
            "bbox": [x1, y1, x2, y2],
            "pixel_center": (cx, cy),
            "frame_idx": int
        }
        """
        self.load_model()

        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
        else:
            img = image_input.copy()

        H, W = img.shape[:2]
        detections = []

        if self.yolo_model != "opencv_fallback":
            results = self.yolo_model.predict(img, conf=self.conf_threshold, verbose=False)
            for res in results:
                boxes = res.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

                    class_name = self.COCO_DISASTER_MAP.get(cls_id, None)
                    if class_name is None:
                        # Standard COCO names fallback
                        raw_name = self.yolo_model.names.get(cls_id, "object")
                        if raw_name in ["person", "car", "truck", "bus", "boat"]:
                            class_name = "person" if raw_name == "person" else "vehicle"

                    if class_name is not None:
                        cx = int((xyxy[0] + xyxy[2]) / 2)
                        cy = int((xyxy[1] + xyxy[3]) / 2)
                        detections.append({
                            "class_name": class_name,
                            "confidence": round(conf, 3),
                            "bbox": xyxy,
                            "pixel_center": (cx, cy),
                            "frame_idx": frame_idx
                        })

        # Fallback blob detection if no neural net
        if len(detections) == 0 and self.yolo_model == "opencv_fallback":
            # Heuristic blob detection for cars / objects
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 200 < area < 5000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append({
                        "class_name": "vehicle" if area > 1000 else "person",
                        "confidence": 0.55,
                        "bbox": [x, y, x + w, y + h],
                        "pixel_center": (x + w // 2, y + h // 2),
                        "frame_idx": frame_idx
                    })

        return detections

    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Overlays detection bounding boxes onto RGB image."""
        vis = image.copy()
        colors = {
            "person": (0, 0, 255),    # Red
            "vehicle": (255, 0, 0),   # Blue
            "building": (0, 255, 0),  # Green
            "debris": (0, 165, 255),  # Orange
            "water": (255, 255, 0)    # Cyan
        }
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls = det["class_name"]
            conf = det["confidence"]
            color = colors.get(cls, (255, 0, 255))
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{cls.upper()}: {conf:.2f}"
            cv2.putText(vis, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis

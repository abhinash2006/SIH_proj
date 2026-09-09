import os
import gc
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional

import numpy as np
import cv2
import torch
from PIL import Image

logger = logging.getLogger("YOLODetector")

class YOLODetector:
    """
    Isolated wrapper for Ultralytics YOLO 2D Object Perception & Tracking Layer.
    Supports lightweight models (yolo26n.pt / yolo11n.pt), coordinate scaling back
    to original image resolution, tiled multi-scale inference, tracking, and GPU memory cleanup.
    """

    COCO_CLASS_NAMES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        8: "boat"
    }

    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "auto",
        half: bool = True,
        classes: Optional[List[int]] = None,
        tiled_inference: bool = False,
        tile_size: int = 1024,
        tile_overlap: float = 0.20
    ):
        self.model_name = model_name
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device_setting = device
        self.half = half
        self.classes = classes
        self.tiled_inference = tiled_inference
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        
        self.model = None
        self.device = self._resolve_device(device)
        self.last_detections: List[Dict[str, Any]] = []

    def _resolve_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def load_model(self, model_path: Optional[str] = None):
        """Loads the specified Ultralytics YOLO model onto target device."""
        if self.model is not None:
            return self.model

        target_model = model_path or self.model_name
        try:
            from ultralytics import YOLO
            logger.info(f"[INFO] Loading Ultralytics YOLO model '{target_model}' on device {self.device}...")
            self.model = YOLO(target_model)
            self.model.to(self.device)
            logger.info(f"[SUCCESS] Ultralytics YOLO model '{target_model}' loaded successfully.")
        except Exception as err:
            logger.warning(f"[WARNING] Could not load requested YOLO model '{target_model}': {err}")
            # Fallback to yolo11n.pt if yolo26n.pt failed
            if "yolo26n" in target_model:
                fallback = "yolo11n.pt"
                logger.info(f"[INFO] Attempting fallback to '{fallback}'...")
                from ultralytics import YOLO
                self.model = YOLO(fallback)
                self.model.to(self.device)
                self.model_name = fallback
                logger.info(f"[SUCCESS] Fallback model '{fallback}' loaded successfully.")
            else:
                raise err

        if self.model is not None and self.device.type == "cuda" and self.half:
            try:
                if hasattr(self.model, "model") and hasattr(self.model.model, "half"):
                    self.model.model.half()
            except Exception:
                pass

        return self.model

    def detect_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        frame_id: str = "frame_0000.jpg",
        frame_idx: int = 0,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Runs YOLO object detection on a single image.
        Preserves ORIGINAL image resolution coordinates.
        """
        self.load_model()
        conf_thresh = conf if conf is not None else self.confidence
        input_sz = imgsz if imgsz is not None else self.imgsz

        # Load image array
        if isinstance(image_input, (str, Path)):
            img_path_str = str(image_input)
            img = cv2.imread(img_path_str)
            if img is None:
                logger.error(f"[ERROR] Could not read image at path: {img_path_str}")
                return []
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_input.copy()

        H_orig, W_orig = img_rgb.shape[:2]
        raw_records = []

        # 1. Full-frame inference
        results = self.model.predict(
            source=img_rgb,
            conf=conf_thresh,
            iou=self.iou_threshold,
            imgsz=input_sz,
            device=self.device,
            classes=self.classes,
            verbose=False
        )
        
        self._parse_results(
            results,
            raw_records,
            frame_id=frame_id,
            frame_idx=frame_idx,
            W_orig=W_orig,
            H_orig=H_orig,
            offset_x=0,
            offset_y=0
        )

        # 2. Tiled multi-scale inference for small drone objects
        if self.tiled_inference and (H_orig > self.tile_size or W_orig > self.tile_size):
            tiles = self._generate_tiles(W_orig, H_orig, self.tile_size, self.tile_overlap)
            for tx1, ty1, tx2, ty2 in tiles:
                crop = img_rgb[ty1:ty2, tx1:tx2]
                t_results = self.model.predict(
                    source=crop,
                    conf=conf_thresh,
                    iou=self.iou_threshold,
                    imgsz=min(input_sz, max(tx2 - tx1, ty2 - ty1)),
                    device=self.device,
                    classes=self.classes,
                    verbose=False
                )
                self._parse_results(
                    t_results,
                    raw_records,
                    frame_id=frame_id,
                    frame_idx=frame_idx,
                    W_orig=W_orig,
                    H_orig=H_orig,
                    offset_x=tx1,
                    offset_y=ty1
                )

        # 3. Non-Maximum Suppression (NMS) across full & tiled detections
        nms_records = self._apply_nms(raw_records, iou_thresh=self.iou_threshold)
        
        # 4. Anti-Boat False Positive Filter for Flooded Houses
        validated_records = self._apply_anti_boat_filter(nms_records)

        self.last_detections = validated_records
        return validated_records

    def detect_frames(
        self,
        frame_list: List[Union[str, Path, np.ndarray]],
        frame_ids: Optional[List[str]] = None,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """Runs detection sequentially across a list of frames."""
        all_detections = []
        for idx, f_item in enumerate(frame_list):
            f_id = frame_ids[idx] if frame_ids and idx < len(frame_ids) else f"frame_{idx:04d}.jpg"
            dets = self.detect_image(f_item, frame_id=f_id, frame_idx=idx, conf=conf, imgsz=imgsz)
            all_detections.append(dets)
        return all_detections

    def track_frames(
        self,
        frame_list: List[Union[str, Path, np.ndarray]],
        frame_ids: Optional[List[str]] = None,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Runs object tracking with persistent ID tracking across consecutive frames.
        Uses Ultralytics model.track(..., persist=True).
        """
        self.load_model()
        conf_thresh = conf if conf is not None else self.confidence
        input_sz = imgsz if imgsz is not None else self.imgsz
        use_half = self.half and self.device.type == "cuda"

        all_tracked_detections = []
        
        for idx, f_item in enumerate(frame_list):
            f_id = frame_ids[idx] if frame_ids and idx < len(frame_ids) else f"frame_{idx:04d}.jpg"
            
            if isinstance(f_item, (str, Path)):
                img = cv2.imread(str(f_item))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = f_item.copy()

            H_orig, W_orig = img_rgb.shape[:2]
            
            try:
                results = self.model.track(
                    source=img_rgb,
                    conf=conf_thresh,
                    iou=self.iou_threshold,
                    imgsz=input_sz,
                    device=self.device,
                    classes=self.classes,
                    persist=True,
                    verbose=False
                )
            except Exception as err:
                logger.warning(f"[WARNING] YOLO tracking failed on frame {idx}: {err}. Falling back to predict.")
                results = self.model.predict(
                    source=img_rgb, conf=conf_thresh, imgsz=input_sz, device=self.device, verbose=False
                )

            raw_records = []
            self._parse_results(
                results,
                raw_records,
                frame_id=f_id,
                frame_idx=idx,
                W_orig=W_orig,
                H_orig=H_orig,
                offset_x=0,
                offset_y=0
            )

            validated_records = self._apply_anti_boat_filter(raw_records)
            all_tracked_detections.append(validated_records)

        return all_tracked_detections

    def _parse_results(
        self,
        results,
        out_records: List[Dict[str, Any]],
        frame_id: str,
        frame_idx: int,
        W_orig: int,
        H_orig: int,
        offset_x: int = 0,
        offset_y: int = 0
    ):
        for res in results:
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                raw_conf = float(box.conf[0].item())
                track_id = int(box.id[0].item()) if box.id is not None else None

                xyxy = box.xyxy[0].cpu().numpy().astype(float)
                x1_val, y1_val, x2_val, y2_val = xyxy[0] + offset_x, xyxy[1] + offset_y, xyxy[2] + offset_x, xyxy[3] + offset_y

                # Clip coordinates to original image bounds
                x1 = int(np.clip(x1_val, 0, W_orig - 1))
                y1 = int(np.clip(y1_val, 0, H_orig - 1))
                x2 = int(np.clip(x2_val, 0, W_orig - 1))
                y2 = int(np.clip(y2_val, 0, H_orig - 1))

                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                cx = int(x1 + bw / 2.0)
                cy = int(y1 + bh / 2.0)

                raw_class_name = self.model.names.get(cls_id, f"class_{cls_id}")
                
                # Standardize class names: person -> PERSON_DETECTED
                if raw_class_name == "person":
                    class_name = "person"
                    detection_label = "PERSON_DETECTED"
                elif raw_class_name in ["car", "truck", "bus", "motorcycle", "bicycle"]:
                    class_name = "vehicle"
                    detection_label = f"VEHICLE_{raw_class_name.upper()}"
                else:
                    class_name = raw_class_name
                    detection_label = raw_class_name.upper()

                rec = {
                    "frame_id": frame_id,
                    "frame_idx": frame_idx,
                    "source_frame": frame_id,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "detection_label": detection_label,
                    "raw_class_name": raw_class_name,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_center": [cx, cy],
                    "pixel_center": (cx, cy),
                    "bbox": [x1, y1, x2, y2],  # Legacy compatibility key
                    "bbox_width": bw,
                    "bbox_height": bh,
                    "image_width": W_orig,
                    "image_height": H_orig,
                    "raw_detector_confidence": round(raw_conf, 4),
                    "validated_detection_confidence": round(raw_conf, 4),
                    "confidence": round(raw_conf, 4),  # Standard key
                    "track_id": track_id,
                    "location_status": "UNLOCALIZED",
                    "3d_position": None,
                    "x_m": None, "y_m": None, "z_m": None,
                    "localization_confidence": None,
                    "reprojection_error_px": None
                }

                out_records.append(rec)

    def _generate_tiles(self, W: int, H: int, tile_sz: int, overlap: float) -> List[Tuple[int, int, int, int]]:
        tiles = []
        step = int(tile_sz * (1.0 - overlap))
        
        y_starts = list(range(0, H - tile_sz + 1, step))
        if not y_starts or y_starts[-1] + tile_sz < H:
            y_starts.append(max(0, H - tile_sz))

        x_starts = list(range(0, W - tile_sz + 1, step))
        if not x_starts or x_starts[-1] + tile_sz < W:
            x_starts.append(max(0, W - tile_sz))

        for ty in y_starts:
            for tx in x_starts:
                tiles.append((tx, ty, min(W, tx + tile_sz), min(H, ty + tile_sz)))

        return tiles

    def _apply_nms(self, records: List[Dict[str, Any]], iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
        if not records:
            return []

        boxes = np.array([r["bbox_xyxy"] for r in records], dtype=np.float32)
        scores = np.array([r["raw_detector_confidence"] for r in records], dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]

        return [records[k] for k in keep]

    def _apply_anti_boat_filter(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters out false 'boat' predictions caused by COCO YOLO misclassifying flooded house roofs.
        Keeps original raw_detector_confidence but flags/remaps invalid detections.
        """
        out = []
        for r in records:
            if r.get("class_id") == 8 or r.get("raw_class_name") == "boat":
                bw = r["bbox_width"]
                bh = r["bbox_height"]
                box_area = bw * bh
                aspect_ratio = float(bw) / float(bh + 1e-6)

                # Flooded houses typically have large pixel area (>2000px) or roof aspect ratio (0.5 to 2.0)
                if box_area > 2000 or (0.5 <= aspect_ratio <= 2.0 and box_area > 800):
                    logger.info(f"[ANTI-CONFUSION] False boat detected on house roof (area={box_area}px) -> Flagged as INVALID_BOAT_DETECTION")
                    r["validated_detection_confidence"] = 0.0
                    r["location_status"] = "FALSE_POSITIVE_DETECTION"
                    r["validation_note"] = "COCO boat misclassification on flooded structure roof."
                    # Do not pass false boat into disaster incidents
                    continue
                else:
                    r["validated_detection_confidence"] = r["raw_detector_confidence"]
            out.append(r)
        return out

    def annotate_frame(
        self,
        image_input: Union[str, Path, np.ndarray],
        detections: List[Dict[str, Any]],
        show_3d_status: bool = True
    ) -> np.ndarray:
        """Draws bounding boxes, class labels, confidence, track IDs, and 3D status on frame."""
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
        else:
            img_bgr = cv2.cvtColor(image_input.copy(), cv2.COLOR_RGB2BGR)

        colors = {
            "person": (0, 0, 255),       # Red
            "vehicle": (255, 128, 0),    # Orange
            "building": (0, 255, 0),     # Green
            "boat": (255, 255, 0)        # Cyan
        }

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            cls_name = det["class_name"]
            conf = det.get("validated_detection_confidence", det.get("raw_detector_confidence", 0.0))
            track_id = det.get("track_id")
            loc_status = det.get("location_status", "UNLOCALIZED")

            color = colors.get(cls_name, (0, 255, 255))
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

            track_str = f" TRK:{track_id}" if track_id is not None else ""
            label_text = f"{cls_name.upper()} {conf:.2f}{track_str}"
            
            if show_3d_status and loc_status == "LOCALIZED_3D":
                x3d, y3d, z3d = det.get("x_m"), det.get("y_m"), det.get("z_m")
                if x3d is not None:
                    label_text += f" (3D: {x3d:.1f},{y3d:.1f},{z3d:.1f}m)"

            cv2.putText(img_bgr, label_text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def release(self):
        """Releases YOLO model from GPU memory to conserve VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("[INFO] Released YOLO model from GPU memory.")

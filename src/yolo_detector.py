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
        confidence: float = 0.08,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "auto",
        half: bool = True,
        classes: Optional[List[int]] = None,
        tiled_inference: bool = True,
        tile_size: int = 640,
        tile_overlap: float = 0.25,
        person_conf_threshold: float = 0.08,
        person_iou_threshold: float = 0.40
    ):
        self.model_name = model_name
        self.confidence = confidence
        self.person_conf_threshold = person_conf_threshold
        self.iou_threshold = iou_threshold
        self.person_iou_threshold = person_iou_threshold
        self.imgsz = imgsz
        self.device_setting = device
        self.half = half
        self.classes = classes if classes is not None else [0, 1, 2, 3, 5, 7, 8]
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
        p_thresh = getattr(self, "person_conf_threshold", 0.08)
        run_conf = min(conf_thresh, p_thresh)

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
            conf=run_conf,
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
            offset_y=0,
            conf_thresh=conf_thresh,
            person_thresh=p_thresh
        )

        # 2. Tiled multi-scale inference for small drone objects
        if self.tiled_inference and (H_orig > self.tile_size or W_orig > self.tile_size):
            tiles = self._generate_tiles(W_orig, H_orig, self.tile_size, self.tile_overlap)
            for tx1, ty1, tx2, ty2 in tiles:
                crop = img_rgb[ty1:ty2, tx1:tx2]
                t_results = self.model.predict(
                    source=crop,
                    conf=run_conf,
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
                    offset_y=ty1,
                    conf_thresh=conf_thresh,
                    person_thresh=p_thresh
                )

        # 3. Class-aware Non-Maximum Suppression (NMS) across full & tiled detections
        nms_records = self._apply_nms(
            raw_records,
            iou_thresh=self.iou_threshold,
            person_iou_thresh=getattr(self, "person_iou_threshold", 0.40)
        )
        
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
        Evaluates detections using tiled multi-scale inference to detect small aerial
        objects/persons at drone resolution, and maintains persistent track IDs
        using spatial & IoU association across consecutive frames.
        """
        self.load_model()
        conf_thresh = conf if conf is not None else self.confidence
        input_sz = imgsz if imgsz is not None else self.imgsz

        all_tracked_detections = []
        active_tracks: Dict[int, Dict[str, Any]] = {}
        next_track_id = 1

        for idx, f_item in enumerate(frame_list):
            f_id = frame_ids[idx] if frame_ids and idx < len(frame_ids) else f"frame_{idx:04d}.jpg"

            # 1. Run detection on this frame (full-frame + tiled inference + NMS + anti-boat filter)
            frame_dets = self.detect_image(
                f_item,
                frame_id=f_id,
                frame_idx=idx,
                conf=conf_thresh,
                imgsz=input_sz
            )

            if not frame_dets:
                all_tracked_detections.append([])
                continue

            # 2. Match with active tracks across consecutive frames
            if not active_tracks:
                # First frame with detections: initialize tracks
                for det in frame_dets:
                    trk_id = next_track_id
                    next_track_id += 1
                    det["track_id"] = trk_id
                    det["id"] = f"TRK_{trk_id}"
                    det["detection_id"] = f"TRK_{trk_id}"
                    det["track_length"] = 1
                    det["tracking_confidence"] = 0.55
                    active_tracks[trk_id] = {
                        "bbox": det["bbox_xyxy"],
                        "center": det["pixel_center"],
                        "class_name": det.get("class_name", "object"),
                        "last_frame_idx": idx,
                        "track_length": 1
                    }
            else:
                # Match new detections against active tracks
                matched_tracks = set()
                matched_dets = set()

                # Build candidate pairs (score, trk_id, det_idx)
                candidates = []
                for trk_id, trk_info in active_tracks.items():
                    frames_missed = idx - trk_info["last_frame_idx"]
                    # Maintain tracks when missed for up to 5 consecutive frames
                    if frames_missed > 5:
                        continue

                    tb = trk_info["bbox"]
                    tc = trk_info["center"]
                    t_cls = trk_info["class_name"]

                    for d_i, det in enumerate(frame_dets):
                        d_cls = det.get("class_name", "object")
                        db = det["bbox_xyxy"]
                        dc = det["pixel_center"]

                        # Compute IoU
                        x_left = max(tb[0], db[0])
                        y_top = max(tb[1], db[1])
                        x_right = min(tb[2], db[2])
                        y_bottom = min(tb[3], db[3])

                        if x_right > x_left and y_bottom > y_top:
                            intersection = (x_right - x_left) * (y_bottom - y_top)
                            area1 = (tb[2] - tb[0]) * (tb[3] - tb[1])
                            area2 = (db[2] - db[0]) * (db[3] - db[1])
                            union = area1 + area2 - intersection
                            iou = intersection / max(1.0, union)
                        else:
                            iou = 0.0

                        # Compute centroid distance
                        dist = float(np.hypot(tc[0] - dc[0], tc[1] - dc[1]))
                        is_same_class = (d_cls == t_cls)

                        if t_cls == "person" and d_cls == "person":
                            # UAV camera motion tolerance up to 250px + extra tolerance per missed frame
                            max_radius = 250.0 + 30.0 * max(0, frames_missed - 1)
                            if dist <= max_radius:
                                dist_score = max(0.0, 1.0 - (dist / max_radius))
                                score = max(iou, dist_score) + 0.25
                                candidates.append((score, trk_id, d_i))
                        else:
                            # Standard objects
                            dist_score = max(0.0, 1.0 - (dist / 200.0))
                            if not is_same_class and (iou < 0.20 and dist > 45.0):
                                continue
                            score = max(iou, 0.5 * dist_score)
                            if is_same_class:
                                score += 0.15
                            if score > 0.15:
                                candidates.append((score, trk_id, d_i))

                # Sort by score descending
                candidates.sort(key=lambda x: x[0], reverse=True)

                for score, trk_id, d_i in candidates:
                    if trk_id in matched_tracks or d_i in matched_dets:
                        continue
                    matched_tracks.add(trk_id)
                    matched_dets.add(d_i)

                    det = frame_dets[d_i]
                    det["track_id"] = trk_id
                    det["id"] = f"TRK_{trk_id}"
                    det["detection_id"] = f"TRK_{trk_id}"

                    new_len = trk_info.get("track_length", 1) + 1
                    trk_info["track_length"] = new_len
                    trk_info["bbox"] = det["bbox_xyxy"]
                    trk_info["center"] = det["pixel_center"]
                    trk_info["last_frame_idx"] = idx

                    trk_conf = min(1.0, round(0.4 + 0.15 * min(new_len, 4), 3))
                    det["track_length"] = new_len
                    det["tracking_confidence"] = trk_conf

                # Assign new track IDs to unmatched detections
                for d_i, det in enumerate(frame_dets):
                    if d_i not in matched_dets:
                        trk_id = next_track_id
                        next_track_id += 1
                        det["track_id"] = trk_id
                        det["id"] = f"TRK_{trk_id}"
                        det["detection_id"] = f"TRK_{trk_id}"
                        det["track_length"] = 1
                        det["tracking_confidence"] = 0.55
                        active_tracks[trk_id] = {
                            "bbox": det["bbox_xyxy"],
                            "center": det["pixel_center"],
                            "class_name": det.get("class_name", "object"),
                            "last_frame_idx": idx,
                            "track_length": 1
                        }

            all_tracked_detections.append(frame_dets)

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
        offset_y: int = 0,
        conf_thresh: float = 0.15,
        person_thresh: float = 0.08
    ):
        for res in results:
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                raw_conf = float(box.conf[0].item())
                track_id = int(box.id[0].item()) if box.id is not None else None
                raw_class_name = self.model.names.get(cls_id, f"class_{cls_id}")

                # Enforce class threshold before building record
                if raw_class_name == "person":
                    if raw_conf < person_thresh:
                        continue
                else:
                    if raw_conf < conf_thresh:
                        continue

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

                source_tile = "full_frame" if (offset_x == 0 and offset_y == 0) else [offset_x, offset_y, min(W_orig, offset_x + self.tile_size), min(H_orig, offset_y + self.tile_size)]
                tile_id = "full_frame" if (offset_x == 0 and offset_y == 0) else f"tile_{offset_x}_{offset_y}"
                tile_coords = [0, 0, W_orig, H_orig] if (offset_x == 0 and offset_y == 0) else [offset_x, offset_y, min(W_orig, offset_x + self.tile_size), min(H_orig, offset_y + self.tile_size)]

                raw_yolo_snapshot = {
                    "frame_id": frame_id,
                    "track_id": track_id,
                    "class": raw_class_name,
                    "confidence": round(raw_conf, 4),
                    "bbox": [x1, y1, x2, y2],
                    "bbox_width": bw,
                    "bbox_height": bh,
                    "bbox_area": bw * bh,
                    "source_tile": source_tile,
                    "original_image_coordinates": [x1, y1, x2, y2]
                }

                raw_person_record = None
                if raw_class_name == "person":
                    raw_person_record = {
                        "frame_id": frame_id,
                        "timestamp": round(frame_idx / 25.0, 3) if frame_idx is not None else None,
                        "bbox": [x1, y1, x2, y2],
                        "center_x": cx,
                        "center_y": cy,
                        "width": bw,
                        "height": bh,
                        "bbox_area": bw * bh,
                        "confidence": round(raw_conf, 4),
                        "class": "PERSON",
                        "tile_id": tile_id,
                        "tile_coordinates": tile_coords,
                        "original_frame_coordinates": [x1, y1, x2, y2]
                    }

                det_uid = f"TRK_{track_id}" if track_id is not None else f"DET_{frame_idx:04d}_{len(out_records)+1:03d}"
                rec = {
                    "id": det_uid,
                    "detection_id": det_uid,
                    "frame_id": frame_id,
                    "frame_idx": frame_idx,
                    "source_frame": frame_id,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "detection_label": detection_label,
                    "raw_class_name": raw_class_name,
                    "raw_detector_confidence": round(raw_conf, 4),
                    "raw_yolo_confidence": round(raw_conf, 4),
                    "raw_yolo_output": raw_yolo_snapshot,
                    "raw_person_record": raw_person_record,
                    "tracking_confidence": None,
                    "person_validation_confidence": round(raw_conf, 4) if raw_class_name == "person" else None,
                    "confidence_3d": None,
                    "rescue_confidence": None,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_center": [cx, cy],
                    "pixel_center": (cx, cy),
                    "center": (cx, cy),
                    "bbox": [x1, y1, x2, y2],  # Legacy compatibility key
                    "bbox_width": bw,
                    "bbox_height": bh,
                    "bbox_area": bw * bh,
                    "source_tile": source_tile,
                    "original_image_coordinates": [x1, y1, x2, y2],
                    "image_width": W_orig,
                    "image_height": H_orig,
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

    def _apply_nms(
        self,
        records: List[Dict[str, Any]],
        iou_thresh: float = 0.45,
        person_iou_thresh: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Class-aware NMS:
        - Evaluates person detections in an independent NMS pool with person_iou_thresh (default 0.40).
        - Large vehicles or roofs never suppress small persons.
        - Preserves person detections across minimally overlapping tiles.
        """
        if not records:
            return []

        p_thresh = person_iou_thresh if person_iou_thresh is not None else getattr(self, "person_iou_threshold", 0.40)

        person_recs = [r for r in records if r.get("raw_class_name") == "person" or r.get("class_name") == "person"]
        other_recs = [r for r in records if not (r.get("raw_class_name") == "person" or r.get("class_name") == "person")]

        def _nms_subset(sub_records: List[Dict[str, Any]], thresh: float) -> List[Dict[str, Any]]:
            if not sub_records:
                return []
            boxes = np.array([r["bbox_xyxy"] for r in sub_records], dtype=np.float32)
            scores = np.array([r["raw_detector_confidence"] for r in sub_records], dtype=np.float32)

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

                inds = np.where(ovr <= thresh)[0]
                order = order[inds + 1]

            return [sub_records[k] for k in keep]

        kept_person = _nms_subset(person_recs, p_thresh)
        kept_other = _nms_subset(other_recs, iou_thresh)
        return kept_person + kept_other

    def _apply_anti_boat_filter(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters out false 'boat' predictions caused by COCO YOLO misclassifying flooded house roofs.
        Strictly preserves raw_class_name and raw_detector_confidence immutable.
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
                    logger.info(f"[ANTI-CONFUSION] False boat detected on house roof (area={box_area}px) -> Flagged building")
                    r["class_name"] = "building"
                    # Preserve raw_class_name! DO NOT overwrite raw prediction.
                    r["validated_detection_confidence"] = 0.0
                    r["validation_note"] = "COCO boat misclassification remapped to flooded structure roof."
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
        return self.annotate_validated_frame(image_input, detections, show_3d_status=show_3d_status)

    def annotate_raw_frame(
        self,
        image_input: Union[str, Path, np.ndarray],
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Draws raw unvalidated YOLO bounding boxes, raw classes, and raw confidences."""
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
        else:
            img_bgr = cv2.cvtColor(image_input.copy(), cv2.COLOR_RGB2BGR)

        colors = {
            "person": (0, 0, 255),       # Red
            "vehicle": (255, 128, 0),    # Orange
            "car": (255, 128, 0),
            "truck": (255, 128, 0),
            "bus": (255, 128, 0),
            "building": (0, 255, 0),     # Green
            "boat": (255, 255, 0)        # Cyan
        }

        # Header banner
        cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 36), (20, 20, 20), -1)
        cv2.putText(img_bgr, "RAW YOLO PREDICTIONS (Unvalidated Model Output)", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        for det in detections:
            bbox = det.get("bbox_xyxy", det.get("bbox", [0, 0, 0, 0]))
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            raw_cls = str(det.get("raw_class_name", det.get("class_name", "object"))).lower()
            raw_conf = float(det.get("raw_detector_confidence", det.get("confidence", 0.0)))
            track_id = det.get("track_id")

            color = colors.get(raw_cls, (0, 255, 255))
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

            trk_str = f" TRK:{track_id}" if track_id is not None else ""
            lbl = f"{raw_cls.upper()} {raw_conf:.2f}{trk_str}"
            cv2.putText(img_bgr, lbl, (x1, max(45, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def annotate_validated_frame(
        self,
        image_input: Union[str, Path, np.ndarray],
        detections: List[Dict[str, Any]],
        show_3d_status: bool = True
    ) -> np.ndarray:
        """Draws validated classifications, flagging UNCERTAIN_OBJECT and remapped classes."""
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
        else:
            img_bgr = cv2.cvtColor(image_input.copy(), cv2.COLOR_RGB2BGR)

        colors = {
            "PERSON": (0, 0, 255),          # Red
            "VEHICLE": (255, 128, 0),       # Orange
            "BUILDING": (0, 255, 0),        # Green
            "BOAT": (255, 255, 0),          # Cyan
            "UNCERTAIN_OBJECT": (255, 0, 255) # Magenta
        }

        # Header banner
        cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], 36), (20, 20, 20), -1)
        cv2.putText(img_bgr, "FINAL VALIDATED CLASSIFICATIONS (Context & Temporal Gating)", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        for det in detections:
            bbox = det.get("bbox_xyxy", det.get("bbox", [0, 0, 0, 0]))
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            val_cls = str(det.get("final_class", det.get("class_name", "UNCERTAIN_OBJECT"))).upper()
            val_conf = float(det.get("validation_confidence", det.get("validated_detection_confidence", det.get("confidence", 0.0))))
            track_id = det.get("track_id")
            is_changed = det.get("is_changed", False)

            color = colors.get(val_cls, (255, 0, 255))
            thickness = 3 if is_changed else 2
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

            trk_str = f" TRK:{track_id}" if track_id is not None else ""
            lbl = f"{val_cls} {val_conf:.2f}{trk_str}"
            if is_changed:
                lbl += " [REMAPPED]"

            cv2.putText(img_bgr, lbl, (x1, max(45, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def create_diagnostic_comparison(
        self,
        image_input: Union[str, Path, np.ndarray],
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Creates a side-by-side visual diagnostic comparison image:
        Left: Raw YOLO Output
        Right: Final Validated Output
        Bottom: Audit Explanation Changelog
        """
        if isinstance(image_input, (str, Path)):
            base_bgr = cv2.imread(str(image_input))
        else:
            base_bgr = cv2.cvtColor(image_input.copy(), cv2.COLOR_RGB2BGR)

        raw_rgb = self.annotate_raw_frame(base_bgr, detections)
        val_rgb = self.annotate_validated_frame(base_bgr, detections)

        raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
        val_bgr = cv2.cvtColor(val_rgb, cv2.COLOR_RGB2BGR)

        # Scale down for side-by-side if wide
        H, W = raw_bgr.shape[:2]
        target_w = 960
        target_h = int(H * (target_w / float(W)))
        left_pane = cv2.resize(raw_bgr, (target_w, target_h))
        right_pane = cv2.resize(val_bgr, (target_w, target_h))

        # Horizontal stack
        side_by_side = np.hstack([left_pane, right_pane])

        # Bottom audit log panel
        changed_items = [d for d in detections if d.get("is_changed", False)]
        panel_h = max(80, 30 + len(changed_items) * 24)
        footer = np.full((panel_h, side_by_side.shape[1], 3), 30, dtype=np.uint8)

        cv2.putText(footer, f"AERIAL DETECTION AUDIT CHANGELOG ({len(changed_items)} remapped / uncertain objects):", (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        for idx, item in enumerate(changed_items[:8]):
            raw_c = item.get("raw_class", item.get("raw_class_name", "object"))
            raw_cf = float(item.get("raw_confidence", item.get("raw_detector_confidence", 0.0)))
            fin_c = item.get("final_class", item.get("class_name", "object"))
            fin_cf = float(item.get("validation_confidence", 0.0))
            reason = item.get("validation_evidence", "Context remapping")
            trk = item.get("track_id", "N/A")

            log_line = f"• [TRK {trk}] RAW: {raw_c.upper()} {raw_cf:.2f} -> FINAL: {fin_c.upper()} {fin_cf:.2f} | Reason: {reason[:100]}"
            cv2.putText(footer, log_line, (16, 46 + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        full_diag = np.vstack([side_by_side, footer])
        return cv2.cvtColor(full_diag, cv2.COLOR_BGR2RGB)

    def annotate_person_diagnostic(
        self,
        image_input: Union[str, Path, np.ndarray],
        raw_person_dets: List[Dict[str, Any]],
        tracked_person_dets: Optional[List[Dict[str, Any]]] = None,
        validated_person_dets: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Generates a 3-panel side-by-side visual inspection contact sheet showing ONLY person detections:
        - Panel 1: RAW PERSON DETECTIONS (red boxes, raw confidence, tile ID)
        - Panel 2: TRACKED PERSONS (unique color per track ID, track length, tracking confidence)
        - Panel 3: FINAL PERSONS (green boxes for validated, red dashed for rejected)
        Strictly excludes vehicles, buildings, or boats to keep inspection focused.
        """
        if isinstance(image_input, (str, Path)):
            base_bgr = cv2.imread(str(image_input))
        else:
            base_bgr = cv2.cvtColor(image_input.copy(), cv2.COLOR_RGB2BGR)

        H, W = base_bgr.shape[:2]
        panel_w = 640
        panel_h = int(H * (panel_w / float(W)))

        # 1. RAW PERSON PANEL
        p1 = cv2.resize(base_bgr.copy(), (panel_w, panel_h))
        cv2.rectangle(p1, (0, 0), (panel_w, 32), (30, 30, 30), -1)
        raw_p = [d for d in raw_person_dets if str(d.get("raw_class_name", d.get("class", ""))).lower() == "person" or str(d.get("class_name", "")).lower() == "person"]
        cv2.putText(p1, f"RAW PERSON DETECTIONS ({len(raw_p)})", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)

        scale_x = panel_w / float(W)
        scale_y = panel_h / float(H)

        for det in raw_p:
            b = det.get("bbox_xyxy", det.get("bbox", [0, 0, 0, 0]))
            x1, y1, x2, y2 = int(b[0] * scale_x), int(b[1] * scale_y), int(b[2] * scale_x), int(b[3] * scale_y)
            conf = float(det.get("raw_yolo_confidence", det.get("raw_detector_confidence", det.get("confidence", 0.0))))
            tile_id = det.get("raw_person_record", {}).get("tile_id", det.get("source_tile", "")) if det.get("raw_person_record") else det.get("source_tile", "")
            tile_str = tile_id if isinstance(tile_id, str) else "tile"
            cv2.rectangle(p1, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(p1, f"P {conf:.2f} [{tile_str[:8]}]", (x1, max(38, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)

        # 2. TRACKED PERSON PANEL
        p2 = cv2.resize(base_bgr.copy(), (panel_w, panel_h))
        cv2.rectangle(p2, (0, 0), (panel_w, 32), (30, 30, 30), -1)
        tracked_p = [d for d in (tracked_person_dets if tracked_person_dets is not None else raw_person_dets) if str(d.get("raw_class_name", d.get("class", ""))).lower() == "person" or str(d.get("class_name", "")).lower() == "person"]
        cv2.putText(p2, f"TRACKED PERSONS ({len(tracked_p)})", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

        def _trk_color(t_id):
            if t_id is None:
                return (200, 200, 200)
            val = int(t_id) * 47 % 255
            return (int(val), int((val * 2) % 255), int((255 - val)))

        for det in tracked_p:
            b = det.get("bbox_xyxy", det.get("bbox", [0, 0, 0, 0]))
            x1, y1, x2, y2 = int(b[0] * scale_x), int(b[1] * scale_y), int(b[2] * scale_x), int(b[3] * scale_y)
            trk_id = det.get("track_id")
            t_len = det.get("track_length", 1)
            t_conf = det.get("tracking_confidence", 0.55)
            col = _trk_color(trk_id)
            cv2.rectangle(p2, (x1, y1), (x2, y2), col, 2)
            cv2.circle(p2, ((x1 + x2) // 2, (y1 + y2) // 2), 3, col, -1)
            cv2.putText(p2, f"TRK_{trk_id} L:{t_len} C:{t_conf:.2f}", (x1, max(38, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        # 3. FINAL VALIDATED PERSON PANEL
        p3 = cv2.resize(base_bgr.copy(), (panel_w, panel_h))
        cv2.rectangle(p3, (0, 0), (panel_w, 32), (30, 30, 30), -1)
        val_source = validated_person_dets if validated_person_dets is not None else tracked_p
        val_p = [d for d in val_source if str(d.get("raw_class_name", d.get("raw_class", d.get("class", "")))).lower() == "person" or str(d.get("final_class", "")).upper() == "PERSON"]
        valid_cnt = sum(1 for d in val_p if str(d.get("final_class", d.get("class_name", ""))).upper() == "PERSON")
        cv2.putText(p3, f"VALIDATED PERSONS ({valid_cnt}/{len(val_p)})", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)

        for det in val_p:
            b = det.get("bbox_xyxy", det.get("bbox", [0, 0, 0, 0]))
            if not b and "raw_yolo_output" in det:
                b = det["raw_yolo_output"].get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = int(b[0] * scale_x), int(b[1] * scale_y), int(b[2] * scale_x), int(b[3] * scale_y)
            f_cls = str(det.get("final_class", det.get("class_name", "PERSON"))).upper()
            val_conf = float(det.get("person_validation_confidence", det.get("validation_confidence", det.get("confidence", 0.0))))
            if f_cls == "PERSON":
                cv2.rectangle(p3, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(p3, f"PERSON {val_conf:.2f}", (x1, max(38, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
            else:
                cv2.rectangle(p3, (x1, y1), (x2, y2), (0, 0, 200), 1)
                cv2.putText(p3, f"REJ: {f_cls}", (x1, max(38, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)

        combined = np.hstack([p1, p2, p3])
        return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

    def release(self):
        """Releases YOLO model from GPU memory to conserve VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("[INFO] Released YOLO model from GPU memory.")

    detect_frame = detect_image

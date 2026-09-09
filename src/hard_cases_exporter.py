"""
Hard Cases Dataset Exporter for Domain-Specific UAV/Flood YOLO Training.

Exports challenging aerial detections (false predictions, remapped classes,
and high-confidence positive references) with image crops and JSON metadata
into data/yolo_hard_cases/ organized by error/validation category.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import cv2
import numpy as np

logger = logging.getLogger("HardCasesExporter")


class HardCasesExporter:
    """
    Organizes difficult cases from UAV inspection into 8 training preparation categories:
    - building_as_vehicle/
    - vehicle_as_building/
    - false_person/
    - false_boat/
    - correct_vehicle/
    - correct_building/
    - correct_person/
    - correct_boat/
    """

    CATEGORIES = [
        "building_as_vehicle",
        "vehicle_as_building",
        "false_person",
        "false_boat",
        "correct_vehicle",
        "correct_building",
        "correct_person",
        "correct_boat"
    ]

    @classmethod
    def export_hard_cases(
        cls,
        selected_frame_paths: List[Union[str, Path]],
        all_validated_detections: List[Dict[str, Any]],
        output_base_dir: Union[str, Path] = "data/yolo_hard_cases",
        context_margin_ratio: float = 0.15
    ) -> Dict[str, int]:
        """
        Extracts cropped detection patches and writes metadata JSONs for every category.
        """
        base_dir = Path(output_base_dir)
        category_dirs: Dict[str, Path] = {}
        for cat in cls.CATEGORIES:
            cd = base_dir / cat
            cd.mkdir(parents=True, exist_ok=True)
            category_dirs[cat] = cd

        counts: Dict[str, int] = {cat: 0 for cat in cls.CATEGORIES}

        # Cache loaded frames
        frame_cache: Dict[str, np.ndarray] = {}

        for det_idx, det in enumerate(all_validated_detections):
            raw_cls = str(det.get("raw_class", det.get("raw_class_name", "object"))).lower()
            raw_conf = float(det.get("raw_confidence", det.get("raw_detector_confidence", 0.0)))
            fin_cls = str(det.get("final_class", det.get("class_name", "UNCERTAIN_OBJECT"))).upper()
            val_conf = float(det.get("validation_confidence", det.get("confidence", 0.0)))
            evidence = str(det.get("validation_evidence", ""))
            trk_id = det.get("track_id", "N/A")
            frame_id = det.get("frame_id", "frame_0000.jpg")
            f_idx = det.get("frame_idx", 0)

            # Determine category
            category = None
            if raw_cls in ["vehicle", "car", "truck", "bus"] and fin_cls == "BUILDING":
                category = "building_as_vehicle"
            elif raw_cls == "building" and fin_cls == "VEHICLE":
                category = "vehicle_as_building"
            elif raw_cls == "person" and fin_cls in ["UNCERTAIN_OBJECT", "BUILDING", "VEHICLE"]:
                category = "false_person"
            elif raw_cls == "boat" and fin_cls in ["UNCERTAIN_OBJECT", "BUILDING", "VEHICLE"]:
                category = "false_boat"
            elif raw_cls in ["vehicle", "car", "truck", "bus"] and fin_cls == "VEHICLE" and val_conf >= 0.15:
                category = "correct_vehicle"
            elif (raw_cls == "building" or "BUILDING" in fin_cls) and fin_cls == "BUILDING" and val_conf >= 0.20:
                category = "correct_building"
            elif raw_cls == "person" and fin_cls == "PERSON" and val_conf >= 0.08:
                category = "correct_person"
            elif raw_cls == "boat" and fin_cls == "BOAT" and val_conf >= 0.15:
                category = "correct_boat"

            if category is None:
                continue

            # Load image
            img_bgr = None
            if 0 <= f_idx < len(selected_frame_paths):
                f_path = str(selected_frame_paths[f_idx])
                if f_path not in frame_cache:
                    frame_cache[f_path] = cv2.imread(f_path)
                img_bgr = frame_cache[f_path]
            elif det.get("source_frame_path") and Path(det["source_frame_path"]).exists():
                f_path = str(det["source_frame_path"])
                if f_path not in frame_cache:
                    frame_cache[f_path] = cv2.imread(f_path)
                img_bgr = frame_cache[f_path]

            if img_bgr is None:
                continue

            H, W = img_bgr.shape[:2]
            bbox = det.get("bbox", det.get("bbox_xyxy", [0, 0, 0, 0]))
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            # Apply context margin
            mx = int(bw * context_margin_ratio)
            my = int(bh * context_margin_ratio)
            cx1 = max(0, x1 - mx)
            cy1 = max(0, y1 - my)
            cx2 = min(W, x2 + mx)
            cy2 = min(H, y2 + my)

            crop = img_bgr[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            # Save crop & metadata
            file_base = f"crop_{Path(frame_id).stem}_trk{trk_id}_{det_idx:03d}"
            crop_path = category_dirs[category] / f"{file_base}.jpg"
            json_path = category_dirs[category] / f"{file_base}.json"

            cv2.imwrite(str(crop_path), crop)

            meta = {
                "frame_id": frame_id,
                "frame_idx": f_idx,
                "track_id": trk_id,
                "raw_yolo_class": raw_cls,
                "raw_yolo_confidence": raw_conf,
                "validated_class": fin_cls,
                "validation_confidence": val_conf,
                "validation_evidence": evidence,
                "original_bbox": [x1, y1, x2, y2],
                "cropped_bbox": [cx1, cy1, cx2, cy2],
                "image_width": W,
                "image_height": H,
                "category": category
            }
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(meta, jf, indent=2)

            counts[category] += 1

        logger.info(f"[HARD CASES EXPORT] Exported cases: {counts}")
        return counts

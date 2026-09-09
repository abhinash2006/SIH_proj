"""
Aerial Class Validation & Temporal Consistency Layer for UAV Disaster Intelligence.

Implements secondary physical, geometric, environmental, and multi-frame temporal
reasoning to audit and refine tiled YOLO classifications (PERSON, VEHICLE, BUILDING, BOAT, UNCERTAIN_OBJECT).
Preserves raw model outputs without overwriting.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("AerialClassifier")


@dataclass
class ClassThresholdConfig:
    """Configurable detection thresholds and geometric parameters."""
    person_thresh: float = 0.08
    vehicle_thresh: float = 0.15
    building_thresh: float = 0.15
    boat_thresh: float = 0.15

    # Person geometric boundaries
    person_min_area: float = 10.0
    person_max_area: float = 3500.0
    person_max_aspect: float = 8.0

    # Vehicle geometric boundaries
    vehicle_min_area: float = 250.0
    vehicle_max_area: float = 12000.0
    vehicle_min_aspect: float = 1.05
    vehicle_max_aspect: float = 4.5

    # Building geometric boundaries
    building_min_area: float = 2500.0
    building_min_aspect: float = 0.35
    building_max_aspect: float = 3.0

    # Boat geometric & water boundaries
    boat_min_water_overlap: float = 0.15
    boat_min_aspect: float = 1.2
    boat_max_aspect: float = 5.0

    # Temporal consistency parameters
    temporal_min_vote_ratio: float = 0.60
    temporal_switch_threshold: int = 3

    @classmethod
    def from_dict(cls, cfg_dict: Optional[Dict[str, Any]] = None) -> "ClassThresholdConfig":
        if not cfg_dict:
            return cls()
        
        ct = cfg_dict.get("class_thresholds", {})
        cv = cfg_dict.get("context_validation", {})
        
        v_ar = cv.get("vehicle_aspect_ratio_range", [1.05, 4.5])
        b_ar = cv.get("building_aspect_ratio_range", [0.35, 3.0])

        return cls(
            person_thresh=float(ct.get("person", 0.08)),
            vehicle_thresh=float(ct.get("vehicle", 0.15)),
            building_thresh=float(ct.get("building", 0.15)),
            boat_thresh=float(ct.get("boat", 0.15)),
            person_min_area=float(cv.get("person_min_area", 10.0)),
            person_max_area=float(cv.get("person_max_area", 3500.0)),
            person_max_aspect=float(cv.get("person_aspect_ratio_max", 8.0)),
            vehicle_min_area=float(cv.get("vehicle_min_area", 250.0)),
            vehicle_max_area=float(cv.get("vehicle_max_area", 12000.0)),
            vehicle_min_aspect=float(v_ar[0]),
            vehicle_max_aspect=float(v_ar[1]),
            building_min_area=float(cv.get("building_min_area", 2500.0)),
            building_min_aspect=float(b_ar[0]),
            building_max_aspect=float(b_ar[1]),
            boat_min_water_overlap=float(cv.get("boat_min_water_overlap", 0.15)),
            temporal_min_vote_ratio=float(cv.get("temporal_min_vote_ratio", 0.60)),
            temporal_switch_threshold=int(cv.get("temporal_switch_threshold", 3))
        )


class TrackClassHistory:
    """
    Accumulates multi-frame detection records for a single track_id
    and evaluates temporal consistency, dominant class, vote ratio,
    mean confidence, and class-switch count.
    """

    def __init__(self, track_id: Union[int, str]):
        self.track_id = track_id
        self.observations: List[Dict[str, Any]] = []

    def add_observation(
        self,
        frame_idx: int,
        frame_id: str,
        raw_class: str,
        raw_conf: float,
        bbox: List[int]
    ):
        self.observations.append({
            "frame_idx": frame_idx,
            "frame_id": frame_id,
            "raw_class": raw_class.lower(),
            "raw_conf": float(raw_conf),
            "bbox": list(bbox)
        })

    @property
    def total_observations(self) -> int:
        return len(self.observations)

    def compute_temporal_metrics(self) -> Dict[str, Any]:
        """Calculates dominant class, vote ratio, mean conf, and switch count."""
        if not self.observations:
            return {
                "dominant_class": "unknown",
                "class_vote_ratio": 0.0,
                "mean_class_confidence": 0.0,
                "class_switch_count": 0,
                "class_distribution": {}
            }

        class_counts: Dict[str, int] = {}
        class_confs: Dict[str, List[float]] = {}
        prev_cls = None
        switch_count = 0

        for obs in self.observations:
            raw_c = obs["raw_class"]
            # Normalize to standard macro categories (car/truck/bus -> vehicle)
            cls_name = "vehicle" if raw_c in ["car", "truck", "bus", "motorcycle", "bicycle", "vehicle"] else raw_c
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            class_confs.setdefault(cls_name, []).append(obs["raw_conf"])

            if prev_cls is not None and cls_name != prev_cls:
                switch_count += 1
            prev_cls = cls_name

        total = len(self.observations)
        dominant_cls = max(
            class_counts.keys(),
            key=lambda c: (class_counts[c], np.mean(class_confs[c]))
        )
        vote_ratio = class_counts[dominant_cls] / float(total)
        mean_conf = float(np.mean(class_confs[dominant_cls]))

        return {
            "dominant_class": dominant_cls,
            "class_vote_ratio": round(vote_ratio, 3),
            "mean_class_confidence": round(mean_conf, 3),
            "class_switch_count": switch_count,
            "class_distribution": class_counts
        }


class AerialContextValidator:
    """
    Secondary physical, geometric, environmental, and temporal validator.
    Audits raw YOLO predictions without mutating raw records and assigns:
    PERSON, VEHICLE, BUILDING, BOAT, or UNCERTAIN_OBJECT.
    """

    VALID_CLASSES = {"PERSON", "VEHICLE", "BUILDING", "BOAT", "UNCERTAIN_OBJECT"}

    def __init__(self, config: Optional[ClassThresholdConfig] = None):
        self.config = config or ClassThresholdConfig()

    @staticmethod
    def compute_water_overlap(bbox: List[int], water_mask: Optional[np.ndarray]) -> float:
        """Calculates the fraction of bbox area overlapping with water mask."""
        if water_mask is None or water_mask.size == 0:
            return 0.0

        H, W = water_mask.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(W - 1, int(x1)))
        y1 = max(0, min(H - 1, int(y1)))
        x2 = max(0, min(W, int(x2)))
        y2 = max(0, min(H, int(y2)))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        box_crop = water_mask[y1:y2, x1:x2]
        box_area = float((x2 - x1) * (y2 - y1))
        water_pixels = float(np.count_nonzero(box_crop > 0))
        return water_pixels / max(1.0, box_area)

    @staticmethod
    def compute_bbox_metrics(bbox: List[int]) -> Tuple[int, int, int, float]:
        """Returns (width, height, area, aspect_ratio)."""
        w = max(1, int(bbox[2] - bbox[0]))
        h = max(1, int(bbox[3] - bbox[1]))
        area = w * h
        aspect = max(float(w) / float(h), float(h) / float(w))
        return w, h, area, aspect

    def validate_detection(
        self,
        raw_detection: Dict[str, Any],
        track_history: Optional[TrackClassHistory] = None,
        water_mask: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        point_3d: Optional[Union[List[float], np.ndarray]] = None,
        nearby_buildings: Optional[List[List[int]]] = None
    ) -> Dict[str, Any]:
        """
        Validates a single detection record.
        Preserves raw YOLO output in 'raw_yolo_output'.
        Returns validated class, confidence, evidence rationale, and auditable metrics.
        """
        raw_cls = str(raw_detection.get("raw_class_name", raw_detection.get("class_name", "object"))).lower()
        raw_conf = float(raw_detection.get("raw_detector_confidence", raw_detection.get("confidence", 0.0)))
        bbox = list(raw_detection.get("bbox_xyxy", raw_detection.get("bbox", [0, 0, 0, 0])))
        bw, bh, area, aspect = self.compute_bbox_metrics(bbox)

        # Build immutable raw output snapshot
        raw_yolo_record = {
            "frame_id": raw_detection.get("frame_id", "N/A"),
            "track_id": raw_detection.get("track_id"),
            "class": raw_cls,
            "confidence": round(raw_conf, 4),
            "bbox": bbox,
            "bbox_width": bw,
            "bbox_height": bh,
            "bbox_area": area,
            "source_tile": raw_detection.get("source_tile", "full_frame"),
            "original_image_coordinates": bbox,
            "raw_person_record": raw_detection.get("raw_person_record"),
            "tracking_confidence": raw_detection.get("tracking_confidence"),
            "confidence_3d": raw_detection.get("confidence_3d", raw_detection.get("3d_confidence")),
            "rescue_confidence": raw_detection.get("rescue_confidence")
        }
        if "raw_yolo_output" in raw_detection and raw_detection["raw_yolo_output"] is not None:
            raw_yolo_record["raw_yolo_output"] = raw_detection["raw_yolo_output"]
        else:
            raw_yolo_record["raw_yolo_output"] = {k: v for k, v in raw_yolo_record.items() if k != "raw_yolo_output"}

        # Normalize raw class to standard categories
        norm_raw_cls = "vehicle" if raw_cls in ["car", "truck", "bus", "motorcycle", "vehicle"] else (
            "person" if raw_cls == "person" else (
                "boat" if raw_cls == "boat" else (
                    "building" if raw_cls == "building" else raw_cls
                )
            )
        )

        water_overlap = self.compute_water_overlap(bbox, water_mask)

        # Check temporal history if available
        temporal_metrics = track_history.compute_temporal_metrics() if track_history else None
        dominant_cls = temporal_metrics.get("dominant_class", norm_raw_cls) if temporal_metrics else norm_raw_cls
        vote_ratio = temporal_metrics.get("class_vote_ratio", 1.0) if temporal_metrics else 1.0
        switch_count = temporal_metrics.get("class_switch_count", 0) if temporal_metrics else 0

        final_class = norm_raw_cls.upper()
        validation_conf = raw_conf
        evidence_reasons: List[str] = []
        is_changed = False

        # -------------------------------------------------------------
        # 1. Class-Specific Confidence Threshold Gate
        # -------------------------------------------------------------
        if norm_raw_cls == "person":
            req_thresh = self.config.person_thresh
        elif norm_raw_cls == "vehicle":
            req_thresh = self.config.vehicle_thresh
        elif norm_raw_cls == "boat":
            req_thresh = self.config.boat_thresh
        elif norm_raw_cls == "building":
            req_thresh = self.config.building_thresh
        else:
            req_thresh = 0.15

        if raw_conf < req_thresh:
            final_class = "UNCERTAIN_OBJECT"
            validation_conf = raw_conf * 0.5
            is_changed = True
            evidence_reasons.append(
                f"Confidence {raw_conf:.2f} below class threshold {req_thresh:.2f} for {norm_raw_cls}."
            )
            return self._build_result(raw_yolo_record, final_class, validation_conf, evidence_reasons, is_changed, temporal_metrics)

        # -------------------------------------------------------------
        # 2. Geometric & Physical Context Evaluation
        # -------------------------------------------------------------

        # CASE A: PERSON VALIDATION
        if norm_raw_cls == "person":
            if area > self.config.person_max_area:
                final_class = "UNCERTAIN_OBJECT"
                validation_conf = 0.0
                is_changed = True
                evidence_reasons.append(f"Person bbox area {area}px exceeds plausible human bounds (max {self.config.person_max_area:.0f}px).")
            elif area < self.config.person_min_area:
                final_class = "UNCERTAIN_OBJECT"
                validation_conf = 0.0
                is_changed = True
                evidence_reasons.append(f"Sub-pixel noise (area {area}px < {self.config.person_min_area:.0f}px).")
            elif aspect > self.config.person_max_aspect:
                final_class = "UNCERTAIN_OBJECT"
                validation_conf = 0.0
                is_changed = True
                evidence_reasons.append(f"Extreme aspect ratio {aspect:.2f} exceeds max {self.config.person_max_aspect:.1f}.")
            else:
                final_class = "PERSON"
                validation_conf = raw_conf
                evidence_reasons.append(f"Valid aerial person geometry (area={area}px, aspect={aspect:.2f}).")

        # CASE B: VEHICLE VALIDATION
        elif norm_raw_cls == "vehicle":
            # Test A check: House roof must not become vehicle
            if area > self.config.vehicle_max_area:
                if area >= self.config.building_min_area and (0.35 <= (bw / float(bh)) <= 3.0):
                    final_class = "BUILDING"
                    validation_conf = min(0.85, raw_conf + 0.15)
                    is_changed = True
                    evidence_reasons.append(
                        f"Building-like geometry (area={area}px > {self.config.vehicle_max_area:.0f}px, aspect={aspect:.2f}); remapped from oversized vehicle."
                    )
                else:
                    final_class = "UNCERTAIN_OBJECT"
                    validation_conf = 0.0
                    is_changed = True
                    evidence_reasons.append(
                        f"Oversized vehicle detection (area={area}px) with conflicting shape; flagged uncertain."
                    )
            elif area < self.config.vehicle_min_area:
                final_class = "UNCERTAIN_OBJECT"
                validation_conf = 0.0
                is_changed = True
                evidence_reasons.append(
                    f"Undersized vehicle candidate (area={area}px < {self.config.vehicle_min_area:.0f}px); noise suppression."
                )
            else:
                # Check for water reflection / submerged artifact (Test D)
                if water_overlap > 0.85 and raw_conf < 0.25:
                    final_class = "UNCERTAIN_OBJECT"
                    validation_conf = 0.0
                    is_changed = True
                    evidence_reasons.append(
                        f"Deep water overlap ({water_overlap*100:.0f}%) with low confidence ({raw_conf:.2f}); likely water reflection artifact."
                    )
                else:
                    final_class = "VEHICLE"
                    validation_conf = raw_conf
                    evidence_reasons.append(
                        f"Vehicle geometry confirmed (area={area}px, aspect={aspect:.2f}, water overlap={water_overlap*100:.0f}%)."
                    )

        # CASE C: BUILDING VALIDATION
        elif norm_raw_cls == "building":
            # Test B check: Real compact vehicle must not become building
            if area < 1200 and (self.config.vehicle_min_aspect <= aspect <= self.config.vehicle_max_aspect):
                final_class = "VEHICLE"
                validation_conf = raw_conf
                is_changed = True
                evidence_reasons.append(
                    f"Compact object (area={area}px, aspect={aspect:.2f}) matches ground vehicle bounds; remapped from building."
                )
            elif area >= self.config.building_min_area:
                # Flooded house remains building even if partially submerged (Test F)
                final_class = "BUILDING"
                validation_conf = max(raw_conf, 0.70)
                evidence_reasons.append(
                    f"Persistent structural footprint (area={area}px, aspect={aspect:.2f}, water overlap={water_overlap*100:.0f}%)."
                )
            else:
                final_class = "BUILDING"
                validation_conf = raw_conf
                evidence_reasons.append(f"Structure geometry (area={area}px).")

        # CASE D: BOAT VALIDATION
        elif norm_raw_cls == "boat":
            # Anti-boat roof filter & dry land check (Test C)
            if water_overlap < self.config.boat_min_water_overlap:
                if area >= self.config.building_min_area:
                    final_class = "BUILDING"
                    validation_conf = min(0.80, raw_conf + 0.10)
                    is_changed = True
                    evidence_reasons.append(
                        f"Boat predicted on dry structure (area={area}px, water overlap={water_overlap*100:.1f}%); remapped to BUILDING."
                    )
                elif area <= self.config.vehicle_max_area and (1.1 <= aspect <= 3.5):
                    final_class = "VEHICLE"
                    validation_conf = raw_conf
                    is_changed = True
                    evidence_reasons.append(
                        f"Boat detected on dry ground with vehicle dimensions (area={area}px, aspect={aspect:.2f}); remapped to VEHICLE."
                    )
                else:
                    final_class = "UNCERTAIN_OBJECT"
                    validation_conf = 0.0
                    is_changed = True
                    evidence_reasons.append(
                        f"Boat detected on dry terrain without water proximity (overlap={water_overlap*100:.1f}%); unsupported."
                    )
            else:
                final_class = "BOAT"
                validation_conf = raw_conf
                evidence_reasons.append(
                    f"Vessel confirmed in water context (overlap={water_overlap*100:.1f}%, aspect={aspect:.2f})."
                )

        # -------------------------------------------------------------
        # 3. Multi-Frame Temporal Consistency Integration
        # -------------------------------------------------------------
        if temporal_metrics and temporal_metrics.get("class_vote_ratio", 0.0) >= self.config.temporal_min_vote_ratio:
            dom_upper = dominant_cls.upper()
            if dom_upper in self.VALID_CLASSES and dom_upper != final_class:
                if vote_ratio >= 0.65 and final_class != "PERSON":
                    orig_fc = final_class
                    final_class = dom_upper
                    is_changed = True
                    evidence_reasons.append(
                        f"Temporal track consistency overrode transient {orig_fc} -> {dom_upper} (vote ratio: {vote_ratio:.2f}, {switch_count} switches)."
                    )

        if switch_count >= self.config.temporal_switch_threshold and vote_ratio < 0.50:
            if final_class != "PERSON":
                final_class = "UNCERTAIN_OBJECT"
                validation_conf = 0.0
                is_changed = True
                evidence_reasons.append(
                    f"Excessive class oscillations ({switch_count} switches across track); flagged UNCERTAIN_OBJECT."
                )

        return self._build_result(
            raw_yolo_record, final_class, validation_conf, evidence_reasons, is_changed, temporal_metrics
        )

    def _build_result(
        self,
        raw_record: Dict[str, Any],
        final_class: str,
        val_conf: float,
        reasons: List[str],
        is_changed: bool,
        temporal_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Builds standardized validation output dictionary with decoupled confidence semantics."""
        raw_conf = float(raw_record.get("confidence", raw_record.get("raw_detector_confidence", 0.0)))
        raw_c = raw_record.get("class", raw_record.get("raw_class_name", "unknown"))
        trk_id = raw_record.get("track_id")

        return {
            "raw_yolo_output": raw_record.get("raw_yolo_output", raw_record),
            "raw_person_record": raw_record.get("raw_person_record"),
            "raw_class": raw_c,
            "raw_confidence": raw_conf,
            "raw_yolo_confidence": round(raw_conf, 4),
            "final_class": final_class,
            "validation_confidence": round(float(val_conf), 4),
            "person_validation_confidence": round(float(val_conf), 4) if final_class == "PERSON" else 0.0,
            "tracking_confidence": raw_record.get("tracking_confidence"),
            "confidence_3d": raw_record.get("confidence_3d", raw_record.get("3d_confidence")),
            "rescue_confidence": raw_record.get("rescue_confidence"),
            "validation_evidence": " | ".join(reasons) if reasons else "Validated by physical context.",
            "is_changed": is_changed,
            "temporal_metrics": temporal_metrics or {},
            "track_id": trk_id,
            "bbox_xyxy": raw_record.get("bbox_xyxy", raw_record.get("bbox")),
            "bbox": raw_record.get("bbox", raw_record.get("bbox_xyxy")),
            "frame_id": raw_record.get("frame_id")
        }

    @staticmethod
    def is_actionable_object(validated_class: str) -> bool:
        """Strictly excludes UNCERTAIN_OBJECT from rescue and disaster incident actions."""
        return str(validated_class).upper() not in ["UNCERTAIN_OBJECT", "UNKNOWN", "NONE"]

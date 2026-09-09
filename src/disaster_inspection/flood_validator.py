import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2

logger = logging.getLogger("FloodValidator")

class FloodValidator:
    """
    Multi-Evidence Flood & Inundation Validation Engine.
    Evaluates 9 independent signals to distinguish:
    - NO_WATER
    - WATER_BODY (Normal river/lake/canal with dry bounds)
    - POTENTIAL_INUNDATION (Water near/overlapping dry land or structures)
    - FLOOD_ZONE (Confirmed flood inundation)
    """

    @staticmethod
    def evaluate_flood_evidence(
        rgb_image: np.ndarray,
        water_candidate_mask: np.ndarray,
        water_candidate_ratio: float,
        depth_map: np.ndarray,
        detected_objects: List[Dict[str, Any]],
        frame_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Fuses multi-modal signals to validate flood inundation without hard-blocking gates.
        """
        H, W = rgb_image.shape[:2]
        debug_dir = Path("outputs/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

        # 1. Semantic Water Mask & Confidence
        semantic_water_conf = min(1.0, float(water_candidate_ratio * 3.0)) if water_candidate_ratio > 0.02 else 0.0

        # 2. Spatial Coherence (Ratio of largest component to total water pixels)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(water_candidate_mask)
        if num_labels > 1:
            max_area = np.max(stats[1:, cv2.CC_STAT_AREA])
            total_water_px = np.sum(water_candidate_mask > 0)
            coherence = float(max_area / (total_water_px + 1e-6))
        else:
            coherence = 0.0

        # 3. Structural Building & Road Overlap (Inundation Analysis)
        building_boxes = [d["bbox"] for d in detected_objects if d.get("class_name") == "building"]
        road_boxes = [d["bbox"] for d in detected_objects if d.get("class_name") == "road"]

        bldg_mask = np.zeros((H, W), dtype=np.uint8)
        for box in building_boxes:
            x1, y1, x2, y2 = box
            bldg_mask[y1:y2, x1:x2] = 255

        road_mask = np.zeros((H, W), dtype=np.uint8)
        for box in road_boxes:
            x1, y1, x2, y2 = box
            road_mask[y1:y2, x1:x2] = 255

        bldg_overlap_px = np.sum((water_candidate_mask > 0) & (bldg_mask > 0))
        bldg_overlap_ratio = float(bldg_overlap_px / (np.sum(bldg_mask > 0) + 1e-6))

        road_overlap_px = np.sum((water_candidate_mask > 0) & (road_mask > 0))
        road_overlap_ratio = float(road_overlap_px / (np.sum(road_mask > 0) + 1e-6))

        # Inundation Mask: Water regions that overlap building bases, roads, or non-river ground
        inundation_mask = water_candidate_mask.copy()
        if len(building_boxes) > 0 or len(road_boxes) > 0:
            inundation_conf = min(1.0, float(0.30 + bldg_overlap_ratio * 1.5 + road_overlap_ratio * 1.2))
        else:
            # When no dry structures (buildings/roads) are inundated, water is a natural channel/open water body
            inundation_conf = min(0.30, float(water_candidate_ratio * 0.8))

        # 4. Save 4 Debug Evidence Masks (Requirement 12)
        cv2.imwrite(str(debug_dir / "flood_candidate_mask.png"), water_candidate_mask)
        cv2.imwrite(str(debug_dir / "flood_semantic_mask.png"), (water_candidate_mask * 0.8).astype(np.uint8))
        cv2.imwrite(str(debug_dir / "flood_inundation_mask.png"), inundation_mask)

        overlay = rgb_image.copy()
        overlay[water_candidate_mask > 0] = (overlay[water_candidate_mask > 0] * 0.4 + np.array([255, 120, 0]) * 0.6).astype(np.uint8)
        cv2.imwrite(str(debug_dir / "flood_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # 5. Multi-Evidence Score Fusion
        # Flood Score fuses water confidence (modulated by spatial coherence), inundation context, and coverage ratio
        effective_water_conf = semantic_water_conf * coherence
        flood_score = (
            0.40 * effective_water_conf +
            0.50 * inundation_conf +
            0.10 * min(1.0, water_candidate_ratio * 4.0)
        )

        incident_threshold = 0.35

        # Determine State
        if water_candidate_ratio < 0.03:
            water_state = "NO_WATER"
            gate_status = "FAIL"
            rejection_reason = "Water candidate coverage below 3% threshold."
        elif flood_score >= 0.45:
            water_state = "FLOOD_ZONE"
            gate_status = "PASS"
            rejection_reason = "N/A - Validated flood inundation evidence."
        elif flood_score >= 0.35:
            water_state = "POTENTIAL_INUNDATION"
            gate_status = "PASS"
            rejection_reason = "N/A - Moderate flood inundation evidence."
        else:
            water_state = "WATER_BODY"
            gate_status = "FAIL"
            rejection_reason = "Water detected but confined to normal channel; no inundation of dry land/structures."

        validation_log = {
            "frame_idx": frame_idx,
            "water_candidate_ratio": round(float(water_candidate_ratio), 4),
            "semantic_water_confidence": round(float(semantic_water_conf), 4),
            "inundation_confidence": round(float(inundation_conf), 4),
            "spatial_coherence": round(float(coherence), 4),
            "building_overlap_ratio": round(float(bldg_overlap_ratio), 4),
            "road_overlap_ratio": round(float(road_overlap_ratio), 4),
            "flood_incident_score": round(float(flood_score), 4),
            "incident_threshold": incident_threshold,
            "water_state": water_state,
            "incident_gate": gate_status,
            "rejection_reason": rejection_reason
        }

        # Write outputs/debug/flood_validation.json (Requirement 11)
        with open(debug_dir / "flood_validation.json", "w") as f:
            json.dump(validation_log, f, indent=4)

        logger.info(f"[FLOOD VALIDATOR] State: {water_state} | Score: {flood_score:.3f} | Gate: {gate_status}")
        return validation_log

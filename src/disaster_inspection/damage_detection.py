import numpy as np
import cv2
import logging
from typing import List, Dict, Any

logger = logging.getLogger("DamageDetector")

class StructuralDamageDetector:
    """
    Detects visual indicators of building structural damage, roof discontinuities,
    and structural anomalies from drone imagery and depth maps.
    Calculates confidence from empirical edge variance and depth gradient metrics.
    """

    def analyze_structural_damage(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        detected_buildings: List[Dict[str, Any]],
        frame_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Analyzes building bounding regions for structural damage indicators.
        """
        H, W = rgb_image.shape[:2]
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(-1)
        if depth_map.shape[:2] != (H, W):
            depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        gray_lap = cv2.Laplacian(gray, cv2.CV_64F)
        damage_incidents = []

        # Analyze detected buildings only; do not invent fake structures if none are present
        bldg_candidates = [
            d for d in detected_buildings
            if d.get("class_name") in ["building", "structure", None] or "STRUCTURE" in str(d.get("object_id", ""))
        ]
        if len(detected_buildings) > 0 and len(bldg_candidates) == 0:
            return []

        regions_to_check = bldg_candidates

        for idx, bldg in enumerate(regions_to_check):
            x1, y1, x2, y2 = bldg["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            if (x2 - x1) < 30 or (y2 - y1) < 30:
                continue

            crop_lap = gray_lap[y1:y2, x1:x2]
            crop_depth = depth_map[y1:y2, x1:x2]

            if crop_depth.shape[0] < 5 or crop_depth.shape[1] < 5 or crop_lap.size == 0:
                continue

            # 1. Texture irregularity / roof disruption score (Laplacian variance)
            lap_var = float(np.var(crop_lap))
            
            # 2. Depth gradient / surface discontinuity
            try:
                gy, gx = np.gradient(crop_depth)
                depth_grad_mag = np.sqrt(gx**2 + gy**2)
                valid_grad_mask = np.isfinite(depth_grad_mag) & (crop_depth > 0.1)
                high_grad_ratio = float(np.mean(depth_grad_mag[valid_grad_mask] > 0.20)) if np.any(valid_grad_mask) else 0.0
            except Exception:
                high_grad_ratio = 0.0

            # Calculate empirical confidence from measured variance
            # lap_var threshold ~ 1200, high_grad_ratio threshold ~ 0.25
            var_score = min(0.5, (lap_var / 2500.0) * 0.5)
            grad_score = min(0.5, (high_grad_ratio / 0.40) * 0.5)
            computed_conf = round(float(var_score + grad_score), 3)

            # Requires significant combined evidence before flagging damage
            if high_grad_ratio > 0.22 and lap_var > 1400:
                damage_level = "potential_major_damage"
                severity = "HIGH"
                evidence = (
                    f"Measured high depth gradient discontinuity ({high_grad_ratio*100:.1f}% area > 0.20m/px) "
                    f"and high surface texture chaos (Laplacian variance: {lap_var:.1f})."
                )
            elif high_grad_ratio > 0.15 and lap_var > 900:
                damage_level = "minor_damage"
                severity = "MEDIUM"
                evidence = (
                    f"Moderate surface elevation gradient ({high_grad_ratio*100:.1f}% area) "
                    f"and edge texture variance ({lap_var:.1f})."
                )
            else:
                # Intact / smooth structure - no incident created
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            damage_incidents.append({
                "incident_type": "STRUCTURAL_DAMAGE",
                "damage_level": damage_level,
                "confidence": max(0.40, min(0.92, computed_conf)),
                "target_object_id": bldg.get("object_id", f"STRUCTURE_{idx+1:03d}"),
                "frame_idx": frame_idx,
                "bbox": [x1, y1, x2, y2],
                "pixel_center": (cx, cy),
                "evidence": evidence,
                "status": "NEEDS_VERIFICATION",
                "severity": severity
            })

        return damage_incidents

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("SpatialContext")

class SpatialContextAnalyzer:
    """
    Environmental spatial reasoning engine for detected persons and vehicles.
    Analyzes spatial relationships between targets and floodwater:
    - Flood proximity (distance to water boundary)
    - Surrounding water ratio (% of surrounding perimeter in water)
    - Rooftop / elevation context
    - Dry land accessibility
    - Classification into NORMAL_PERSON, POTENTIAL_ISOLATED_PERSON, POTENTIAL_STRANDED_PERSON.
    
    IMPORTANT:
    NEVER equates "PERSON = VICTIM".
    Every label is strictly grounded in measurable environmental spatial evidence.
    """

    @classmethod
    def analyze_person_context(
        cls,
        person_detection: Optional[Dict[str, Any]] = None,
        flood_mask: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        image_shape: Tuple[int, int] = (1080, 1920),
        person_center_orig: Optional[Tuple[float, float]] = None,
        person_bbox_orig: Optional[List[float]] = None,
        water_mask_orig: Optional[np.ndarray] = None,
        point_3d: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluates the spatial environmental context for a single detected person.
        Supports both structured person_detection dict and explicit coordinate kwargs.
        """
        # Resolve flood mask
        f_mask_in = flood_mask if flood_mask is not None else water_mask_orig
        if f_mask_in is None:
            f_mask_in = np.zeros(image_shape[:2], dtype=np.uint8)

        H, W = f_mask_in.shape[:2]

        # Resolve bbox
        if person_bbox_orig is not None:
            x1, y1, x2, y2 = [int(v) for v in person_bbox_orig]
        elif person_detection and "bbox" in person_detection:
            x1, y1, x2, y2 = [int(v) for v in person_detection["bbox"]]
        elif person_center_orig is not None:
            cx, cy = int(person_center_orig[0]), int(person_center_orig[1])
            x1, y1, x2, y2 = max(0, cx - 15), max(0, cy - 30), min(W, cx + 15), min(H, cy + 30)
        else:
            x1, y1, x2, y2 = 0, 0, 10, 10

        cx = int((x1 + x2) / 2.0) if person_center_orig is None else int(person_center_orig[0])
        cy = int((y1 + y2) / 2.0) if person_center_orig is None else int(person_center_orig[1])
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        f_mask = f_mask_in

        has_flood_in_scene = np.count_nonzero(f_mask) > (0.01 * H * W)

        if not has_flood_in_scene:
            return {
                "classification": "NORMAL_PERSON",
                "isolation_type": "NORMAL_PERSON",
                "flood_proximity": "NONE",
                "flood_distance_px": 999.0,
                "surrounding_flood_pct": 0.0,
                "surrounded_percentage": 0.0,
                "accessible_dry_area": "HIGH",
                "rooftop_elevation_context": "GROUND_LEVEL",
                "isolation_score": 0.0,
                "nearest_water_distance_m": 99.0,
                "elevation_relative_m": 0.0,
                "evidence": "Person observed in dry environment with no significant floodwater detected."
            }

        # 1. Distance to nearest floodwater
        # Invert mask so flood is 0 and non-flood is 255 for distanceTransform
        flood_binary = (f_mask > 0).astype(np.uint8)
        dist_to_flood = cv2.distanceTransform(1 - flood_binary, cv2.DIST_L2, 5)
        
        cx_c = max(0, min(W - 1, cx))
        cy_c = max(0, min(H - 1, cy))
        distance_px = float(dist_to_flood[cy_c, cx_c])

        # If distance is 0, person's center is directly inside or contacting floodwater
        inside_water = bool(flood_binary[cy_c, cx_c] > 0)

        # 2. Surrounding flood ratio (analyze ring around the bounding box)
        pad_x = int(bw * 1.5)
        pad_y = int(bh * 1.5)
        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(W, x2 + pad_x)
        ry2 = min(H, y2 + pad_y)

        outer_region = flood_binary[ry1:ry2, rx1:rx2]
        surrounding_water_ratio = float(np.mean(outer_region)) if outer_region.size > 0 else 0.0

        # 3. Rooftop / Elevation Context
        is_elevated = False
        elevation_label = "GROUND_LEVEL"
        if depth_map is not None:
            d_map = depth_map if depth_map.shape[:2] == (H, W) else cv2.resize(depth_map, (W, H))
            person_depths = d_map[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            valid_pd = person_depths[np.isfinite(person_depths) & (person_depths > 0.1)]
            
            surrounding_depths = d_map[ry1:ry2, rx1:rx2]
            valid_sd = surrounding_depths[np.isfinite(surrounding_depths) & (surrounding_depths > 0.1)]

            if len(valid_pd) > 5 and len(valid_sd) > 20:
                med_person_d = float(np.median(valid_pd))
                med_surround_d = float(np.median(valid_sd))
                # If camera is top-down (aerial), smaller depth means closer to camera = higher elevation
                if (med_surround_d - med_person_d) > 0.80:
                    is_elevated = True
                    elevation_label = "ELEVATED_STRUCTURE"

        # 4. Synthesize isolation score [0.0 - 1.0]
        # High isolation: small distance to flood + high surrounding water ratio + elevated
        proximity_factor = float(np.clip(1.0 - (distance_px / 120.0), 0.0, 1.0))
        surrounding_factor = float(np.clip(surrounding_water_ratio * 1.5, 0.0, 1.0))
        elevation_factor = 0.85 if is_elevated else 0.30

        isolation_score = round(
            0.45 * proximity_factor + 0.35 * surrounding_factor + 0.20 * elevation_factor, 3
        )

        # 5. Classify
        if inside_water or surrounding_water_ratio > 0.55 or (is_elevated and surrounding_water_ratio > 0.35):
            classification = "POTENTIAL_STRANDED_PERSON"
            flood_proximity = "HIGH (INUNDATED SURROUNDINGS)"
            accessible_dry = "LOW / SEVERELY RESTRICTED"
            evidence = (
                f"Person is isolated on an elevated structure surrounded by floodwater "
                f"({surrounding_water_ratio*100:.0f}% water perimeter coverage; distance to water: {distance_px:.0f}px)."
                if is_elevated else
                f"Person is located in/adjacent to floodwater ({surrounding_water_ratio*100:.0f}% water contact) with restricted dry ground access."
            )
        elif distance_px < 150.0 or surrounding_water_ratio > 0.20:
            classification = "POTENTIAL_ISOLATED_PERSON"
            flood_proximity = "MODERATE"
            accessible_dry = "PARTIALLY ACCESSIBLE"
            evidence = f"Person detected within {distance_px:.0f}px of floodwater with potential access route."
        else:
            classification = "NORMAL_PERSON"
            flood_proximity = "LOW"
            accessible_dry = "HIGH"
            evidence = f"Person observed on accessible dry ground ({distance_px:.0f}px from floodwater)."

        return {
            "classification": classification,
            "isolation_type": classification,
            "flood_proximity": flood_proximity,
            "flood_distance_px": round(distance_px, 1),
            "surrounding_flood_pct": round(surrounding_water_ratio * 100.0, 1),
            "surrounded_percentage": round(surrounding_water_ratio * 100.0, 1),
            "nearest_water_distance_m": round(distance_px * 0.05, 2),
            "accessible_dry_area": accessible_dry,
            "rooftop_elevation_context": elevation_label,
            "elevation_relative_m": 2.0 if is_elevated else 0.0,
            "isolation_score": isolation_score,
            "evidence": evidence
        }

    @classmethod
    def analyze_vehicle_context(
        cls,
        vehicle_detection: Dict[str, Any],
        flood_mask: np.ndarray,
        image_shape: Tuple[int, int] = (1080, 1920)
    ) -> Dict[str, Any]:
        """Evaluates vehicle flood context without assuming every vehicle is stranded."""
        H, W = image_shape[:2]
        x1, y1, x2, y2 = vehicle_detection["bbox"]
        cx = int((x1 + x2) / 2.0)
        cy = int((y1 + y2) / 2.0)

        f_mask = flood_mask if flood_mask.shape[:2] == (H, W) else cv2.resize(flood_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        flood_binary = (f_mask > 0).astype(np.uint8)

        v_crop = flood_binary[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
        water_contact_pct = float(np.mean(v_crop)) if v_crop.size > 0 else 0.0

        if water_contact_pct > 0.35:
            classification = "FLOODED_VEHICLE"
            severity = "HIGH"
            ev = f"Vehicle partially/fully submerged in floodwater ({water_contact_pct*100:.0f}% water contact)."
        elif water_contact_pct > 0.05:
            classification = "VEHICLE_NEAR_FLOOD"
            severity = "MEDIUM"
            ev = f"Vehicle in close proximity to active flood boundary ({water_contact_pct*100:.0f}% water contact)."
        else:
            classification = "NORMAL_VEHICLE"
            severity = "LOW"
            ev = "Vehicle on dry roadway/ground."

        return {
            "classification": classification,
            "severity": severity,
            "water_contact_pct": round(water_contact_pct * 100.0, 1),
            "evidence": ev
        }

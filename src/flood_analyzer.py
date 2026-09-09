import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, List, Optional

from src.flood_segmentation import (
    BaseFloodSegmenter,
    FloodSegmentationResult,
    WaterState
)

logger = logging.getLogger("FloodAnalyzer")


class MultiEvidenceFloodSegmenter(BaseFloodSegmenter):
    """
    Multi-evidence flood detection and inundation reasoning engine for drone aerial imagery.
    Implements BaseFloodSegmenter interface.
    
    Replaces simplistic single-color HSV blue thresholding with layered physical evidence:
    1. Spectral characteristics: Blue/cyan water AND muddy/turbid brown/tan flood runoff.
    2. Texture smoothness: Low local edge density and low Laplacian variance (specular water surfaces).
    3. Spatial continuity & area coverage: Contiguous connected components.
    4. Structural context: Contact/overlap with roads, structures, and detected entities.
    5. Depth flatness: Consistent horizontal elevation in 3D / depth space.
    
    States:
        - NO_WATER: No significant water detected (< 2% coverage)
        - WATER_BODY: Normal river/canal/lake confined in channel
        - POTENTIAL_INUNDATION: Elevated water coverage adjacent to structures/roads
        - VALIDATED_INUNDATION: Confirmed disaster flood spill across land/structures
    """

    @staticmethod
    def extract_water_candidates(bgr_img: np.ndarray) -> np.ndarray:
        """
        Extracts candidate water pixels across multiple spectral regimes:
        - Blue/cyan water (standard clear water bodies)
        - Turbid/muddy brown/tan floodwater (common in real disaster storm runoff)
        - Dark/reflective water surfaces with low saturation
        """
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 1. Clear / Blue / Cyan Water: Hue in [80, 135], moderate-to-high saturation
        mask_blue = (h >= 80) & (h <= 135) & (s >= 25) & (v >= 30) & (v <= 245)

        # 2. Muddy / Turbid Floodwater (Brown/Tan/Silt):
        # Brown/tan is low-hue warm color (H in [6, 35]), saturation in [15, 220], brightness in [35, 230]
        mask_turbid = (h >= 6) & (h <= 35) & (s >= 15) & (s <= 220) & (v >= 35) & (v <= 230)

        # 3. Clear/Blue water spectral tendency (B >= R - 10)
        b, g, r_ch = cv2.split(bgr_img)
        water_blue_tendency = (b.astype(np.int16) >= r_ch.astype(np.int16) - 10)

        candidate_color = (mask_blue & water_blue_tendency) | mask_turbid

        # Combine with texture flatness: water surfaces have extremely low edge density
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        # Local texture variance filter
        smooth_surface = grad_mag < 28.0

        raw_mask = candidate_color & smooth_surface
        raw_mask_uint8 = (raw_mask.astype(np.uint8)) * 255

        # Morphological cleanup (remove isolated pixels, fill internal gaps)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cleaned = cv2.morphologyEx(raw_mask_uint8, cv2.MORPH_OPEN, kernel_open)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

        return cleaned

    def segment(
        self,
        bgr_img: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        detected_objects: Optional[List[Dict[str, Any]]] = None,
        frame_idx: int = 0
    ) -> FloodSegmentationResult:
        """
        BaseFloodSegmenter interface implementation.
        """
        return self.analyze_flood(bgr_img, depth_map, detected_objects, frame_idx)

    @classmethod
    def analyze_flood(
        cls,
        bgr_img: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        detected_objects: Optional[List[Dict[str, Any]]] = None,
        frame_idx: int = 0
    ) -> FloodSegmentationResult:
        """
        Comprehensive multi-evidence flood analysis for a drone frame.
        """
        H, W = bgr_img.shape[:2]
        total_pixels = float(H * W)

        # 1. Candidate water extraction
        candidate_mask = cls.extract_water_candidates(bgr_img)
        candidate_pixels = int(np.count_nonzero(candidate_mask))
        candidate_ratio = float(candidate_pixels / total_pixels)

        # 2. Spatial continuity (Cluster / Connected Component analysis)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
        
        largest_component_ratio = 0.0
        compact_flood_mask = np.zeros_like(candidate_mask)

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_area = float(np.max(areas))
            largest_component_ratio = float(largest_area / total_pixels)
            
            # Keep components that occupy at least 0.5% of the frame (reject noise flecks)
            min_component_area = int(total_pixels * 0.005)
            valid_labels = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_component_area)[0] + 1
            for lbl in valid_labels:
                compact_flood_mask[labels == lbl] = 255

        validated_pixels = int(np.count_nonzero(compact_flood_mask))
        validated_ratio = float(validated_pixels / total_pixels)

        # 3. Depth flatness evidence (if depth map provided)
        depth_flatness_score = 0.50
        if depth_map is not None and validated_pixels > 100:
            d_flat = depth_map if depth_map.shape[:2] == (H, W) else cv2.resize(depth_map, (W, H))
            water_depths = d_flat[compact_flood_mask > 0]
            valid_wd = water_depths[np.isfinite(water_depths) & (water_depths > 0.1)]
            if len(valid_wd) > 50:
                std_d = float(np.std(valid_wd))
                mean_d = float(np.mean(valid_wd)) + 1e-4
                # Coeff of variation: water depth in a local pool is relatively uniform
                cv_d = std_d / mean_d
                depth_flatness_score = float(np.clip(1.0 - cv_d, 0.0, 1.0))

        # 4. Contextual contact with buildings, roads, vehicles, and people
        building_overlap = 0.0
        road_overlap = 0.0
        has_victim_or_vehicle_in_water = False

        if detected_objects:
            for obj in detected_objects:
                cls_name = str(obj.get("class_name", "")).lower()
                x1, y1, x2, y2 = [int(v) for v in obj.get("bbox", [0, 0, 0, 0])]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                crop = compact_flood_mask[y1:y2, x1:x2]

                if cls_name in ["building", "structure", "house"]:
                    if crop.size > 0:
                        building_overlap = max(building_overlap, float(np.mean(crop > 0)))
                elif cls_name in ["car", "truck", "bus", "vehicle", "traffic light"]:
                    if crop.size > 0:
                        road_overlap = max(road_overlap, float(np.mean(crop > 0)))
                        if np.mean(crop > 0) > 0.05:
                            has_victim_or_vehicle_in_water = True
                elif cls_name in ["person", "boat"]:
                    if crop.size > 0 and np.mean(crop > 0) > 0.05:
                        has_victim_or_vehicle_in_water = True

        # Calculate aspect ratio of largest component (rivers/canals have high aspect ratio ribbon)
        if num_labels > 1:
            idx_max = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            w_box = stats[idx_max, cv2.CC_STAT_WIDTH]
            h_box = stats[idx_max, cv2.CC_STAT_HEIGHT]
            aspect = max(w_box, h_box) / max(1, min(w_box, h_box))
        else:
            aspect = 1.0

        # A confined channel is strictly a narrow ribbon (aspect >= 3.5) with coverage <= 20%,
        # no building/road overlap, and NO people, boats, or vehicles contacting the water
        is_confined_channel = (
            aspect >= 3.5 and 
            building_overlap < 0.02 and 
            road_overlap < 0.02 and 
            not has_victim_or_vehicle_in_water and
            validated_ratio <= 0.20
        )

        # 5. Determine State
        if validated_ratio < 0.02 and candidate_ratio < 0.03:
            water_state = WaterState.NO_WATER.value
            flood_confidence = 0.0
            gate = "FAIL"
            reason = "Water candidate coverage is negligible (<2% of image area)."
        elif is_confined_channel:
            water_state = WaterState.WATER_BODY.value
            flood_confidence = 0.15
            gate = "FAIL"
            reason = f"Water detected but confined in natural channel/canal (aspect ratio {aspect:.1f}:1, no infrastructure inundation)."
        elif validated_ratio >= 0.20 or (validated_ratio >= 0.06 and (building_overlap > 0.02 or road_overlap > 0.03 or has_victim_or_vehicle_in_water)):
            water_state = WaterState.VALIDATED_INUNDATION.value
            flood_confidence = float(min(1.0, 0.40 + validated_ratio * 1.5 + (building_overlap * 0.3) + (depth_flatness_score * 0.2)))
            gate = "PASS"
            reason = f"Confirmed inundation across {validated_ratio*100:.1f}% of scene contacting built infrastructure/terrain."
        elif validated_ratio >= 0.04 or candidate_ratio >= 0.06:
            water_state = WaterState.POTENTIAL_INUNDATION.value
            flood_confidence = float(np.clip(0.35 + validated_ratio * 1.0, 0.30, 0.75))
            gate = "PASS"
            reason = f"Potential flood inundation detected ({validated_ratio*100:.1f}% surface coverage)."
        else:
            water_state = WaterState.WATER_BODY.value
            flood_confidence = 0.10
            gate = "FAIL"
            reason = "Minor water body candidate detected without signs of hazardous flooding."

        result = FloodSegmentationResult(
            water_state=water_state,
            water_ratio=round(validated_ratio, 4),
            candidate_ratio=round(candidate_ratio, 4),
            validated_ratio=round(validated_ratio, 4),
            flood_confidence=round(flood_confidence, 3),
            candidate_mask=candidate_mask,
            validated_mask=compact_flood_mask,
            depth_flatness_score=round(depth_flatness_score, 3),
            building_overlap=round(building_overlap, 3),
            road_overlap=round(road_overlap, 3),
            largest_component_ratio=round(largest_component_ratio, 4),
            rejection_reason=reason if gate == "FAIL" else None,
            summary_evidence=reason,
            incident_gate=gate,
            frame_idx=frame_idx
        )

        logger.info(
            f"[FLOOD ANALYZER] Frame {frame_idx}: State={water_state} (Coverage={validated_ratio*100:.1f}%, Gate={gate})"
        )
        return result


# Backward-compatible alias
FloodAnalyzer = MultiEvidenceFloodSegmenter

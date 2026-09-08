import numpy as np
import cv2
from typing import Dict, Any, Tuple

class DepthConsistencyAnalyzer:
    """
    Computes dense depth agreement/disagreement metrics between metric-aligned Depth Anything V2
    and VGGT multi-view predictions to evaluate structural consistency.
    """

    def __init__(self, sigma: float = 0.5, max_diff: float = 5.0):
        self.sigma = sigma
        self.max_diff = max_diff

    def compute_consistency(
        self,
        vggt_depth: np.ndarray,
        aligned_da2_depth: np.ndarray,
        vggt_conf: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Computes absolute depth difference map and normalized exponential consistency map.

        Args:
            vggt_depth: [H, W] VGGT depth map
            aligned_da2_depth: [H, W] metric-aligned Depth Anything V2 depth map
            vggt_conf: [H, W] VGGT confidence

        Returns:
            abs_diff_map: [H, W] float32 absolute error (meters)
            consistency_map: [H, W] float32 agreement score in [0.0, 1.0]
            stats: dict containing mean_diff, median_diff, consistency_percentage
        """
        if vggt_depth.ndim == 3:
            vggt_depth = vggt_depth.squeeze(-1)
        if aligned_da2_depth.ndim == 3:
            aligned_da2_depth = aligned_da2_depth.squeeze(-1)

        # 1. Absolute depth difference
        abs_diff_map = np.abs(vggt_depth - aligned_da2_depth)
        abs_diff_map = np.nan_to_num(abs_diff_map, nan=self.max_diff, posinf=self.max_diff)

        # 2. Exponential consistency map: C = exp(- abs_diff / sigma)
        consistency_map = np.exp(-abs_diff_map / (self.sigma + 1e-6)).astype(np.float32)

        # 3. Valid region mask
        valid_mask = (vggt_depth > 0.1) & (aligned_da2_depth > 0.1) & ~np.isnan(vggt_depth)
        if vggt_conf is not None:
            if vggt_conf.ndim == 3:
                vggt_conf = vggt_conf.squeeze(-1)
            valid_mask = valid_mask & (vggt_conf > 0.5)

        if np.sum(valid_mask) > 0:
            mean_diff = float(np.mean(abs_diff_map[valid_mask]))
            median_diff = float(np.median(abs_diff_map[valid_mask]))
            high_agreement_pct = float(np.mean(consistency_map[valid_mask] > 0.6) * 100.0)
        else:
            mean_diff = 0.0
            median_diff = 0.0
            high_agreement_pct = 0.0

        stats = {
            "mean_depth_difference_m": mean_diff,
            "median_depth_difference_m": median_diff,
            "high_consistency_percentage": high_agreement_pct
        }

        return abs_diff_map.astype(np.float32), consistency_map, stats

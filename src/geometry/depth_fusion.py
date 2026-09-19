"""
Depth Comparison & Fusion Engine for UAV 3D Reconstruction.
Evaluates monocular Depth Anything V2 against VGGT metric depth.
Supports modes: 'vggt', 'depth_anything', 'comparison', and 'hybrid'.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DepthComparisonReport:
    mode: str
    vggt_mean_depth: float
    monocular_mean_depth: float
    scale_factor: float
    shift_offset: float
    alignment_rmse: float
    alignment_mae: float
    mean_confidence: float
    final_depth_source: str
    final_depth: np.ndarray  # [H, W] float32


class DepthFusionEngine:
    """
    Robust scale and shift alignment between relative monocular depth and VGGT metric depth.
    Computes statistical disparity metrics and handles comparison/fusion.
    """

    def __init__(
        self,
        mode: str = "vggt",  # "vggt", "depth_anything", "comparison", "hybrid"
        confidence_threshold: float = 0.25,
    ):
        self.mode = mode.lower()
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def align_scale_shift(
        metric_depth: np.ndarray,
        relative_depth: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Solves for scale (s) and shift (t) minimizing:
            || s * relative_depth + t - metric_depth ||_2
        Uses robust median-based estimator to resist outliers.
        """
        if mask is None:
            valid = (metric_depth > 0.01) & (relative_depth > 0.01)
        else:
            valid = (metric_depth > 0.01) & (relative_depth > 0.01) & (mask > 0)

        if np.count_nonzero(valid) < 50:
            # Insufficient overlapping pixels for regression
            return 1.0, 0.0, relative_depth

        y = metric_depth[valid]
        x = relative_depth[valid]

        # Least squares closed-form solution for 1D affine alignment
        # s * x + t = y
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        denom = np.sum((x - x_mean) ** 2)
        if denom < 1e-8:
            s = 1.0
            t = y_mean - x_mean
        else:
            s = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
            t = float(y_mean - s * x_mean)

        # Enforce positive scale
        if s <= 0:
            s = float(np.median(y) / (np.median(x) + 1e-7))
            t = 0.0

        aligned_rel = s * relative_depth + t
        return s, t, aligned_rel

    def process(
        self,
        vggt_depth: np.ndarray,
        vggt_conf: np.ndarray,
        monocular_depth: Optional[np.ndarray] = None,
        monocular_conf: Optional[np.ndarray] = None,
        dynamic_mask: Optional[np.ndarray] = None,
    ) -> DepthComparisonReport:
        """
        Evaluates VGGT depth against monocular depth and returns the comparison report.

        Args:
            vggt_depth: [H, W] float32 metric depth from VGGT.
            vggt_conf: [H, W] float32 confidence map from VGGT.
            monocular_depth: Optional [H, W] float32 from Depth Anything V2.
            monocular_conf: Optional [H, W] float32 confidence from Depth Anything V2.
            dynamic_mask: Optional binary mask [H, W] (non-zero = dynamic to exclude).

        Returns:
            DepthComparisonReport with metrics and selected depth map.
        """
        # Ensure 2D
        v_depth = vggt_depth.squeeze()
        v_conf = vggt_conf.squeeze()
        h, w = v_depth.shape[:2]

        # Invert dynamic mask so static=True (resize if resolutions differ)
        if dynamic_mask is not None:
            if dynamic_mask.shape[:2] != (h, w):
                import cv2
                d_mask_resized = cv2.resize(dynamic_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                d_mask_resized = dynamic_mask
            static_mask = (d_mask_resized == 0)
        else:
            static_mask = np.ones((h, w), dtype=bool)

        if monocular_depth is None or self.mode == "vggt":
            return DepthComparisonReport(
                mode="vggt",
                vggt_mean_depth=float(np.mean(v_depth[static_mask])),
                monocular_mean_depth=0.0,
                scale_factor=1.0,
                shift_offset=0.0,
                alignment_rmse=0.0,
                alignment_mae=0.0,
                mean_confidence=float(np.mean(v_conf[static_mask])),
                final_depth_source="vggt",
                final_depth=v_depth,
            )

        m_depth = monocular_depth.squeeze()
        m_conf = monocular_conf.squeeze() if monocular_conf is not None else np.ones_like(m_depth)

        # Match dimensions if needed
        if m_depth.shape != (h, w):
            import cv2
            m_depth = cv2.resize(m_depth, (w, h), interpolation=cv2.INTER_LINEAR)
            m_conf = cv2.resize(m_conf, (w, h), interpolation=cv2.INTER_LINEAR)

        # Scale and shift alignment
        scale, shift, aligned_mono = self.align_scale_shift(v_depth, m_depth, static_mask)

        # Compute residuals in static regions
        diff = np.abs(aligned_mono - v_depth)
        valid_eval = static_mask & (v_depth > 0.01)
        if np.any(valid_eval):
            rmse = float(np.sqrt(np.mean((aligned_mono[valid_eval] - v_depth[valid_eval]) ** 2)))
            mae = float(np.mean(diff[valid_eval]))
        else:
            rmse = mae = 0.0

        logger.info(
            f"[DEPTH_FUSION] Mode: '{self.mode}' | Scale: {scale:.3f}, Shift: {shift:.3f} | "
            f"Alignment RMSE: {rmse:.3f}m, MAE: {mae:.3f}m"
        )

        if self.mode == "depth_anything":
            final_depth = aligned_mono
            final_source = "depth_anything_v2"
        elif self.mode == "hybrid":
            # Confidence-weighted fusion: where VGGT has low confidence, rely more on aligned Depth Anything
            weight_v = np.clip(v_conf, 0.0, 1.0)
            weight_m = np.clip(m_conf, 0.0, 1.0) * 0.5  # slight damping for monocular prior
            sum_w = weight_v + weight_m + 1e-7
            final_depth = (v_depth * weight_v + aligned_mono * weight_m) / sum_w
            final_source = "hybrid_vggt_depth_anything"
        else:
            # Default 'comparison' mode: preserves VGGT metric depth, records comparison stats
            final_depth = v_depth
            final_source = "vggt"

        return DepthComparisonReport(
            mode=self.mode,
            vggt_mean_depth=float(np.mean(v_depth[valid_eval])) if np.any(valid_eval) else 0.0,
            monocular_mean_depth=float(np.mean(m_depth[valid_eval])) if np.any(valid_eval) else 0.0,
            scale_factor=round(scale, 4),
            shift_offset=round(shift, 4),
            alignment_rmse=round(rmse, 4),
            alignment_mae=round(mae, 4),
            mean_confidence=float(np.mean(v_conf[valid_eval])) if np.any(valid_eval) else 0.0,
            final_depth_source=final_source,
            final_depth=final_depth,
        )

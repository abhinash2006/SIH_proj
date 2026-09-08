import numpy as np
import cv2
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("DepthAlignment")

class RobustDepthAligner:
    """
    Estimates optimal scale (s) and shift (t) parameters to align relative monocular depth maps
    from Depth Anything V2 into metric coordinate space of VGGT depth maps:
        D_aligned = s * D_da2 + t
    Uses RANSAC / Median-ratio statistics to reject sky regions, zero depth values, and invalid points.
    """

    def __init__(self, method: str = "robust_ransac", max_depth: float = 100.0, min_depth: float = 0.1):
        self.method = method
        self.max_depth = max_depth
        self.min_depth = min_depth

    def align_depth_maps(
        self,
        da2_depth: np.ndarray,
        vggt_depth: np.ndarray,
        vggt_confidence: np.ndarray = None,
        conf_threshold: float = 1.0
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Aligns a single Depth Anything V2 depth map to VGGT depth map.

        Args:
            da2_depth: [H, W] Depth Anything V2 monocular depth
            vggt_depth: [H, W] VGGT predicted depth map
            vggt_confidence: [H, W] optional VGGT confidence mask
            conf_threshold: minimum confidence score required for alignment pixels

        Returns:
            aligned_da2_depth: [H, W] metric-aligned Depth Anything V2 depth
            metrics: dict containing scale, shift, residual, and valid_pixel_ratio
        """
        # Ensure dimensions match
        H, W = vggt_depth.shape[:2]
        if da2_depth.shape[:2] != (H, W):
            da2_depth = cv2.resize(da2_depth, (W, H), interpolation=cv2.INTER_LINEAR)

        if da2_depth.ndim == 3:
            da2_depth = da2_depth.squeeze(-1)
        if vggt_depth.ndim == 3:
            vggt_depth = vggt_depth.squeeze(-1)

        # Build valid mask
        valid_mask = (
            (vggt_depth > self.min_depth) &
            (vggt_depth < self.max_depth) &
            (da2_depth > 1e-4) &
            ~np.isnan(vggt_depth) &
            ~np.isnan(da2_depth) &
            ~np.isinf(vggt_depth) &
            ~np.isinf(da2_depth)
        )

        if vggt_confidence is not None:
            if vggt_confidence.ndim == 3:
                vggt_confidence = vggt_confidence.squeeze(-1)
            valid_mask = valid_mask & (vggt_confidence >= conf_threshold)

        valid_ratio = float(np.mean(valid_mask))

        if np.sum(valid_mask) < 100:
            logger.warning("[WARNING] Insufficient valid depth pixels for robust alignment. Returning raw DA-2 depth.")
            return da2_depth, {
                "scale": 1.0, "shift": 0.0, "residual_rmse": 0.0, "valid_pixel_ratio": valid_ratio
            }

        y_target = vggt_depth[valid_mask].astype(np.float64)
        x_source = da2_depth[valid_mask].astype(np.float64)

        if self.method == "median_ratio":
            # Scale-only median alignment
            ratio = y_target / (x_source + 1e-8)
            scale = float(np.median(ratio))
            shift = 0.0
        else:
            # Robust RANSAC / Least-Squares Scale + Shift
            # Solve min || s * x + t - y ||_2
            A = np.vstack([x_source, np.ones_like(x_source)]).T
            # RANSAC / Huber M-estimator
            scale, shift = self._robust_lstsq(A, y_target)

        # Apply alignment
        aligned_depth = scale * da2_depth + shift
        aligned_depth = np.clip(aligned_depth, self.min_depth, self.max_depth).astype(np.float32)

        # Compute residual RMSE over valid mask
        residual = np.sqrt(np.mean((aligned_depth[valid_mask] - vggt_depth[valid_mask]) ** 2))

        metrics = {
            "scale": float(scale),
            "shift": float(shift),
            "residual_rmse": float(residual),
            "valid_pixel_ratio": valid_ratio
        }

        return aligned_depth, metrics

    def _robust_lstsq(self, A: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Robust Iterative Reweighted Least Squares (IRLS) for scale & shift estimation."""
        # Initial least squares solution
        try:
            sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            s, t = sol[0], sol[1]
        except Exception:
            return 1.0, 0.0

        # 3 IRLS iterations to suppress outliers/sky
        for _ in range(3):
            residuals = np.abs(A[:, 0] * s + t - y)
            med_res = np.median(residuals) + 1e-6
            weights = 1.0 / np.maximum(1.0, residuals / (1.345 * med_res))
            
            try:
                # (A^T W A) sol = A^T W y via broadcasting
                Aw = A * weights[:, None]
                sol, _, _, _ = np.linalg.lstsq(Aw, y * weights, rcond=None)
                s, t = sol[0], sol[1]
            except Exception:
                break

        return max(1e-4, float(s)), float(t)

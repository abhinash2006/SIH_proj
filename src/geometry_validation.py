import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger("GeometryValidation")

class GeometryValidator:
    """
    Validates geometric integrity of Meta VGGT 3D reconstructions, camera parameters,
    depth maps, and point clouds before spatial disaster analysis.
    """

    @staticmethod
    def validate_intrinsics(intrinsic: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[bool, str]:
        """
        Validates 3x3 camera intrinsic matrix K.
        """
        if intrinsic is None or intrinsic.shape != (3, 3):
            return False, f"Invalid intrinsic matrix shape: {getattr(intrinsic, 'shape', None)}"
        
        if not np.all(np.isfinite(intrinsic)):
            return False, "Intrinsic matrix contains NaN or Inf values."
        
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
        H, W = image_shape[:2]

        if fx <= 0 or fy <= 0:
            return False, f"Non-positive focal lengths: fx={fx:.2f}, fy={fy:.2f}"
        
        if not (-W * 0.5 <= cx <= W * 1.5) or not (-H * 0.5 <= cy <= H * 1.5):
            return False, f"Principal point outside valid bounds: cx={cx:.1f}, cy={cy:.1f} for image {W}x{H}"

        return True, "Valid intrinsics"

    @staticmethod
    def validate_extrinsics(extrinsic: np.ndarray) -> Tuple[bool, str]:
        """
        Validates 3x4 or 4x4 camera extrinsic matrix [R|t].
        """
        if extrinsic is None or extrinsic.shape not in [(3, 4), (4, 4)]:
            return False, f"Invalid extrinsic matrix shape: {getattr(extrinsic, 'shape', None)}"
        
        if not np.all(np.isfinite(extrinsic)):
            return False, "Extrinsic matrix contains NaN or Inf values."
        
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]

        # Orthogonality check R^T R ≈ I
        ortho_err = np.max(np.abs(R.T @ R - np.eye(3)))
        if ortho_err > 0.1:
            return False, f"Extrinsic rotation matrix is non-orthogonal (error: {ortho_err:.4f})"
        
        det_R = np.linalg.det(R)
        if abs(det_R - 1.0) > 0.1:
            return False, f"Extrinsic rotation matrix determinant is invalid (det: {det_R:.4f})"
        
        if np.max(np.abs(t)) > 1000.0:
            return False, f"Extrinsic translation is unreasonably large: |t|={np.max(np.abs(t)):.1f}"

        return True, "Valid extrinsics"

    @staticmethod
    def validate_depth_map(depth_map: np.ndarray) -> Tuple[bool, Dict[str, float], str]:
        """
        Validates depth map values and valid pixel ratio.
        """
        if depth_map is None or depth_map.size == 0:
            return False, {}, "Empty or None depth map"

        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(-1)

        finite_mask = np.isfinite(depth_map)
        if not np.any(finite_mask):
            return False, {"valid_ratio": 0.0}, "Depth map contains no finite values"

        valid_mask = finite_mask & (depth_map > 0.05) & (depth_map < 200.0)
        valid_ratio = float(np.mean(valid_mask))
        
        if valid_ratio < 0.05:
            return False, {"valid_ratio": valid_ratio}, f"Depth valid pixel ratio too low ({valid_ratio*100:.1f}%)"

        valid_depths = depth_map[valid_mask]
        depth_min = float(np.min(valid_depths))
        depth_max = float(np.max(valid_depths))
        depth_std = float(np.std(valid_depths))

        stats = {
            "valid_ratio": valid_ratio,
            "min_depth": depth_min,
            "max_depth": depth_max,
            "std_depth": depth_std
        }

        if depth_std < 1e-4:
            return False, stats, f"Flat/constant depth map detected (std={depth_std:.6f})"

        return True, stats, "Valid depth map"

    @staticmethod
    def validate_point_cloud(points: np.ndarray) -> Tuple[bool, Dict[str, float], str]:
        """
        Validates 3D point cloud spatial variance and point density.
        """
        if points is None or len(points) < 50:
            return False, {"point_count": len(points) if points is not None else 0}, f"Insufficient points: {len(points) if points is not None else 0}"

        if not np.all(np.isfinite(points)):
            return False, {"point_count": len(points)}, "Point cloud contains NaN or Inf values"

        var_x = float(np.var(points[:, 0]))
        var_y = float(np.var(points[:, 1]))
        var_z = float(np.var(points[:, 2]))

        min_xyz = np.min(points, axis=0)
        max_xyz = np.max(points, axis=0)
        extent = max_xyz - min_xyz

        stats = {
            "point_count": len(points),
            "var_x": var_x,
            "var_y": var_y,
            "var_z": var_z,
            "extent_x": float(extent[0]),
            "extent_y": float(extent[1]),
            "extent_z": float(extent[2])
        }

        if var_x < 1e-5 or var_y < 1e-5 or var_z < 1e-5:
            return False, stats, f"Degenerate 3D point cloud (collapsed dimensions: var_x={var_x:.6f}, var_y={var_y:.6f}, var_z={var_z:.6f})"

        if extent[0] < 0.05 and extent[1] < 0.05:
            return False, stats, f"Spatial extent too small (X extent: {extent[0]:.3f}m, Y extent: {extent[1]:.3f}m)"

        return True, stats, "Valid 3D point cloud"

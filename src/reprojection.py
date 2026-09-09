import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("ReprojectionValidator")

class ReprojectionValidator:
    """
    Validates 3D localized objects by reprojecting their 3D world coordinates (X, Y, Z)
    back to 2D image pixel space (u_proj, v_proj) through camera extrinsics and intrinsics.
    Measures the Euclidean pixel discrepancy against the original 2D bounding-box center.
    """

    @staticmethod
    def project_world_to_pixel(
        X: float,
        Y: float,
        Z: float,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Projects 3D world coordinate (X, Y, Z) to 2D pixel coordinate (u, v) in intrinsic resolution,
        and returns the camera depth Z_c.
        """
        R_w2c = extrinsic[:3, :3]
        t_w2c = extrinsic[:3, 3]
        fu, fv = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cu, cv = float(intrinsic[0, 2]), float(intrinsic[1, 2])

        pt_world = np.array([X, Y, Z], dtype=np.float64)
        pt_cam = R_w2c @ pt_world + t_w2c
        x_c, y_c, z_c = float(pt_cam[0]), float(pt_cam[1]), float(pt_cam[2])

        if z_c <= 0.001:
            # Behind camera
            return -1.0, -1.0, z_c

        u = (fu * x_c / z_c) + cu
        v = (fv * y_c / z_c) + cv
        return float(u), float(v), z_c

    @classmethod
    def validate_detection_3d(
        cls,
        x_world: Optional[float],
        y_world: Optional[float],
        z_world: Optional[float],
        u_det_orig: float,
        v_det_orig: float,
        extrinsic: np.ndarray,
        intrinsic_scaled: np.ndarray,
        orig_shape: Tuple[int, int],
        scaled_shape: Tuple[int, int],
        max_reprojection_error_px: float = 30.0
    ) -> Dict[str, Any]:
        """
        Validates a 3D localization by reprojecting to original image coordinates.
        
        Args:
            x_world, y_world, z_world: 3D world coordinates (meters)
            u_det_orig, v_det_orig: center of 2D bounding box in original image coordinates
            extrinsic: [3, 4] world-to-camera matrix [R|t]
            intrinsic_scaled: [3, 3] intrinsic matrix corresponding to scaled_shape
            orig_shape: (H_orig, W_orig)
            scaled_shape: (H_scaled, W_scaled)
            max_reprojection_error_px: error threshold in original image pixels
            
        Returns:
            Dict containing:
                - reprojection_status: "VALID" or "REJECTED" or "UNAVAILABLE"
                - reprojection_error_px: float error in pixels
                - projected_pixel_orig: (u_proj, v_proj) in original image coordinates
                - reason: diagnostic message
        """
        if x_world is None or y_world is None or z_world is None:
            return {
                "reprojection_status": "UNAVAILABLE",
                "reprojection_error_px": None,
                "projected_pixel_orig": None,
                "reason": "No 3D location available for reprojection test."
            }

        H_orig, W_orig = orig_shape[:2]
        if scaled_shape is None:
            cx = float(intrinsic_scaled[0, 2])
            cy = float(intrinsic_scaled[1, 2])
            if cx < W_orig * 0.35 and cx > 0:
                s_shape = (int(round(cy * 2.0)), int(round(cx * 2.0)))
            else:
                s_shape = (H_orig, W_orig)
        else:
            s_shape = scaled_shape

        H_scaled, W_scaled = s_shape[:2]

        # Project 3D world point to scaled intrinsic pixel space
        u_proj_s, v_proj_s, z_c = cls.project_world_to_pixel(
            x_world, y_world, z_world, extrinsic, intrinsic_scaled
        )

        if z_c <= 0.001 or u_proj_s < 0 or v_proj_s < 0:
            return {
                "reprojection_status": "REJECTED",
                "reprojection_error_px": 999.0,
                "projected_pixel_orig": None,
                "reason": f"Projected point lies behind camera or outside valid field of view (Z_cam={z_c:.2f}m)."
            }

        # Scale projected pixel back to original image space
        scale_x = float(W_orig) / float(W_scaled) if W_scaled > 0 else 1.0
        scale_y = float(H_orig) / float(H_scaled) if H_scaled > 0 else 1.0

        u_proj_orig = u_proj_s * scale_x
        v_proj_orig = v_proj_s * scale_y

        # Compute Euclidean distance in original image pixel space
        error_px = float(np.sqrt((u_proj_orig - u_det_orig)**2 + (v_proj_orig - v_det_orig)**2))

        is_valid = error_px <= max_reprojection_error_px
        status = "VALID" if is_valid else "REJECTED"
        reason = (
            f"Reprojection error is {error_px:.1f}px (within {max_reprojection_error_px:.1f}px threshold)."
            if is_valid else
            f"High reprojection error ({error_px:.1f}px exceeds {max_reprojection_error_px:.1f}px threshold)."
        )

        return {
            "reprojection_status": status,
            "reprojection_error_px": round(error_px, 2),
            "projected_pixel_orig": (round(u_proj_orig, 1), round(v_proj_orig, 1)),
            "camera_depth_m": round(z_c, 2),
            "reason": reason
        }

    @classmethod
    def validate_reprojection(
        cls,
        point_3d: Any,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        original_pixel_uv: Tuple[float, float],
        original_image_shape: Tuple[int, int] = (1080, 1920),
        scaled_shape: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Validates reprojection directly from a 3D point array and original (u, v).
        Correctly handles intrinsic resolution scaling against original drone image coordinates.
        """
        if point_3d is None:
            return {"is_in_front": False, "pixel_discrepancy": 999.0, "reprojection_status": "UNAVAILABLE"}

        pt = np.asarray(point_3d, dtype=np.float64).flatten()
        X, Y, Z = float(pt[0]), float(pt[1]), float(pt[2])
        u_det, v_det = float(original_pixel_uv[0]), float(original_pixel_uv[1])

        H_orig, W_orig = original_image_shape[:2]
        
        # Auto-detect intrinsic coordinate grid if scaled_shape was not explicitly provided
        if scaled_shape is None:
            cx = float(intrinsics[0, 2])
            cy = float(intrinsics[1, 2])
            if cx < W_orig * 0.35 and cx > 0:
                s_shape = (int(round(cy * 2.0)), int(round(cx * 2.0)))
            else:
                s_shape = (H_orig, W_orig)
        else:
            s_shape = scaled_shape

        u_proj, v_proj, z_c = cls.project_world_to_pixel(X, Y, Z, extrinsics, intrinsics)
        is_front = bool(z_c > 0.001)

        if not is_front:
            return {
                "is_in_front": False,
                "pixel_discrepancy": 999.0,
                "reprojection_status": "BEHIND_CAMERA",
                "reprojection_error_px": 999.0
            }

        scale_x = float(W_orig) / float(s_shape[1]) if s_shape[1] > 0 else 1.0
        scale_y = float(H_orig) / float(s_shape[0]) if s_shape[0] > 0 else 1.0
        u_proj_orig = u_proj * scale_x
        v_proj_orig = v_proj * scale_y

        error_px = float(np.hypot(u_proj_orig - u_det, v_proj_orig - v_det))
        status = "HIGH_CONFIDENCE" if error_px < 5.0 else ("VALID" if error_px < 25.0 else "REJECTED")

        return {
            "is_in_front": True,
            "pixel_discrepancy": round(error_px, 2),
            "reprojection_error_px": round(error_px, 2),
            "reprojection_status": status,
            "projected_uv": (round(u_proj_orig, 1), round(v_proj_orig, 1)),
            "camera_depth_m": round(z_c, 3)
        }

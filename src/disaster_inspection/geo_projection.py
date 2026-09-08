import numpy as np
from typing import Tuple, Dict, Any, Optional

class GeoProjection3D:
    """
    Projects 2D image pixel coordinates (cx, cy) and predicted depth into 3D world coordinates (X, Y, Z)
    using VGGT camera intrinsics K and extrinsics [R|t]. Handles optional GPS geo-referencing.
    """

    @staticmethod
    def pixel_depth_to_camera_coords(
        cx: int, cy: int, depth: float, intrinsic: np.ndarray
    ) -> Tuple[float, float, float]:
        """Unprojects pixel (cx, cy) with depth d to camera coordinates (x_c, y_c, z_c)."""
        fu, fv = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cu, cv = float(intrinsic[0, 2]), float(intrinsic[1, 2])

        x_c = (cx - cu) * depth / fu
        y_c = (cy - cv) * depth / fv
        z_c = depth
        return x_c, y_c, z_c

    @staticmethod
    def camera_to_world_coords(
        x_c: float, y_c: float, z_c: float, extrinsic: np.ndarray
    ) -> Tuple[float, float, float]:
        """Transforms camera coordinates (x_c, y_c, z_c) into world coordinates (X, Y, Z)."""
        # Extrinsic: [3, 4] OpenCV convention (cam_from_world)
        R_w2c = extrinsic[:3, :3]
        t_w2c = extrinsic[:3, 3]

        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c

        pt_cam = np.array([x_c, y_c, z_c], dtype=np.float64)
        pt_world = R_c2w @ pt_cam + t_c2w
        return float(pt_world[0]), float(pt_world[1]), float(pt_world[2])

    @classmethod
    def project_detection_to_3d(
        cls,
        detection: Dict[str, Any],
        depth_map: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        gps_metadata: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculates 3D coordinates (X, Y, Z) for a 2D object detection.
        """
        cx, cy = detection["pixel_center"]
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(-1)
        H, W = depth_map.shape[:2]

        # Clamp pixel coordinates
        cx_clamped = max(0, min(W - 1, cx))
        cy_clamped = max(0, min(H - 1, cy))

        # Sample depth around center (3x3 median)
        y_min, y_max = max(0, cy_clamped - 1), min(H, cy_clamped + 2)
        x_min, x_max = max(0, cx_clamped - 1), min(W, cx_clamped + 2)
        depth_patch = depth_map[y_min:y_max, x_min:x_max]
        valid_depths = depth_patch[depth_patch > 0.1]

        depth_val = float(np.median(valid_depths)) if len(valid_depths) > 0 else float(depth_map[cy_clamped, cx_clamped])

        # Project 2D -> 3D
        x_c, y_c, z_c = cls.pixel_depth_to_camera_coords(cx_clamped, cy_clamped, depth_val, intrinsic)
        X, Y, Z = cls.camera_to_world_coords(x_c, y_c, z_c, extrinsic)

        result = detection.copy()
        result.update({
            "depth_m": round(depth_val, 2),
            "x_m": round(X, 2),
            "y_m": round(Y, 2),
            "z_m": round(Z, 2),
            "coordinate_system": "Local reconstruction coordinates"
        })

        if gps_metadata and "latitude" in gps_metadata:
            # Simple local tangential approximation if drone GPS is available
            lat0 = gps_metadata["latitude"]
            lon0 = gps_metadata["longitude"]
            alt0 = gps_metadata.get("altitude", 0.0)
            
            # 1 meter ~ 1 / 111139 degrees latitude
            obj_lat = lat0 + (Y / 111139.0)
            obj_lon = lon0 + (X / (111139.0 * np.cos(np.radians(lat0))))
            obj_alt = alt0 + Z
            
            result["latitude"] = round(obj_lat, 6)
            result["longitude"] = round(obj_lon, 6)
            result["altitude_m"] = round(obj_alt, 2)
            result["coordinate_system"] = "WGS84 GPS Geographic coordinates"

        return result

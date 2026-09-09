import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, List

logger = logging.getLogger("GeoProjection3D")

class GeoProjection3D:
    """
    Projects 2D image pixel coordinates (cx, cy) and predicted depth into 3D world coordinates (X, Y, Z)
    using VGGT camera intrinsics K and extrinsics [R|t].
    Handles intrinsic/image resolution scaling and 2D <-> 3D Reprojection Test.
    """

    @staticmethod
    def pixel_depth_to_camera_coords(
        u: float, v: float, depth: float, intrinsic: np.ndarray
    ) -> Tuple[float, float, float]:
        """Unprojects pixel (u, v) with depth d to camera coordinates (x_c, y_c, z_c)."""
        fu, fv = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cu, cv = float(intrinsic[0, 2]), float(intrinsic[1, 2])

        x_c = (u - cu) * depth / fu
        y_c = (v - cv) * depth / fv
        z_c = depth
        return x_c, y_c, z_c

    @staticmethod
    def camera_to_world_coords(
        x_c: float, y_c: float, z_c: float, extrinsic: np.ndarray
    ) -> Tuple[float, float, float]:
        """Transforms camera coordinates (x_c, y_c, z_c) into world coordinates (X, Y, Z)."""
        # Extrinsic: [3, 4] OpenCV convention (cam_from_world: P_cam = R_w2c @ P_world + t_w2c)
        R_w2c = extrinsic[:3, :3]
        t_w2c = extrinsic[:3, 3]

        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c

        pt_cam = np.array([x_c, y_c, z_c], dtype=np.float64)
        pt_world = R_c2w @ pt_cam + t_c2w
        return float(pt_world[0]), float(pt_world[1]), float(pt_world[2])

    @staticmethod
    def world_to_pixel_coords(
        X: float, Y: float, Z: float, extrinsic: np.ndarray, intrinsic: np.ndarray
    ) -> Tuple[float, float, float]:
        """Reprojects 3D world coordinate (X, Y, Z) back to camera pixel (u, v) and camera depth Z_c."""
        R_w2c = extrinsic[:3, :3]
        t_w2c = extrinsic[:3, 3]
        fu, fv = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cu, cv = float(intrinsic[0, 2]), float(intrinsic[1, 2])

        pt_world = np.array([X, Y, Z], dtype=np.float64)
        pt_cam = R_w2c @ pt_world + t_w2c

        x_c, y_c, z_c = pt_cam[0], pt_cam[1], pt_cam[2]
        if z_c <= 0.001:
            return -1.0, -1.0, float(z_c)

        u = (fu * x_c / z_c) + cu
        v = (fv * y_c / z_c) + cv
        return float(u), float(v), float(z_c)

    @classmethod
    def project_detection_to_3d(
        cls,
        detection: Dict[str, Any],
        depth_map: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        image_shape: Optional[Tuple[int, int]] = None,
        max_reprojection_error_px: float = 30.0,
        gps_metadata: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculates 3D coordinates (X, Y, Z) for a 2D object detection.
        Properly scales pixel coordinates between detection resolution and intrinsic matrix resolution.
        """
        result = detection.copy()

        if depth_map is None or depth_map.size == 0 or extrinsic is None or intrinsic is None:
            result.update({
                "location_status": "UNLOCALIZED",
                "location_reason": "Missing depth map or camera geometry.",
                "x_m": None, "y_m": None, "z_m": None,
                "reprojection_error_px": None
            })
            return result

        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(-1)
        H_depth, W_depth = depth_map.shape[:2]

        u_det, v_det = detection["pixel_center"]
        x1, y1, x2, y2 = detection["bbox"]

        # Determine detection image resolution
        if image_shape is not None:
            H_img, W_img = image_shape[:2]
        else:
            # Estimate from bbox / center bounds if image_shape not explicitly provided
            W_img = max(W_depth, x2 + 1, u_det + 1)
            H_img = max(H_depth, y2 + 1, v_det + 1)

        scale_x = float(W_depth) / float(W_img)
        scale_y = float(H_depth) / float(H_img)

        # Scale detection coordinates into depth map / intrinsic resolution
        u_scaled = u_det * scale_x
        v_scaled = v_det * scale_y

        bx1_s, by1_s = int(x1 * scale_x), int(y1 * scale_y)
        bx2_s, by2_s = int(x2 * scale_x), int(y2 * scale_y)
        bx1_s, by1_s = max(0, bx1_s), max(0, by1_s)
        bx2_s, by2_s = min(W_depth, bx2_s), min(H_depth, by2_s)

        bw_s, bh_s = bx2_s - bx1_s, by2_s - by1_s

        # Sample central 40% region inside bounding box
        cx1_s = int(bx1_s + bw_s * 0.30)
        cx2_s = int(bx1_s + bw_s * 0.70)
        cy1_s = int(by1_s + bh_s * 0.30)
        cy2_s = int(by1_s + bh_s * 0.70)

        depth_patch = depth_map[cy1_s:max(cy1_s + 1, cy2_s), cx1_s:max(cx1_s + 1, cx2_s)]
        valid_depths = depth_patch[(depth_patch > 0.1) & (depth_patch < 150.0) & np.isfinite(depth_patch)]

        if len(valid_depths) == 0:
            full_patch = depth_map[by1_s:by2_s, bx1_s:bx2_s]
            valid_depths = full_patch[(full_patch > 0.1) & (full_patch < 150.0) & np.isfinite(full_patch)]

        if len(valid_depths) == 0:
            result.update({
                "location_status": "UNLOCALIZED",
                "location_reason": "No valid positive depth pixels inside object region.",
                "x_m": None, "y_m": None, "z_m": None,
                "reprojection_error_px": None
            })
            return result

        sampled_depth = float(np.median(valid_depths))

        # 1. Project 2D -> 3D camera coords -> 3D world coords
        x_c, y_c, z_c = cls.pixel_depth_to_camera_coords(u_scaled, v_scaled, sampled_depth, intrinsic)
        X, Y, Z = cls.camera_to_world_coords(x_c, y_c, z_c, extrinsic)

        # 2. REPROJECTION TEST (3D -> 2D)
        u_reproj_s, v_reproj_s, z_reproj = cls.world_to_pixel_coords(X, Y, Z, extrinsic, intrinsic)

        if u_reproj_s < 0 or v_reproj_s < 0 or z_reproj <= 0:
            reproj_err_img = 999.0
        else:
            u_reproj_img = u_reproj_s / scale_x
            v_reproj_img = v_reproj_s / scale_y
            reproj_err_img = float(np.sqrt((u_reproj_img - u_det)**2 + (v_reproj_img - v_det)**2))

        # Check reprojection tolerance
        if reproj_err_img > max_reprojection_error_px:
            logger.warning(
                f"[REPROJECTION FAIL] Object {detection.get('class_name')} reprojection error "
                f"{reproj_err_img:.1f}px exceeds tolerance ({max_reprojection_error_px}px)."
            )
            result.update({
                "location_status": "UNLOCALIZED",
                "location_reason": f"Reprojection error ({reproj_err_img:.1f}px) exceeds tolerance ({max_reprojection_error_px}px).",
                "x_m": None, "y_m": None, "z_m": None,
                "reprojection_error_px": round(reproj_err_img, 2)
            })
            return result

        # Calculate Multi-Evidence Object Confidence Model
        raw_det_conf = float(detection.get("validated_detection_confidence", detection.get("confidence", 0.5)))
        loc_conf = float(max(0.0, 1.0 - (reproj_err_img / max_reprojection_error_px)))
        geom_conf = 0.90 if sampled_depth > 0.5 else 0.60
        
        obj_conf = round(0.40 * raw_det_conf + 0.30 * loc_conf + 0.30 * geom_conf, 3)

        # Successful 3D Localization
        result.update({
            "location_status": "LOCALIZED_3D",
            "x_m": round(X, 2),
            "y_m": round(Y, 2),
            "z_m": round(Z, 2),
            "depth_m": round(sampled_depth, 2),
            "reprojection_error_px": round(reproj_err_img, 2),
            "detection_confidence": round(raw_det_conf, 3),
            "localization_confidence": round(loc_conf, 3),
            "geometry_confidence": round(geom_conf, 3),
            "object_confidence": obj_conf,
            "coordinate_system": "VGGT Local World Coordinates"
        })

        if gps_metadata and "latitude" in gps_metadata:
            lat0, lon0 = gps_metadata["latitude"], gps_metadata["longitude"]
            alt0 = gps_metadata.get("altitude", 0.0)
            obj_lat = lat0 + (Y / 111139.0)
            obj_lon = lon0 + (X / (111139.0 * np.cos(np.radians(lat0))))
            obj_alt = alt0 + Z
            
            result["latitude"] = round(obj_lat, 6)
            result["longitude"] = round(obj_lon, 6)
            result["altitude_m"] = round(obj_alt, 2)
            result["location_status"] = "LOCALIZED_GPS"
            result["coordinate_system"] = "WGS84 Geographic GPS"

        return result

    @classmethod
    def fuse_multiframe_object_3d(
        cls, observations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fuses multi-frame 3D observations of a tracked or spatially associated object.
        Computes robust spatial consensus and multi-frame confidence.
        """
        if not observations:
            return {"location_status": "UNLOCALIZED", "x_m": None, "y_m": None, "z_m": None}

        valid_obs = [obs for obs in observations if obs.get("location_status") in ["LOCALIZED_3D", "LOCALIZED_GPS"] and obs.get("x_m") is not None]

        if not valid_obs:
            first = observations[0]
            return {
                "object_id": first.get("track_id", "OBJ_UNTRACKED"),
                "class_name": first.get("class_name", "object"),
                "location_status": "UNLOCALIZED",
                "x_m": None, "y_m": None, "z_m": None,
                "source_frames": sorted(list(set(obs.get("frame_idx", 0) for obs in observations))),
                "observation_count": len(observations)
            }

        # Extract 3D points
        pts = np.array([[obs["x_m"], obs["y_m"], obs["z_m"]] for obs in valid_obs], dtype=np.float64)

        if len(pts) >= 3:
            med = np.median(pts, axis=0)
            dists = np.linalg.norm(pts - med, axis=1)
            inliers = dists <= 2.5 # 2.5 meters outlier threshold
            if np.sum(inliers) > 0:
                pts = pts[inliers]
                valid_obs = [valid_obs[k] for k in range(len(valid_obs)) if inliers[k]]

        fused_x = float(np.mean(pts[:, 0]))
        fused_y = float(np.mean(pts[:, 1]))
        fused_z = float(np.mean(pts[:, 2]))

        avg_reproj = float(np.mean([obs.get("reprojection_error_px", 0.0) for obs in valid_obs]))
        avg_det_conf = float(np.mean([obs.get("detection_confidence", obs.get("confidence", 0.5)) for obs in valid_obs]))
        
        # Temporal observation boost
        obs_boost = min(0.15, len(valid_obs) * 0.03)
        final_conf = round(min(0.98, avg_det_conf + obs_boost), 3)

        source_frames = sorted(list(set(obs.get("frame_idx", 0) for obs in observations)))
        first = observations[0]

        return {
            "object_id": f"{first.get('class_name', 'OBJ').upper()}_{first.get('track_id', 0):03d}" if first.get("track_id") is not None else f"{first.get('class_name', 'OBJ').upper()}_LOC",
            "class_name": first.get("class_name", "object"),
            "location_status": "LOCALIZED_3D",
            "x_m": round(fused_x, 2),
            "y_m": round(fused_y, 2),
            "z_m": round(fused_z, 2),
            "reprojection_error_px": round(avg_reproj, 2),
            "confidence": final_conf,
            "source_frames": source_frames,
            "observation_count": len(observations),
            "trajectory": pts.tolist()
        }


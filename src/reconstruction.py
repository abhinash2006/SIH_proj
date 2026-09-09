import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
import cv2
import open3d as o3d

from src.rescue_target import ValidatedRescueTarget, is_validated_rescue_target

logger = logging.getLogger(__name__)


class PointCloudReconstructor:
    """
    Constructs, formats, and exports 3D point clouds from VGGT predictions.
    Supports confidence masking, depth unprojection vs direct point maps,
    Open3D PLY serialization, camera flight trajectory, and SAR validated rescue markers.
    """

    @staticmethod
    def build_pointcloud_from_predictions(
        predictions: Dict[str, Any],
        use_depth_unprojection: bool = True,
        confidence_threshold: float = 1.1,
        max_points: Optional[int] = 1_000_000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts 3D points, RGB colors, and confidence values from VGGT predictions.
        """
        if use_depth_unprojection:
            pts_all = predictions["unprojected_points"] # [S, H, W, 3]
            conf_all = predictions["depth_conf"]        # [S, H, W]
        else:
            pts_all = predictions["point_maps"]         # [S, H, W, 3]
            conf_all = predictions["point_conf"]        # [S, H, W]

        images_rgb = predictions["images"]              # [S, H, W, 3] uint8

        S, H, W, _ = pts_all.shape
        
        flat_pts = pts_all.reshape(-1, 3)
        flat_conf = conf_all.reshape(-1)
        flat_colors = images_rgb.reshape(-1, 3).astype(np.float32) / 255.0

        valid_mask = (
            np.isfinite(flat_pts).all(axis=1) &
            (flat_conf >= confidence_threshold) &
            (np.linalg.norm(flat_pts, axis=1) > 1e-4) &
            (np.linalg.norm(flat_pts, axis=1) < 500.0)
        )

        pts = flat_pts[valid_mask]
        cols = flat_colors[valid_mask]
        confs = flat_conf[valid_mask]

        if max_points is not None and len(pts) > max_points:
            indices = np.random.choice(len(pts), max_points, replace=False)
            pts = pts[indices]
            cols = cols[indices]
            confs = confs[indices]

        return pts.astype(np.float32), cols.astype(np.float32), confs.astype(np.float32)

    @staticmethod
    def save_ply(
        points: np.ndarray,
        colors: np.ndarray,
        output_path: Union[str, Path]
    ) -> str:
        """
        Saves points and RGB colors to an ASCII or binary PLY file using Open3D.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pcd = o3d.geometry.PointCloud()
        if len(points) > 0:
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            if colors is not None and len(colors) == len(points):
                if colors.max() > 1.0:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
        logger.info(f"Saved {len(points):,} points to {output_path}")
        return str(output_path)

    @staticmethod
    def generate_markers_pointcloud(
        objects_3d: List[Union[ValidatedRescueTarget, Dict[str, Any]]],
        radius: float = 0.25,
        num_samples: int = 120
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates 3D colored sphere markers for validated victims and localized vehicles.
        
        CRITICAL DIRECTIVE 2:
        Only ValidatedRescueTarget objects may generate human rescue markers.
        Raw YOLO detections alone must NEVER create a rescue marker.
        """
        if not objects_3d:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        marker_points = []
        marker_colors = []

        # Pre-generate unit sphere points
        phi = np.random.uniform(0, 2 * np.pi, num_samples)
        costheta = np.random.uniform(-1, 1, num_samples)
        u = np.random.uniform(0, 1, num_samples)
        theta = np.arccos(costheta)
        r = radius * (u ** (1 / 3))
        unit_sphere_x = r * np.sin(theta) * np.cos(phi)
        unit_sphere_y = r * np.sin(theta) * np.sin(phi)
        unit_sphere_z = r * np.cos(theta)
        unit_sphere = np.stack([unit_sphere_x, unit_sphere_y, unit_sphere_z], axis=1)

        for obj in objects_3d:
            if isinstance(obj, ValidatedRescueTarget):
                if not is_validated_rescue_target(obj):
                    logger.info(f"[RESCUE MARKER GATED] Target {obj.target_id} skipped: Not a VALIDATED RESCUE TARGET (reprojection error > 25px or unlocalized).")
                    continue
                pos = obj.location_3d
                priority = obj.rescue_priority
                cls_name = "person"
            elif isinstance(obj, dict):
                cls_name = str(obj.get("class_name", "")).lower()
                # Enforcement: If person, verify it is a validated rescue target
                if "person" in cls_name or "victim" in cls_name:
                    if not is_validated_rescue_target(obj):
                        logger.info(f"[RESCUE MARKER BLOCKED] Object {obj.get('id', 'N/A')} rejected: Raw YOLO detection or unvalidated reprojection cannot generate a rescue marker.")
                        continue
                pos = obj.get("point_3d", obj.get("location_3d"))
                priority = str(obj.get("priority", obj.get("rescue_priority", "LOW"))).upper()
            else:
                continue

            if pos is None or len(pos) != 3 or not np.all(np.isfinite(pos)):
                continue

            cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])

            if "person" in cls_name or "victim" in cls_name:
                if priority == "HIGH":
                    color = [1.0, 0.1, 0.1]   # Red
                elif priority == "MEDIUM":
                    color = [1.0, 0.65, 0.0]  # Amber
                else:
                    color = [0.1, 0.95, 0.2]  # Green
            elif any(k in cls_name for k in ["vehicle", "car", "truck", "boat"]):
                color = [0.1, 0.4, 1.0]       # Blue
            else:
                color = [0.9, 0.9, 0.1]       # Yellow

            sphere_pts = unit_sphere + np.array([cx, cy, cz])
            marker_points.append(sphere_pts)
            marker_colors.append(np.tile(color, (num_samples, 1)))

        if not marker_points:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        pts = np.vstack(marker_points).astype(np.float32)
        cols = np.vstack(marker_colors).astype(np.float32)
        return pts, cols

    @staticmethod
    def generate_camera_trajectory_points(
        extrinsics: np.ndarray,
        num_interp: int = 40
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts camera optical centers and generates an interpolated flight path
        with dense points colored Cyan [0.0, 0.9, 1.0] and golden waypoint spheres.
        """
        if extrinsics is None or len(extrinsics) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        centers = []
        for ext in extrinsics:
            R = ext[:3, :3]
            t = ext[:3, 3] if ext.shape[1] > 3 else np.zeros(3)
            C = -R.T @ t
            centers.append(C)
        centers = np.array(centers, dtype=np.float32)

        if len(centers) <= 1:
            return centers, np.tile([0.0, 0.9, 1.0], (len(centers), 1)).astype(np.float32)

        span = float(np.linalg.norm(np.max(centers, axis=0) - np.min(centers, axis=0)))
        w_rad = max(0.02, min(0.15, span * 0.03))

        path_pts = []
        path_cols = []

        # 1. Waypoint sphere points
        num_sp = 50
        phi = np.random.uniform(0, 2 * np.pi, num_sp)
        costheta = np.random.uniform(-1, 1, num_sp)
        u = np.random.uniform(0, 1, num_sp)
        theta = np.arccos(costheta)
        r = w_rad * (u ** (1 / 3))
        unit_sp = np.stack([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)], axis=1)

        for i, c in enumerate(centers):
            sp_pts = unit_sp + c
            path_pts.append(sp_pts)
            # Start Green, End Red, Intermediates Amber
            if i == 0:
                col = [0.1, 0.95, 0.2]
            elif i == len(centers) - 1:
                col = [1.0, 0.2, 0.2]
            else:
                col = [1.0, 0.75, 0.0]
            path_cols.append(np.tile(col, (num_sp, 1)))

        # 2. Dense interpolated segment points with radial thickness
        for i in range(len(centers) - 1):
            p0 = centers[i]
            p1 = centers[i + 1]
            alphas = np.linspace(0.0, 1.0, num_interp, endpoint=False)[:, None]
            segment = (1.0 - alphas) * p0 + alphas * p1
            offsets = [
                np.array([0, 0, 0], dtype=np.float32),
                np.array([w_rad * 0.3, 0, 0], dtype=np.float32),
                np.array([-w_rad * 0.3, 0, 0], dtype=np.float32),
                np.array([0, w_rad * 0.3, 0], dtype=np.float32),
                np.array([0, -w_rad * 0.3, 0], dtype=np.float32)
            ]
            for offset_dir in offsets:
                thick_segment = segment + offset_dir
                path_pts.append(thick_segment)
                path_cols.append(np.tile([0.0, 0.9, 1.0], (len(thick_segment), 1)))

        dense_path = np.vstack(path_pts).astype(np.float32)
        dense_colors = np.vstack(path_cols).astype(np.float32)

        return dense_path, dense_colors

    @staticmethod
    def generate_flood_region_points(
        water_masks: List[np.ndarray],
        point_maps: np.ndarray,
        max_flood_points: int = 80_000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates 3D point cloud representing the flood surface in RESCUE INTELLIGENCE mode.
        Water points are colored deep aqua/cyan [0.05, 0.65, 0.85].
        """
        if not water_masks or point_maps is None or len(point_maps) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        flood_pts_list = []
        S, H_v, W_v = point_maps.shape[:3]

        for i, mask in enumerate(water_masks):
            if i >= S:
                break
            if mask is None or np.count_nonzero(mask) == 0:
                continue

            # Resize mask to VGGT dimensions
            if mask.shape[:2] != (H_v, W_v):
                mask_vggt = cv2.resize(mask, (W_v, H_v), interpolation=cv2.INTER_NEAREST)
            else:
                mask_vggt = mask

            pts_frame = point_maps[i]  # [H_v, W_v, 3]
            valid_water = (mask_vggt > 0) & np.isfinite(pts_frame).all(axis=-1)
            f_pts = pts_frame[valid_water]

            if len(f_pts) > 0:
                flood_pts_list.append(f_pts)

        if not flood_pts_list:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        all_f_pts = np.vstack(flood_pts_list).astype(np.float32)
        if len(all_f_pts) > max_flood_points:
            sub_idx = np.random.choice(len(all_f_pts), max_flood_points, replace=False)
            all_f_pts = all_f_pts[sub_idx]

        water_color = np.tile([0.05, 0.65, 0.85], (len(all_f_pts), 1)).astype(np.float32)
        return all_f_pts, water_color

    @classmethod
    def build_rescue_intelligence_scene(
        cls,
        base_points: np.ndarray,
        base_colors: np.ndarray,
        validated_rescue_targets: List[ValidatedRescueTarget],
        localized_vehicles: Optional[List[Dict[str, Any]]] = None,
        water_masks: Optional[List[np.ndarray]] = None,
        point_maps: Optional[np.ndarray] = None,
        extrinsics: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds composite RESCUE INTELLIGENCE 3D Scene:
        - 3D Environment (base points)
        - Flood region overlay
        - Validated rescue targets (priority-colored spheres)
        - Localized vehicles
        - Camera flight trajectory
        """
        all_pts = [base_points]
        all_cols = [base_colors]

        # 1. Flood region points
        if water_masks and point_maps is not None:
            fl_pts, fl_cols = cls.generate_flood_region_points(water_masks, point_maps)
            if len(fl_pts) > 0:
                all_pts.append(fl_pts)
                all_cols.append(fl_cols)

        # 2. Validated rescue targets + vehicles markers
        all_marker_objs = list(validated_rescue_targets)
        if localized_vehicles:
            all_marker_objs.extend(localized_vehicles)

        if all_marker_objs:
            m_pts, m_cols = cls.generate_markers_pointcloud(all_marker_objs)
            if len(m_pts) > 0:
                all_pts.append(m_pts)
                all_cols.append(m_cols)

        # 3. Camera flight path
        if extrinsics is not None:
            c_pts, c_cols = cls.generate_camera_trajectory_points(extrinsics)
            if len(c_pts) > 0:
                all_pts.append(c_pts)
                all_cols.append(c_cols)

        merged_pts = np.vstack(all_pts).astype(np.float32)
        merged_cols = np.vstack(all_cols).astype(np.float32)
        return merged_pts, merged_cols

    @classmethod
    def save_scene_with_markers(
        cls,
        base_points: np.ndarray,
        base_colors: np.ndarray,
        output_path: Union[str, Path],
        objects_3d: Optional[List[Any]] = None,
        extrinsics: Optional[np.ndarray] = None
    ) -> str:
        """
        Saves merged interactive point cloud containing terrain + victims/vehicles + camera path.
        """
        all_pts = [base_points]
        all_cols = [base_colors]

        if objects_3d:
            m_pts, m_cols = cls.generate_markers_pointcloud(objects_3d)
            if len(m_pts) > 0:
                all_pts.append(m_pts)
                all_cols.append(m_cols)

        if extrinsics is not None:
            c_pts, c_cols = cls.generate_camera_trajectory_points(extrinsics)
            if len(c_pts) > 0:
                all_pts.append(c_pts)
                all_cols.append(c_cols)

        merged_pts = np.vstack(all_pts).astype(np.float32)
        merged_cols = np.vstack(all_cols).astype(np.float32)
        return cls.save_ply(merged_pts, merged_cols, output_path)

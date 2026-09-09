"""
Export Manager for UAV Disaster Intelligence System.

Packages, formats, and exports all 3D geometry stages, canonical incident tables,
rescue priorities, and camera trajectories into standard formats:
- PLY point clouds (Raw, Depth, Filtered)
- PLY / OBJ surface mesh
- Camera trajectory (PLY / JSON)
- Incidents JSON & CSV
- Detected objects CSV
- Rescue priorities CSV
- COLMAP reconstruction bundle ZIP
- Master disaster package ZIP
"""

import os
import csv
import json
import zipfile
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import open3d as o3d

logger = logging.getLogger("ExportManager")


class ExportManager:
    """
    Centralized export packaging for 3D reconstructions, detections, and incident reports.
    """

    @staticmethod
    def export_geometry_stages(
        output_dir: Union[str, Path],
        raw_pcd: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        depth_pcd: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        filtered_pcd: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        mesh_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Saves all 4 geometry stages to disk in output_dir.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        # 1. Raw Point Cloud
        if raw_pcd is not None and len(raw_pcd[0]) > 0:
            raw_path = out_dir / "reconstruction_raw.ply"
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(raw_pcd[0].astype(np.float64))
            if raw_pcd[1] is not None and len(raw_pcd[1]) == len(raw_pcd[0]):
                pcd.colors = o3d.utility.Vector3dVector(raw_pcd[1].astype(np.float64))
            o3d.io.write_point_cloud(str(raw_path), pcd, write_ascii=False)
            results["raw_ply"] = str(raw_path)

        # 2. Depth Point Cloud
        if depth_pcd is not None and len(depth_pcd[0]) > 0:
            depth_path = out_dir / "reconstruction_depth.ply"
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(depth_pcd[0].astype(np.float64))
            if depth_pcd[1] is not None and len(depth_pcd[1]) == len(depth_pcd[0]):
                pcd.colors = o3d.utility.Vector3dVector(depth_pcd[1].astype(np.float64))
            o3d.io.write_point_cloud(str(depth_path), pcd, write_ascii=False)
            results["depth_ply"] = str(depth_path)

        # 3. Filtered Point Cloud (DEFAULT view)
        if filtered_pcd is not None and len(filtered_pcd[0]) > 0:
            filtered_path = out_dir / "reconstruction_filtered.ply"
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filtered_pcd[0].astype(np.float64))
            if filtered_pcd[1] is not None and len(filtered_pcd[1]) == len(filtered_pcd[0]):
                pcd.colors = o3d.utility.Vector3dVector(filtered_pcd[1].astype(np.float64))
            try:
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            except Exception:
                pass
            o3d.io.write_point_cloud(str(filtered_path), pcd, write_ascii=False)
            results["filtered_ply"] = str(filtered_path)

        # 4. Surface Mesh
        if mesh_path and os.path.exists(mesh_path):
            results["mesh_ply"] = str(mesh_path)

        return results

    @staticmethod
    def _create_cylinder_segment(p0: np.ndarray, p1: np.ndarray, radius: float = 0.03) -> Optional[o3d.geometry.TriangleMesh]:
        v = p1 - p0
        height = float(np.linalg.norm(v))
        if height < 1e-6:
            return None
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=16)
        z_axis = np.array([0, 0, 1], dtype=np.float64)
        v_norm = v / height
        axis = np.cross(z_axis, v_norm)
        axis_len = float(np.linalg.norm(axis))
        if axis_len < 1e-6:
            R = -np.eye(3) if np.dot(z_axis, v_norm) < 0 else np.eye(3)
        else:
            axis = axis / axis_len
            angle = float(np.arccos(np.clip(np.dot(z_axis, v_norm), -1.0, 1.0)))
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        cyl.rotate(R, center=(0, 0, 0))
        cyl.translate((p0 + p1) / 2.0)
        return cyl

    @classmethod
    def export_camera_trajectory(
        cls,
        extrinsics: np.ndarray,
        output_path: Union[str, Path]
    ) -> str:
        """
        Exports camera trajectory as a Babylon.js-compatible 3D TriangleMesh PLY.
        Constructs waypoint spheres, orientation cones, and cylindrical flight tubes.
        Avoids unsupported LineSet 'element edge' which crashes browser 3D viewers.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if extrinsics is not None and len(extrinsics.shape) >= 2:
            centers = []
            rotations = []
            ext_list = extrinsics if len(extrinsics.shape) == 3 else [extrinsics]
            for ext in ext_list:
                R = ext[:3, :3]
                t = ext[:3, 3] if ext.shape[1] > 3 else np.zeros(3)
                C = -R.T @ t
                centers.append(C)
                rotations.append(R)
            centers = np.array(centers, dtype=np.float64)
        else:
            centers = np.zeros((1, 3), dtype=np.float64)
            rotations = [np.eye(3)]

        # Determine dynamic physical scale based on trajectory span
        if len(centers) > 1:
            span = float(np.linalg.norm(np.max(centers, axis=0) - np.min(centers, axis=0)))
        else:
            span = 1.0
        scale = max(0.03, min(0.30, span * 0.04))
        tube_radius = scale * 0.35
        sphere_radius = scale * 0.75
        arrow_len = scale * 1.6

        meshes = []

        # 1. Camera waypoints with forward orientation indicators
        for i in range(len(centers)):
            sp = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=16)
            if i == 0:
                sp.paint_uniform_color([0.1, 0.95, 0.2])  # Green: Takeoff
            elif i == len(centers) - 1 and len(centers) > 1:
                sp.paint_uniform_color([1.0, 0.2, 0.2])   # Red: Final pose
            else:
                sp.paint_uniform_color([1.0, 0.75, 0.0])  # Amber: Waypoints
            sp.translate(centers[i])
            meshes.append(sp)

            # Direction cone pointing along camera optical axis
            if i < len(rotations):
                R_cam = rotations[i]
                forward_dir = R_cam.T @ np.array([0, 0, 1], dtype=np.float64)
                forward_dir = forward_dir / (np.linalg.norm(forward_dir) + 1e-6)
                cone_tip = centers[i] + forward_dir * arrow_len
                cone = cls._create_cylinder_segment(centers[i], cone_tip, radius=tube_radius * 0.8)
                if cone is not None:
                    cone.paint_uniform_color([1.0, 0.9, 0.1])
                    meshes.append(cone)

        # 2. Continuous cylindrical flight tubes between consecutive waypoints
        for i in range(len(centers) - 1):
            cyl = cls._create_cylinder_segment(centers[i], centers[i + 1], radius=tube_radius)
            if cyl is not None:
                cyl.paint_uniform_color([0.0, 0.85, 1.0])  # Vibrant Cyan flight path
                meshes.append(cyl)

        # Combine into unified TriangleMesh with vertex normals
        if meshes:
            full_mesh = o3d.geometry.TriangleMesh()
            for m in meshes:
                full_mesh += m
            full_mesh.compute_vertex_normals()
        else:
            full_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            full_mesh.compute_vertex_normals()

        # Save as standard triangle mesh PLY (element vertex + element face)
        o3d.io.write_triangle_mesh(str(out_path), full_mesh, write_ascii=False)
        logger.info(f"[SUCCESS] Exported camera trajectory 3D mesh: {out_path} ({len(centers)} poses, {len(full_mesh.vertices)} vertices)")
        return str(out_path)

    @staticmethod
    def export_incidents(
        incidents: List[Dict[str, Any]],
        output_dir: Union[str, Path]
    ) -> Dict[str, str]:
        """
        Exports canonical incidents to JSON and CSV formats.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        json_path = out_dir / "incidents.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(incidents, f, indent=2, default=str)
        results["incidents_json"] = str(json_path)

        csv_path = out_dir / "incidents.csv"
        if incidents:
            keys = [
                "incident_id", "incident_type", "severity", "confidence",
                "source_frame_id", "world_x", "world_y", "world_z",
                "flood_status", "summary", "timestamp"
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                for inc in incidents:
                    loc_3d = inc.get("location_3d") or {}
                    row = {
                        "incident_id": inc.get("incident_id"),
                        "incident_type": inc.get("incident_type"),
                        "severity": inc.get("severity"),
                        "confidence": inc.get("confidence"),
                        "source_frame_id": inc.get("source_frame_id"),
                        "world_x": loc_3d.get("x", "N/A"),
                        "world_y": loc_3d.get("y", "N/A"),
                        "world_z": loc_3d.get("z", "N/A"),
                        "flood_status": inc.get("flood_status"),
                        "summary": inc.get("summary"),
                        "timestamp": inc.get("timestamp")
                    }
                    writer.writerow(row)
        else:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                f.write("incident_id,incident_type,severity,confidence,source_frame_id,world_x,world_y,world_z,flood_status,summary,timestamp\n")
        results["incidents_csv"] = str(csv_path)

        return results

    @staticmethod
    def export_rescue_priorities(
        rescue_priorities: List[Dict[str, Any]],
        output_dir: Union[str, Path]
    ) -> str:
        """
        Exports rescue priority rankings to CSV.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "rescue_priorities.csv"

        keys = [
            "victim_id", "track_id", "frame_id", "rescue_priority", "priority_score",
            "isolation_type", "surrounded_pct", "nearest_water_dist_m",
            "world_x", "world_y", "world_z", "rationale"
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in rescue_priorities:
                pos = r.get("world_position") or {}
                ctx = r.get("spatial_context") or {}
                row = {
                    "victim_id": r.get("victim_id"),
                    "track_id": r.get("track_id"),
                    "frame_id": r.get("frame_id"),
                    "rescue_priority": r.get("rescue_priority"),
                    "priority_score": r.get("priority_score"),
                    "isolation_type": ctx.get("isolation_type", "N/A"),
                    "surrounded_pct": ctx.get("surrounded_percentage", 0.0),
                    "nearest_water_dist_m": ctx.get("nearest_water_distance_m", "N/A"),
                    "world_x": pos.get("x", "N/A"),
                    "world_y": pos.get("y", "N/A"),
                    "world_z": pos.get("z", "N/A"),
                    "rationale": r.get("rationale")
                }
                writer.writerow(row)

        return str(csv_path)

    @staticmethod
    def export_detected_objects(
        all_detections: List[Dict[str, Any]],
        output_dir: Union[str, Path]
    ) -> str:
        """
        Exports all 2D/3D detected objects to CSV.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "detected_objects.csv"

        keys = [
            "detection_id", "class_name", "confidence", "frame_idx",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "world_x", "world_y", "world_z", "is_tiled"
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for d in all_detections:
                bbox = d.get("bbox", [0, 0, 0, 0])
                pos_3d = d.get("point_3d")
                row = {
                    "detection_id": d.get("id", ""),
                    "class_name": d.get("class_name", ""),
                    "confidence": round(float(d.get("confidence", 0.0)), 3),
                    "frame_idx": d.get("frame_idx", 0),
                    "bbox_x1": round(float(bbox[0]), 1) if len(bbox) >= 4 else 0,
                    "bbox_y1": round(float(bbox[1]), 1) if len(bbox) >= 4 else 0,
                    "bbox_x2": round(float(bbox[2]), 1) if len(bbox) >= 4 else 0,
                    "bbox_y2": round(float(bbox[3]), 1) if len(bbox) >= 4 else 0,
                    "world_x": round(float(pos_3d[0]), 3) if pos_3d is not None else "N/A",
                    "world_y": round(float(pos_3d[1]), 3) if pos_3d is not None else "N/A",
                    "world_z": round(float(pos_3d[2]), 3) if pos_3d is not None else "N/A",
                    "is_tiled": d.get("is_tiled", False)
                }
                writer.writerow(row)

        return str(csv_path)

    @staticmethod
    def create_master_zip(
        output_dir: Union[str, Path],
        zip_filename: str = "disaster_intelligence_package.zip"
    ) -> str:
        """
        Zips all files in output_dir into a single archive.
        """
        out_dir = Path(output_dir)
        zip_path = out_dir / zip_filename

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in out_dir.glob("*"):
                if file_path.is_file() and file_path.name != zip_filename:
                    zf.write(file_path, arcname=file_path.name)

        logger.info(f"[SUCCESS] Created master archive: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
        return str(zip_path)

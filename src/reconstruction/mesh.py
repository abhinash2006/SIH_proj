"""
Mesh Reconstruction & Simplification using Open3D and Trimesh.
Generates 3D surface meshes from filtered point clouds using Screened Poisson
Surface Reconstruction or Ball Pivoting, decodes vertex colors, and exports to GLB, OBJ, and PLY.
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import open3d as o3d
import trimesh

from .pointcloud import PointCloudData

logger = logging.getLogger(__name__)


@dataclass
class MeshData:
    mesh_o3d: o3d.geometry.TriangleMesh
    num_vertices: int
    num_triangles: int
    has_vertex_colors: bool
    has_normals: bool
    export_paths: Dict[str, str]


class MeshReconstructor:
    """
    Constructs clean, manifold, textured surface meshes from oriented point clouds.
    Exports optimized binary GLB files for Three.js web rendering.
    """

    def __init__(
        self,
        method: str = "poisson",        # "poisson" or "ball_pivoting"
        poisson_depth: int = 9,
        density_quantile_cutoff: float = 0.05,
        target_triangles: int = 250000,
        simplify_mesh: bool = True,
    ):
        self.method = method.lower()
        self.poisson_depth = poisson_depth
        self.density_quantile_cutoff = density_quantile_cutoff
        self.target_triangles = target_triangles
        self.simplify_mesh = simplify_mesh

    def reconstruct(self, pcd_data: PointCloudData) -> o3d.geometry.TriangleMesh:
        """
        Reconstructs triangular surface mesh from oriented point cloud.
        """
        pcd = pcd_data.pcd_o3d

        if len(pcd.points) < 10:
            logger.warning("[MESH] Point cloud contains fewer than 10 points. Cannot reconstruct mesh.")
            return o3d.geometry.TriangleMesh()

        # Sample distances for radius estimation
        if len(pcd.points) > 3000:
            sample_pcd = pcd.random_down_sample(sampling_ratio=min(1.0, 3000.0 / len(pcd.points)))
        else:
            sample_pcd = pcd
        distances = sample_pcd.compute_nearest_neighbor_distance()
        avg_dist = float(np.mean(distances)) if len(distances) > 0 else 0.1

        # Ensure normals are computed
        if not pcd.has_normals():
            logger.info("[MESH] Estimating point normals before surface reconstruction...")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(0.2, avg_dist * 3.0), max_nn=30))
            try:
                pcd.orient_normals_consistent_tangent_plane(k=min(15, len(pcd.points) - 1))
            except Exception:
                pass

        logger.info(f"[MESH] Reconstructing surface using '{self.method.upper()}' (depth={self.poisson_depth})...")

        if self.method == "ball_pivoting":
            radii = [avg_dist, avg_dist * 2.0, avg_dist * 4.0]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
        else:
            # Screened Poisson Surface Reconstruction (default)
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=self.poisson_depth
            )

            # Filter low density boundary artifacts safely
            densities = np.asarray(densities)
            if len(densities) > 0 and self.density_quantile_cutoff > 0:
                density_threshold = np.quantile(densities, self.density_quantile_cutoff)
                vertices_to_remove = densities < density_threshold
                # Only remove if it leaves at least 10 vertices
                if np.count_nonzero(~vertices_to_remove) >= 10:
                    mesh.remove_vertices_by_mask(vertices_to_remove)
                    logger.info(f"[MESH] Removed boundary vertices below density quantile {self.density_quantile_cutoff:.2f}")

        # Fallback to Alpha Shapes or Convex Hull if mesh is degenerate or empty
        if len(mesh.triangles) == 0:
            logger.info("[MESH] Poisson produced 0 triangles. Falling back to Alpha Shape / Convex Hull.")
            alpha = max(0.5, avg_dist * 5.0)
            try:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            except Exception:
                pass
            if len(mesh.triangles) == 0:
                mesh, _ = pcd.compute_convex_hull()

        # Post-processing mesh cleanup
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # Mesh Decimation / Simplification if requested
        if self.simplify_mesh and len(mesh.triangles) > self.target_triangles:
            logger.info(f"[MESH] Decimating mesh from {len(mesh.triangles):,} to {self.target_triangles:,} triangles...")
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=self.target_triangles)
            mesh.remove_degenerate_triangles()

        # Interpolate vertex colors from nearest point cloud points if missing
        if not mesh.has_vertex_colors() and pcd.has_colors():
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            vertex_colors = []
            pcd_colors = np.asarray(pcd.colors)
            for v in mesh.vertices:
                _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
                vertex_colors.append(pcd_colors[idx[0]])
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vertex_colors))

        mesh.compute_vertex_normals()
        logger.info(f"[MESH] Reconstruction complete: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles.")
        return mesh

    def export_all_formats(
        self,
        mesh: o3d.geometry.TriangleMesh,
        base_output_dir: str,
        base_name: str = "final",
    ) -> MeshData:
        """
        Exports mesh to GLB, OBJ, and PLY.
        Uses Trimesh for optimal binary GLB web delivery.
        """
        out_dir = Path(base_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ply_path = out_dir / f"{base_name}.ply"
        obj_path = out_dir / f"{base_name}.obj"
        glb_path = out_dir / f"{base_name}.glb"

        # 1. Export PLY and OBJ via Open3D
        o3d.io.write_triangle_mesh(str(ply_path), mesh, write_ascii=False)
        o3d.io.write_triangle_mesh(str(obj_path), mesh)

        # 2. Export GLB via Trimesh
        try:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            colors = np.asarray(mesh.vertex_colors)

            # Vertex colors in RGBA uint8 for GLB
            if colors.shape[0] == vertices.shape[0]:
                rgba = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
                if rgba.shape[1] == 3:
                    alpha = np.full((len(rgba), 1), 255, dtype=np.uint8)
                    rgba = np.hstack([rgba, alpha])
                tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=rgba, process=False)
            else:
                tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

            tri_mesh.export(str(glb_path), file_type="glb")
            logger.info(f"[MESH] Exported web-optimized GLB to '{glb_path}'.")
        except Exception as e:
            logger.warning(f"[MESH] Failed to export GLB via Trimesh ({e}). Falling back to PLY/OBJ only.")

        export_paths = {
            "ply": str(ply_path.resolve()),
            "obj": str(obj_path.resolve()),
            "glb": str(glb_path.resolve()) if glb_path.exists() else "",
        }

        return MeshData(
            mesh_o3d=mesh,
            num_vertices=len(mesh.vertices),
            num_triangles=len(mesh.triangles),
            has_vertex_colors=mesh.has_vertex_colors(),
            has_normals=mesh.has_vertex_normals(),
            export_paths=export_paths,
        )

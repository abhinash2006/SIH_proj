import os
import logging
from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)

class MeshGenerator:
    """
    Generates 3D polygonal surface meshes from point clouds using Open3D.
    Supports Poisson Surface Reconstruction and Ball Pivoting.
    """

    @staticmethod
    def generate_mesh_poisson(
        points: np.ndarray,
        colors: np.ndarray,
        output_path: Union[str, Path],
        depth: int = 8,
        density_trim_quantile: float = 0.08
    ) -> Optional[str]:
        """
        Generates surface mesh using Screened Poisson Surface Reconstruction.
        Trims low-density surface artifacts.
        """
        if len(points) < 500:
            logger.warning(f"[WARNING] Point cloud has only {len(points)} points. Insufficient for mesh generation.")
            return None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[INFO] Generating Poisson mesh from {len(points):,} points (depth={depth})...")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        
        # Estimate and orient normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        # Run Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, linear_fit=True
        )
        
        # Density trimming to remove spurious outer hulls
        if density_trim_quantile > 0 and len(densities) > 0:
            densities_np = np.asarray(densities)
            density_threshold = np.quantile(densities_np, density_trim_quantile)
            vertices_to_remove = densities_np < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        o3d.io.write_triangle_mesh(str(output_path), mesh)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"[SUCCESS] Saved 3D Mesh: {output_path} ({size_mb:.2f} MB, {len(mesh.triangles):,} triangles)")
        return str(output_path)

    @staticmethod
    def generate_poisson_mesh(
        points: np.ndarray,
        colors: np.ndarray,
        output_path: Union[str, Path],
        depth: int = 8,
        density_trim_quantile: float = 0.08
    ) -> Optional[str]:
        return MeshGenerator.generate_mesh_poisson(
            points=points, colors=colors, output_path=output_path, depth=depth, density_trim_quantile=density_trim_quantile
        )

    @staticmethod
    def generate_mesh_ball_pivoting(
        points: np.ndarray,
        colors: np.ndarray,
        output_path: Union[str, Path],
        radii: Optional[list] = None
    ) -> Optional[str]:
        """
        Generates surface mesh using Ball Pivoting Algorithm.
        """
        if len(points) < 500:
            logger.warning(f"[WARNING] Point cloud has only {len(points)} points. Insufficient for mesh generation.")
            return None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        if radii is None:
            # Estimate mean nearest neighbor distance
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist, avg_dist * 2.0, avg_dist * 4.0]
            
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        
        o3d.io.write_triangle_mesh(str(output_path), mesh)
        logger.info(f"[SUCCESS] Saved Ball-Pivoting Mesh: {output_path}")
        return str(output_path)

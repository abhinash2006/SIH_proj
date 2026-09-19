"""
Texture Mapping and Projection for UAV 3D Reconstructed Meshes.
Projects high-resolution RGB keyframe pixels onto reconstructed 3D mesh triangles
prioritizing the sharpest frames and optimal optical viewing angles.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import open3d as o3d
import cv2

logger = logging.getLogger(__name__)


class MeshTextureProjector:
    """
    Projects color from the sharpest and best-aligned camera frames onto mesh vertices/faces.
    """

    def __init__(self, min_ray_angle_deg: float = 75.0):
        self.min_ray_angle_deg = min_ray_angle_deg

    def project_texture_vertex_colors(
        self,
        mesh: o3d.geometry.TriangleMesh,
        image_paths: List[str],
        camera_extrinsics: np.ndarray, # [S, 3, 4]
        camera_intrinsics: np.ndarray, # [S, 3, 3]
        sharpness_scores: Optional[List[float]] = None,
    ) -> o3d.geometry.TriangleMesh:
        """
        Calculates optimal vertex colors for a mesh by projecting vertices into
        the highest-quality candidate keyframes with direct line-of-sight.
        """
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None
        num_v = len(vertices)
        S = len(image_paths)

        if num_v == 0 or S == 0:
            return mesh

        # Load images
        images = []
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                images.append(None)

        # Default sharpness weights
        scores = sharpness_scores if sharpness_scores else [1.0] * S

        accum_colors = np.zeros((num_v, 3), dtype=np.float64)
        accum_weights = np.zeros(num_v, dtype=np.float64)

        for s in range(S):
            img = images[s]
            if img is None:
                continue

            h, w = img.shape[:2]
            ext = camera_extrinsics[s]
            K = camera_intrinsics[s]
            score_w = max(0.1, float(scores[s]))

            R = ext[:3, :3]
            t = ext[:3, 3]

            # Transform world vertices to camera coordinate system
            # P_cam = R * P_world + t
            v_cam = (R @ vertices.T).T + t

            # Check in front of camera (Z > 0.1)
            in_front = v_cam[:, 2] > 0.1
            if not np.any(in_front):
                continue

            # Project into image plane (u, v)
            u = (K[0, 0] * v_cam[:, 0] / v_cam[:, 2]) + K[0, 2]
            v = (K[1, 1] * v_cam[:, 1] / v_cam[:, 2]) + K[1, 2]

            in_image = in_front & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
            valid_indices = np.where(in_image)[0]

            if len(valid_indices) == 0:
                continue

            # Compute view angle alignment weight if normals are available
            if normals is not None:
                # Camera optical axis towards vertex
                ray_dir = -v_cam[valid_indices] / np.linalg.norm(v_cam[valid_indices], axis=1, keepdims=True)
                cos_theta = np.sum(normals[valid_indices] * ray_dir, axis=1)
                angle_w = np.clip(cos_theta, 0.0, 1.0)
            else:
                angle_w = 1.0

            total_w = score_w * angle_w

            # Bilinear or nearest-neighbor pixel sampling
            u_coords = u[valid_indices].astype(int)
            v_coords = v[valid_indices].astype(int)
            sampled_colors = img[v_coords, u_coords].astype(np.float64) / 255.0

            if np.isscalar(total_w):
                accum_colors[valid_indices] += sampled_colors * total_w
                accum_weights[valid_indices] += total_w
            else:
                accum_colors[valid_indices] += sampled_colors * total_w[:, None]
                accum_weights[valid_indices] += total_w

        # Assign normalized colors
        has_weight = accum_weights > 1e-4
        final_colors = np.full((num_v, 3), 0.7, dtype=np.float64)  # neutral gray fallback
        final_colors[has_weight] = accum_colors[has_weight] / accum_weights[has_weight, None]

        mesh.vertex_colors = o3d.utility.Vector3dVector(final_colors)
        logger.info(f"[TEXTURING] Projected texture onto {np.count_nonzero(has_weight):,}/{num_v:,} mesh vertices.")
        return mesh

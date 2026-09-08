import os
import cv2
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class Visualizer:
    """
    Visualization engine for Drone-VGGT 3D reconstruction pipeline.
    Generates depth maps, confidence maps, camera trajectories,
    and side-by-side comparative graphics.
    """

    @staticmethod
    def colorize_depth(depth_map: np.ndarray, cmap: str = "turbo") -> np.ndarray:
        """
        Colorizes a depth map (2D float array) using a perceptual colormap.
        """
        if len(depth_map.shape) == 3 and depth_map.shape[-1] == 1:
            depth_map = depth_map.squeeze(-1)
            
        valid = np.isfinite(depth_map) & (depth_map > 0)
        if not np.any(valid):
            return np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
            
        d_min = np.percentile(depth_map[valid], 2)
        d_max = np.percentile(depth_map[valid], 98)
        
        normalized = np.clip((depth_map - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)
        
        colormap = matplotlib.colormaps[cmap]
        colored = (colormap(normalized)[:, :, :3] * 255.0).astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

    @staticmethod
    def colorize_confidence(conf_map: np.ndarray, cmap: str = "plasma") -> np.ndarray:
        """
        Colorizes a confidence map (2D float array).
        """
        if len(conf_map.shape) == 3 and conf_map.shape[-1] == 1:
            conf_map = conf_map.squeeze(-1)
            
        valid = np.isfinite(conf_map)
        if not np.any(valid):
            return np.zeros((conf_map.shape[0], conf_map.shape[1], 3), dtype=np.uint8)
            
        c_min = np.percentile(conf_map[valid], 2)
        c_max = np.percentile(conf_map[valid], 98)
        
        normalized = np.clip((conf_map - c_min) / (c_max - c_min + 1e-6), 0.0, 1.0)
        colormap = matplotlib.colormaps[cmap]
        colored = (colormap(normalized)[:, :, :3] * 255.0).astype(np.uint8)
        return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

    @classmethod
    def save_depth_gallery(
        cls,
        depth_maps: np.ndarray,
        output_dir: Union[str, Path],
        prefix: str = "depth"
    ) -> List[str]:
        """
        Saves individual colorized depth maps to output directory.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for idx in range(depth_maps.shape[0]):
            d_colored = cls.colorize_depth(depth_maps[idx])
            out_file = out_dir / f"{prefix}_{idx:04d}.png"
            cv2.imwrite(str(out_file), d_colored)
            saved_paths.append(str(out_file))
            
        return saved_paths

    @classmethod
    def plot_camera_trajectory(
        cls,
        extrinsics: np.ndarray, # [S, 3, 4]
        output_path: Union[str, Path],
        title: str = "Drone Camera Trajectory"
    ) -> str:
        """
        Plots the 3D camera trajectory and camera viewing directions.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        num_cams = extrinsics.shape[0]
        # Camera centers in world coordinates: C = -R^T * t
        cam_centers = []
        look_dirs = []
        for i in range(num_cams):
            R = extrinsics[i, :3, :3]
            t = extrinsics[i, :3, 3]
            C = -R.T @ t
            look_dir = R.T @ np.array([0, 0, 1]) # optical axis forward
            cam_centers.append(C)
            look_dirs.append(look_dir)
            
        cam_centers = np.array(cam_centers)
        look_dirs = np.array(look_dirs)

        fig = plt.figure(figsize=(12, 5))
        
        # 3D Trajectory Plot
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot(cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2], 'b-o', markersize=4, label='Drone Path')
        ax1.scatter(cam_centers[0, 0], cam_centers[0, 1], cam_centers[0, 2], color='green', s=60, label='Start')
        ax1.scatter(cam_centers[-1, 0], cam_centers[-1, 1], cam_centers[-1, 2], color='red', s=60, label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f"{title} (3D)")
        ax1.legend()

        # 2D Top-down Flight Path Plot
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(cam_centers[:, 0], cam_centers[:, 1], 'b-o', markersize=4, label='Drone Path')
        ax2.scatter(cam_centers[0, 0], cam_centers[0, 1], color='green', s=60, label='Start')
        ax2.scatter(cam_centers[-1, 0], cam_centers[-1, 1], color='red', s=60, label='End')
        # Draw small heading vectors
        for i in range(0, num_cams, max(1, num_cams // 10)):
            ax2.arrow(cam_centers[i, 0], cam_centers[i, 1], look_dirs[i, 0]*0.2, look_dirs[i, 1]*0.2, 
                      head_width=0.08, color='orange')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f"{title} (Top-Down Bird's Eye)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=200)
        plt.close(fig)
        
        logger.info(f"[SUCCESS] Saved camera trajectory plot: {output_path}")
        return str(output_path)

    @classmethod
    def create_comparison_grid(
        cls,
        baseline_img: np.ndarray,
        refined_img: np.ndarray,
        baseline_depth: np.ndarray,
        refined_depth: np.ndarray,
        output_path: Union[str, Path]
    ) -> str:
        """
        Creates a side-by-side 2x2 comparison image:
        [Raw Frame       ] [Refined Frame  ]
        [Raw Depth Map   ] [Refined Depth  ]
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        h, w = 400, 400
        raw_rgb = cv2.resize(baseline_img, (w, h))
        ref_rgb = cv2.resize(refined_img, (w, h))
        raw_d = cv2.resize(cls.colorize_depth(baseline_depth), (w, h))
        ref_d = cv2.resize(cls.colorize_depth(refined_depth), (w, h))

        # Add text labels
        cv2.putText(raw_rgb, "Baseline Frame", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(ref_rgb, "Refined Frame", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(raw_d, "Baseline Depth Map", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(ref_d, "Refined Depth Map", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        top_row = np.hstack([raw_rgb, ref_rgb])
        bot_row = np.hstack([raw_d, ref_d])
        grid = np.vstack([top_row, bot_row])

        cv2.imwrite(str(output_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"[SUCCESS] Saved side-by-side comparison figure: {output_path}")
        return str(output_path)

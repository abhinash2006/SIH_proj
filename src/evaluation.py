import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class PipelineEvaluator:
    """
    Evaluates and quantitatively compares Baseline vs Refined 3D reconstructions.
    """

    @staticmethod
    def compute_trajectory_smoothness(extrinsics: np.ndarray) -> Dict[str, float]:
        """
        Measures smoothness of estimated camera trajectory (mean velocity variation and acceleration).
        Smoother, physically consistent trajectories have lower jerk and lower discontinuous acceleration.
        """
        num_cams = extrinsics.shape[0]
        if num_cams < 3:
            return {"mean_acceleration": 0.0, "mean_velocity": 0.0}
            
        cam_centers = []
        for i in range(num_cams):
            R = extrinsics[i, :3, :3]
            t = extrinsics[i, :3, 3]
            C = -R.T @ t
            cam_centers.append(C)
            
        cam_centers = np.array(cam_centers)
        velocities = np.diff(cam_centers, axis=0) # [N-1, 3]
        speed = np.linalg.norm(velocities, axis=1)
        
        accelerations = np.diff(velocities, axis=0) # [N-2, 3]
        acc_mag = np.linalg.norm(accelerations, axis=1)
        
        return {
            "mean_velocity": round(float(np.mean(speed)), 4),
            "mean_acceleration": round(float(np.mean(acc_mag)), 4),
            "path_length": round(float(np.sum(speed)), 4)
        }

    @staticmethod
    def compute_pointcloud_metrics(points: np.ndarray, confidences: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Computes geometric density, bounding extent, and confidence stats for a point cloud.
        """
        if len(points) == 0:
            return {"point_count": 0}
            
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        extent = max_bound - min_bound
        volume = float(np.prod(np.maximum(extent, 1e-3)))
        density = float(len(points) / volume) if volume > 0 else 0.0
        
        metrics = {
            "point_count": len(points),
            "bounding_box_extent": [round(float(x), 3) for x in extent],
            "spatial_volume_m3": round(volume, 3),
            "point_density_pts_m3": round(density, 2),
        }
        
        if confidences is not None and len(confidences) > 0:
            metrics["mean_confidence"] = round(float(np.mean(confidences)), 4)
            metrics["median_confidence"] = round(float(np.median(confidences)), 4)
            metrics["std_confidence"] = round(float(np.std(confidences)), 4)
            
        return metrics

    @classmethod
    def evaluate_comparison(
        cls,
        baseline_data: Dict[str, Any],
        refined_data: Dict[str, Any],
        output_json_path: Optional[str] = "outputs/metrics.json"
    ) -> Dict[str, Any]:
        """
        Compares Baseline and Refined pipeline outputs and exports metrics JSON.
        """
        base_pcd_stats = cls.compute_pointcloud_metrics(
            baseline_data.get("points", np.empty((0, 3))),
            baseline_data.get("confidences")
        )
        ref_pcd_stats = cls.compute_pointcloud_metrics(
            refined_data.get("points", np.empty((0, 3))),
            refined_data.get("confidences")
        )
        
        base_traj = cls.compute_trajectory_smoothness(baseline_data.get("extrinsics", np.empty((0, 3, 4))))
        ref_traj = cls.compute_trajectory_smoothness(refined_data.get("extrinsics", np.empty((0, 3, 4))))

        comparison = {
            "baseline": {
                "num_frames": baseline_data.get("num_frames", 0),
                "vggt_inference_time_seconds": round(baseline_data.get("inference_time", 0.0), 3),
                "total_processing_time_seconds": round(baseline_data.get("total_time", 0.0), 3),
                "peak_gpu_vram_mb": round(baseline_data.get("peak_vram_mb", 0.0), 2),
                "pointcloud": base_pcd_stats,
                "trajectory": base_traj,
            },
            "refined": {
                "num_frames": refined_data.get("num_frames", 0),
                "vggt_inference_time_seconds": round(refined_data.get("inference_time", 0.0), 3),
                "total_processing_time_seconds": round(refined_data.get("total_time", 0.0), 3),
                "peak_gpu_vram_mb": round(refined_data.get("peak_vram_mb", 0.0), 2),
                "pointcloud": ref_pcd_stats,
                "trajectory": ref_traj,
                "filtered_outliers": refined_data.get("filter_stats", {}),
            },
            "comparison_summary": {
                "point_count_diff": ref_pcd_stats.get("point_count", 0) - base_pcd_stats.get("point_count", 0),
                "mean_confidence_improvement": round(
                    ref_pcd_stats.get("mean_confidence", 0.0) - base_pcd_stats.get("mean_confidence", 0.0), 4
                ),
                "acceleration_smoothness_improvement": round(
                    base_traj.get("mean_acceleration", 0.0) - ref_traj.get("mean_acceleration", 0.0), 4
                ),
                "has_ground_truth": False,
                "ground_truth_note": "Evaluated on UAVLight real drone flight sequence. Direct relative consistency and point quality metrics measured."
            }
        }

        if output_json_path:
            Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, "w") as f:
                json.dump(comparison, f, indent=4)
            logger.info(f"[SUCCESS] Exported quantitative evaluation report to: {output_json_path}")

        return comparison

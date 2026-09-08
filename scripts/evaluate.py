import os
import sys
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

# Ensure local source directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import PipelineEvaluator

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare Baseline vs Refined 3D reconstructions")
    parser.add_argument("--baseline_ply", type=str, default="outputs/baseline/pointcloud_raw.ply", help="Path to baseline PLY")
    parser.add_argument("--refined_ply", type=str, default="outputs/refined/pointcloud_filtered.ply", help="Path to refined PLY")
    parser.add_argument("--output_json", type=str, default="outputs/metrics.json", help="Output path for metrics JSON")
    
    args = parser.parse_args()
    
    base_path = Path(args.baseline_ply)
    ref_path = Path(args.refined_ply)
    
    if not base_path.exists() or not ref_path.exists():
        print(f"[ERROR] One or both PLY files not found:\n  Baseline: {base_path} (exists: {base_path.exists()})\n  Refined: {ref_path} (exists: {ref_path.exists()})")
        sys.exit(1)
        
    pcd_base = o3d.io.read_point_cloud(str(base_path))
    pcd_ref = o3d.io.read_point_cloud(str(ref_path))
    
    pts_base = np.asarray(pcd_base.points)
    pts_ref = np.asarray(pcd_ref.points)
    
    base_data = {"points": pts_base, "num_frames": 0}
    ref_data = {"points": pts_ref, "num_frames": 0}
    
    results = PipelineEvaluator.evaluate_comparison(
        baseline_data=base_data,
        refined_data=ref_data,
        output_json_path=args.output_json
    )
    
    print("\n" + "="*50)
    print("  QUANTITATIVE EVALUATION RESULTS")
    print("="*50)
    print(f"Baseline Points: {len(pts_base):,}")
    print(f"Refined Points:  {len(pts_ref):,}")
    print(f"Volume Baseline: {results['baseline']['pointcloud']['spatial_volume_m3']} m³")
    print(f"Volume Refined:  {results['refined']['pointcloud']['spatial_volume_m3']} m³")
    print(f"Density Baseline: {results['baseline']['pointcloud']['point_density_pts_m3']} pts/m³")
    print(f"Density Refined:  {results['refined']['pointcloud']['point_density_pts_m3']} pts/m³")
    print(f"Report saved to: {args.output_json}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

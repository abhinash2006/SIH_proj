import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vggt_inference import VGGTInferenceEngine
from src.reconstruction import PointCloudReconstructor
from src.pointcloud_filter import PointCloudFilter
from src.mesh_utils import MeshGenerator
from src.disaster_inspection.segmentation import SceneSegmentationAnalyzer
from src.visualization import Visualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DiagnosticV2")

def run_diagnostic():
    output_dir = Path("outputs/debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load candidate frames
    sequence_dir = Path("data/raw/uav_sequence/1121222322212102-4/images")
    image_paths = sorted(list(sequence_dir.rglob("*.JPG")) + list(sequence_dir.rglob("*.jpg")))
    
    if not image_paths:
        logger.error(f"[ERROR] No images found in {sequence_dir}")
        return

    # Select 12 evenly spaced frames for optimal baseline
    num_selected = min(12, len(image_paths))
    step = len(image_paths) // num_selected
    selected_paths = [image_paths[i * step] for i in range(num_selected)]
    logger.info(f"[DIAGNOSTIC] Selected {len(selected_paths)} sequence frames.")

    # 2. Run Single-Pass Unified VGGT Inference
    engine = VGGTInferenceEngine()
    predictions = engine.run_inference(selected_paths)

    # -------------------------------------------------------------
    # COORDINATE & CAMERA AUDIT
    # -------------------------------------------------------------
    first_img = cv2.imread(str(selected_paths[0]))
    orig_h, orig_w = first_img.shape[:2]
    
    rgb_maps = predictions["images"]          # [S, H, W, 3]
    point_maps = predictions["point_maps"]    # [S, H, W, 3]
    depth_maps = predictions["depth_maps"]    # [S, H, W, 1]
    unproj_maps = predictions["unprojected_points"] # [S, H, W, 3]
    intrinsics = predictions["intrinsics"]    # [S, 3, 3]
    extrinsics = predictions["extrinsics"]    # [S, 3, 4]
    
    S, model_h, model_w, _ = point_maps.shape

    logger.info("=" * 60)
    logger.info("CRITICAL COORDINATE CHECK")
    logger.info("=" * 60)
    logger.info(f"Original image size: {orig_w} x {orig_h}")
    logger.info(f"VGGT input size:    {model_w} x {model_h}")
    logger.info(f"RGB tensor size:    {rgb_maps.shape}")
    logger.info(f"Point map size:     {point_maps.shape}")
    logger.info(f"Depth map size:     {depth_maps.shape}")
    logger.info(f"K_scaled frame 0:\n{intrinsics[0]}")

    # Scale K back to original image coordinates for reference
    scale_x = orig_w / model_w
    scale_y = orig_h / model_h
    K_orig = intrinsics[0].copy()
    K_orig[0, 0] *= scale_x
    K_orig[1, 1] *= scale_y
    K_orig[0, 2] *= scale_x
    K_orig[1, 2] *= scale_y
    logger.info(f"K_original frame 0:\n{K_orig}")

    # -------------------------------------------------------------
    # 4 MANDATORY PLY OUTPUTS
    # -------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("GENERATING 4 MANDATORY PLY OUTPUTS")
    logger.info("=" * 60)

    # OUTPUT 1: RAW VGGT POINT MAP (Direct predicted point map, NO filtering)
    pts_01 = point_maps.reshape(-1, 3)
    cols_01 = (rgb_maps.reshape(-1, 3) / 255.0).astype(np.float32)
    valid_01 = np.isfinite(pts_01).all(axis=1) & (np.linalg.norm(pts_01, axis=1) > 1e-4)
    ply_01_path = output_dir / "01_raw_vggt_pointmap.ply"
    PointCloudReconstructor.save_ply(pts_01[valid_01], cols_01[valid_01], ply_01_path)

    # OUTPUT 2: VGGT DEPTH UNPROJECTION (Depth + K + Extrinsics, NO filtering)
    pts_02 = unproj_maps.reshape(-1, 3)
    cols_02 = cols_01.copy()
    valid_02 = np.isfinite(pts_02).all(axis=1) & (np.linalg.norm(pts_02, axis=1) > 1e-4)
    ply_02_path = output_dir / "02_depth_unprojection_raw.ply"
    PointCloudReconstructor.save_ply(pts_02[valid_02], cols_02[valid_02], ply_02_path)

    # OUTPUT 3: FILTERED POINT CLOUD (Confidence + SOR + Voxel + DBSCAN)
    pts_03_raw, cols_03_raw, _ = PointCloudReconstructor.build_pointcloud_from_predictions(
        predictions, use_depth_unprojection=True, confidence_threshold=1.1, max_points=1_000_000
    )
    filter_obj = PointCloudFilter()
    pts_03, cols_03, meta_03 = filter_obj.filter_pointcloud(pts_03_raw, cols_03_raw)
    ply_03_path = output_dir / "03_filtered_pointcloud.ply"
    PointCloudReconstructor.save_ply(pts_03, cols_03, ply_03_path)

    # OUTPUT 4: FINAL MESH (Poisson mesh strictly from Output 3)
    ply_04_path = output_dir / "04_final_mesh.ply"
    MeshGenerator.generate_mesh_poisson(pts_03, cols_03, ply_04_path, depth=8)
    logger.info(f"[SUCCESS] Saved Output 04 Final Mesh to {ply_04_path}")

    # -------------------------------------------------------------
    # CAMERA TRAJECTORY & FRUSTUMS VISUALIZATION
    # -------------------------------------------------------------
    traj_path = output_dir / "camera_trajectory.png"
    Visualizer.plot_camera_trajectory(extrinsics, traj_path)

    # -------------------------------------------------------------
    # FEATURE TRACKING & REPROJECTION CHECK
    # -------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SCENE COVERAGE & FEATURE REPROJECTION CHECK")
    logger.info("=" * 60)

    # Pick 5 landmark pixel locations in frame 0 (scaled to 392x392)
    test_pixels = [
        ("Roof House A", int(model_w * 0.35), int(model_h * 0.40)),
        ("Roof House B", int(model_w * 0.65), int(model_h * 0.35)),
        ("Road/Embankment", int(model_w * 0.50), int(model_h * 0.70)),
        ("Tree Region", int(model_w * 0.20), int(model_h * 0.60)),
        ("Ground Region", int(model_w * 0.80), int(model_h * 0.75)),
    ]

    K0 = intrinsics[0]
    R0 = extrinsics[0, :3, :3]
    t0 = extrinsics[0, :3, 3]

    reproj_errors = []
    for label, u, v in test_pixels:
        p3d = unproj_maps[0, v, u] # image[v, u] -> point_map[v, u]
        # Project 3D back to camera 0
        p_cam = R0 @ p3d + t0
        u_proj = (K0[0, 0] * p_cam[0] / p_cam[2]) + K0[0, 2]
        v_proj = (K0[1, 1] * p_cam[1] / p_cam[2]) + K0[1, 2]
        err = np.sqrt((u - u_proj)**2 + (v - v_proj)**2)
        reproj_errors.append(err)
        logger.info(f"Feature: {label:15s} | 2D: ({u:3d}, {v:3d}) | 3D: ({p3d[0]:.2f}, {p3d[1]:.2f}, {p3d[2]:.2f}) | Reproj: ({u_proj:.1f}, {v_proj:.1f}) | Err: {err:.2f} px")

    mean_reproj_err = float(np.mean(reproj_errors))
    logger.info(f"[MEAN REPROJECTION ERROR]: {mean_reproj_err:.2f} pixels")

    # -------------------------------------------------------------
    # AXIS & SPATIAL EXTENT CHECK
    # -------------------------------------------------------------
    pts_clean = pts_03
    min_xyz = pts_clean.min(axis=0)
    max_xyz = pts_clean.max(axis=0)
    extents = max_xyz - min_xyz

    logger.info("=" * 60)
    logger.info("AXIS & SPATIAL BOUNDS CHECK")
    logger.info("=" * 60)
    logger.info(f"X range: [{min_xyz[0]:.2f}, {max_xyz[0]:.2f}] (Extent: {extents[0]:.2f})")
    logger.info(f"Y range: [{min_xyz[1]:.2f}, {max_xyz[1]:.2f}] (Extent: {extents[1]:.2f})")
    logger.info(f"Z range: [{min_xyz[2]:.2f}, {max_xyz[2]:.2f}] (Extent: {extents[2]:.2f})")

    # -------------------------------------------------------------
    # WATER ARTIFACT SEPARATION
    # -------------------------------------------------------------
    water_mask, water_ratio = SceneSegmentationAnalyzer.segment_water_and_flood(rgb_maps[0])
    logger.info(f"[WATER SEPARATION] Frame 0 Flood Water Ratio: {water_ratio * 100:.1f}% of scene area")

    # Save flood debug images
    cv2.imwrite(str(output_dir / "flood_mask_frame_0000.png"), water_mask)
    overlay = rgb_maps[0].copy()
    overlay[water_mask > 0] = (overlay[water_mask > 0] * 0.5 + np.array([255, 120, 0]) * 0.5).astype(np.uint8)
    cv2.imwrite(str(output_dir / "flood_overlay_frame_0000.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # -------------------------------------------------------------
    # DUAL GEOMETRY QUALITY SCORING
    # -------------------------------------------------------------
    # 1. Geometric Validity Score (Mathematical consistency: camera baseline, finite points, depth validity)
    cam_centers = []
    for i in range(S):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        cam_centers.append(-R.T @ t)
    cam_centers = np.array(cam_centers)
    baseline = float(np.linalg.norm(cam_centers[0] - cam_centers[-1]))

    geom_validity_score = min(100.0, (baseline / 2.0) * 50.0 + min(50.0, len(pts_clean) / 500.0))

    # 2. Visual / Structural Reconstruction Score (Feature preservation & planarity)
    # Check if point cloud has clean structural density and reasonable height variance (Z ratio)
    height_variance = float(np.var(pts_clean[:, 2]))
    repr_score = max(0.0, 100.0 - mean_reproj_err * 2.0)
    visual_structural_score = min(100.0, repr_score * 0.6 + min(40.0, height_variance * 500.0))

    overall_status = "GOOD" if (geom_validity_score > 70 and visual_structural_score > 60) else "POOR" if (geom_validity_score > 40) else "INVALID"

    logger.info("=" * 60)
    logger.info("NEW DUAL GEOMETRY QUALITY SCORE")
    logger.info("=" * 60)
    logger.info(f"Geometric Validity Score:           {geom_validity_score:.1f} / 100")
    logger.info(f"Visual / Structural Score:           {visual_structural_score:.1f} / 100")
    logger.info(f"Overall Reconstruction Status:      {overall_status}")

    # Write summary diagnostic report
    summary = {
        "original_image_size": [orig_w, orig_h],
        "vggt_input_size": [model_w, model_h],
        "num_frames": S,
        "raw_pointmap_points": int(valid_01.sum()),
        "depth_unprojected_points": int(valid_02.sum()),
        "filtered_points": len(pts_clean),
        "mean_reprojection_error_px": mean_reproj_err,
        "spatial_extents": [float(e) for e in extents],
        "camera_baseline": baseline,
        "water_coverage_ratio": water_ratio,
        "geometric_validity_score": geom_validity_score,
        "visual_structural_score": visual_structural_score,
        "overall_status": overall_status,
        "ply_outputs": {
            "01_raw_vggt_pointmap": str(ply_01_path),
            "02_depth_unprojection_raw": str(ply_02_path),
            "03_filtered_pointcloud": str(ply_03_path),
            "04_final_mesh": str(ply_04_path)
        }
    }

    with open(output_dir / "diagnostic_v2_report.json", "w") as f:
        json.dump(summary, f, indent=4)

    logger.info(f"[SUCCESS] Diagnostic complete! Report saved to {output_dir / 'diagnostic_v2_report.json'}")

if __name__ == "__main__":
    run_diagnostic()

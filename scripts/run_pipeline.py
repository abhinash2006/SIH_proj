import os
import sys

# Prevent OpenBLAS memory allocation errors on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import json
import yaml
import logging
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Ensure local source directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.video_processor import VideoProcessor
from src.frame_selector import FrameSelector
from src.image_refiner import ImageRefiner
from src.vggt_inference import VGGTInferenceEngine
from src.reconstruction import PointCloudReconstructor
from src.pointcloud_filter import PointCloudFilter
from src.colmap_utils import COLMAPExporter
from src.mesh_utils import MeshGenerator
from src.visualization import Visualizer

from src.depth_anything_v2 import DepthAnythingV2Estimator
from src.depth_alignment import RobustDepthAligner
from src.depth_consistency import DepthConsistencyAnalyzer
from src.depth_confidence import DepthConfidenceFused

from src.disaster_inspection.object_detection import DisasterObjectDetector
from src.disaster_inspection.tracking import MultiFrameObjectTracker
from src.disaster_inspection.geo_projection import GeoProjection3D
from src.disaster_inspection.damage_detection import StructuralDamageDetector
from src.disaster_inspection.hazard_detection import HazardDetector
from src.disaster_inspection.incident_manager import DisasterIncidentManager
from src.disaster_inspection.risk_scoring import DisasterRiskScorer
from src.disaster_inspection.report_generator import DisasterReportGenerator

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("DroneVGGT_DisasterPipeline")

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def run_pipeline(
    input_path: str,
    output_dir: str = "outputs",
    mode: str = "all", # "all", "A", "B", "C", "D", "both"
    max_frames: int = 16,
    use_depth_anything: bool = True,
    depth_model_size: str = "small",
    depth_aware_filtering: bool = True,
    config_path: str = "configs/config.yaml"
):
    config = load_config(config_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Frame Extraction & Geometric Selection
    frames_dir = output_dir_path / "temp_frames"
    if os.path.isfile(input_path) and VideoProcessor.is_video_file(input_path):
        extracted_dir = frames_dir / "extracted"
        frame_paths, _ = VideoProcessor.extract_frames(input_path, extracted_dir, target_fps=2.0)
    elif os.path.isdir(input_path):
        frame_paths = VideoProcessor.process_image_folder(input_path)
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    selector = FrameSelector(config)
    selected_dir = frames_dir / "selected"
    selected_paths, quality_metrics = selector.select_optimal_frames(frame_paths, selected_dir, max_frames=max_frames)
    logger.info(f"[SUCCESS] Selected {len(selected_paths)} optimal frames.")

    # -------------------------------------------------------------
    # STEP 2: DEPTH ANYTHING V2 MONOCULAR ESTIMATION
    # -------------------------------------------------------------
    da2_depth_maps = []
    da2_visualizations = []
    if use_depth_anything:
        logger.info("\n>>> RUNNING DEPTH ANYTHING V2 MONOCULAR ESTIMATION <<<")
        da2_estimator = DepthAnythingV2Estimator(model_name=depth_model_size, encoder="vits" if depth_model_size=="small" else "vitb")
        da2_dir = output_dir_path / "depth_anything_v2"
        vis_dir = da2_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        for i, f_path in enumerate(selected_paths):
            d_map = da2_estimator.predict_image(f_path)
            da2_depth_maps.append(d_map)
            np.save(str(da2_dir / f"frame_{i:04d}.npy"), d_map)
            v_path = da2_estimator.save_depth_visualizations(d_map, vis_dir / f"frame_{i:04d}.png")
            da2_visualizations.append(v_path)

        da2_estimator.unload_model()
        logger.info(f"[SUCCESS] Depth Anything V2 monocular depth maps computed for {len(da2_depth_maps)} frames.")

    # -------------------------------------------------------------
    # STEP 3: VGGT MULTI-VIEW GEOMETRY INFERENCE
    # -------------------------------------------------------------
    logger.info("\n>>> RUNNING VGGT 3D FOUNDATION INFERENCE <<<")
    engine = VGGTInferenceEngine()
    predictions = engine.run_inference(selected_paths)
    engine.unload_model()

    # -------------------------------------------------------------
    # STEP 4: DEPTH ALIGNMENT & CONSISTENCY ANALYSIS (EXPERIMENT B)
    # -------------------------------------------------------------
    aligned_da2_maps = []
    consistency_maps = []
    fused_confidence_maps = []

    if use_depth_anything and len(da2_depth_maps) > 0:
        logger.info("\n>>> RUNNING DEPTH ALIGNMENT & CONSISTENCY FUSION <<<")
        aligner = RobustDepthAligner()
        analyzer = DepthConsistencyAnalyzer()
        fuser = DepthConfidenceFused()

        expB_dir = output_dir_path / "experiment_B"
        expB_dir.mkdir(parents=True, exist_ok=True)

        for i in range(len(selected_paths)):
            vggt_d = predictions["depth_maps"][i]
            vggt_c = predictions["depth_conf"][i]
            raw_da2_d = da2_depth_maps[i]

            aligned_da2, align_metrics = aligner.align_depth_maps(raw_da2_d, vggt_d, vggt_c)
            abs_diff, consistency, cons_stats = analyzer.compute_consistency(vggt_d, aligned_da2, vggt_c)
            fused_conf = fuser.fuse_confidence(vggt_c, consistency)

            aligned_da2_maps.append(aligned_da2)
            consistency_maps.append(consistency)
            fused_confidence_maps.append(fused_conf)

        logger.info("[SUCCESS] Depth Anything V2 metric scale/shift alignment and consistency fusion complete.")

    # -------------------------------------------------------------
    # EXPERIMENT A: BASELINE VGGT 3D RECONSTRUCTION
    # -------------------------------------------------------------
    if mode in ["all", "A", "both"]:
        logger.info("\n>>> EXECUTING EXPERIMENT A: BASELINE VGGT <<<")
        expA_dir = output_dir_path / "experiment_A"
        expA_dir.mkdir(parents=True, exist_ok=True)
        reconstructor = PointCloudReconstructor()
        raw_pts, raw_cols, raw_conf = reconstructor.build_pointcloud_from_predictions(predictions, confidence_threshold=1.1)
        reconstructor.save_ply(raw_pts, raw_cols, expA_dir / "pointcloud_raw.ply")
        COLMAPExporter.export_colmap(expA_dir, selected_paths, predictions["extrinsics"], predictions["intrinsics"], raw_pts, raw_cols)
        MeshGenerator.generate_poisson_mesh(raw_pts, raw_cols, expA_dir / "surface_mesh.ply")
        logger.info(f"[SUCCESS] Experiment A complete: {len(raw_pts):,} 3D points.")

    # -------------------------------------------------------------
    # EXPERIMENT C: DEPTH-AWARE FILTERED 3D RECONSTRUCTION
    # -------------------------------------------------------------
    if mode in ["all", "C", "both"]:
        logger.info("\n>>> EXECUTING EXPERIMENT C: DEPTH-AWARE RECONSTRUCTION <<<")
        expC_dir = output_dir_path / "experiment_C"
        expC_dir.mkdir(parents=True, exist_ok=True)
        reconstructor = PointCloudReconstructor()
        raw_pts, raw_cols, raw_conf = reconstructor.build_pointcloud_from_predictions(predictions, confidence_threshold=1.1)

        pcd_filter = PointCloudFilter(config)
        fused_conf_flat = np.concatenate([c.flatten() for c in fused_confidence_maps]) if fused_confidence_maps else None
        filt_pts, filt_cols, filter_stats = pcd_filter.filter_pointcloud(raw_pts, raw_cols, fused_conf_flat, conf_threshold=0.55)
        
        reconstructor.save_ply(filt_pts, filt_cols, expC_dir / "pointcloud_filtered.ply")
        COLMAPExporter.export_colmap(expC_dir, selected_paths, predictions["extrinsics"], predictions["intrinsics"], filt_pts, filt_cols)
        MeshGenerator.generate_poisson_mesh(filt_pts, filt_cols, expC_dir / "surface_mesh.ply")
        logger.info(f"[SUCCESS] Experiment C complete: {len(filt_pts):,} high-confidence 3D points retained.")

    # -------------------------------------------------------------
    # EXPERIMENT D: FULL AI DISASTER INSPECTION & 3D RESCUE MAP
    # -------------------------------------------------------------
    if mode in ["all", "D"]:
        logger.info("\n>>> EXECUTING EXPERIMENT D: AI DISASTER INSPECTION & RESCUE MAPPING <<<")
        expD_dir = output_dir_path / "experiment_D"
        expD_dir.mkdir(parents=True, exist_ok=True)

        obj_detector = DisasterObjectDetector()
        tracker = MultiFrameObjectTracker()
        damage_detector = StructuralDamageDetector()
        hazard_detector = HazardDetector()
        inc_manager = DisasterIncidentManager(db_dir=str(expD_dir))

        all_detections_3d = []
        all_hazards = []
        all_damage = []

        for i, f_path in enumerate(selected_paths):
            img_bgr = cv2.imread(f_path)
            depth_m = predictions["depth_maps"][i]
            ext = predictions["extrinsics"][i]
            intri = predictions["intrinsics"][i]

            # 1. 2D Detect & Track
            dets = obj_detector.detect_objects_in_frame(img_bgr, frame_idx=i)
            tracked_dets = tracker.track_frame_detections(dets, frame_idx=i)

            # 2. 3D Geo Projection
            for det in tracked_dets:
                proj3d = GeoProjection3D.project_detection_to_3d(det, depth_m, ext, intri)
                all_detections_3d.append(proj3d)

                if det["class_name"] == "person":
                    inc_manager.add_incident({
                        "incident_type": "VICTIM_PERSON_DETECTED",
                        "object_id": proj3d.get("object_id", "PERSON_001"),
                        "confidence": proj3d["confidence"],
                        "severity": "CRITICAL",
                        "x_m": proj3d["x_m"],
                        "y_m": proj3d["y_m"],
                        "z_m": proj3d["z_m"],
                        "frame_idx": i,
                        "evidence": f"Person detected at 3D location ({proj3d['x_m']}m, {proj3d['y_m']}m, {proj3d['z_m']}m)."
                    })

            # 3. Damage & Hazard inspection
            dmg = damage_detector.analyze_structural_damage(img_bgr, depth_m, tracked_dets, frame_idx=i)
            hz = hazard_detector.analyze_hazards(img_bgr, depth_m, tracked_dets, frame_idx=i)

            all_damage.extend(dmg)
            all_hazards.extend(hz)

            for d in dmg:
                inc_manager.add_incident(d)
            for h in hz:
                inc_manager.add_incident(h)

        # Calculate Overall Risk & Priorities
        risk_level, risk_score, reasons, priorities = DisasterRiskScorer.calculate_scene_risk(
            all_detections_3d, all_hazards, all_damage
        )

        reconstructor = PointCloudReconstructor()
        raw_pts, raw_cols, raw_conf = reconstructor.build_pointcloud_from_predictions(predictions, confidence_threshold=1.1)

        summary_data = {
            "total_frames": len(frame_paths),
            "selected_frames": len(selected_paths),
            "reconstructed_points": len(raw_pts),
            "detected_people": sum(1 for d in all_detections_3d if d.get("class_name") == "person"),
            "detected_vehicles": sum(1 for d in all_detections_3d if d.get("class_name") == "vehicle"),
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_reasons": reasons,
            "rescue_priorities": priorities
        }

        report_path = DisasterReportGenerator.generate_html_report(summary_data, inc_manager.get_all_incidents(), expD_dir / "disaster_report.html")
        logger.info(f"[SUCCESS] Disaster inspection report generated: {report_path}")
        logger.info(f"[SUCCESS] Overall Rescue Risk Level: {risk_level} (Score: {risk_score}/100)")

def main():
    parser = argparse.ArgumentParser(description="Drone-VGGT AI Disaster Inspection & 3D Rescue Mapping Pipeline")
    parser.add_argument("--input", required=True, help="Path to input video file or image sequence directory")
    parser.add_argument("--output_dir", default="outputs", help="Directory where outputs and reports are stored")
    parser.add_argument("--mode", default="all", choices=["all", "A", "B", "C", "D", "both"], help="Experiment mode")
    parser.add_argument("--max_frames", type=int, default=16, help="Maximum frames selected")
    parser.add_argument("--use-depth-anything", action="store_true", default=True, help="Enable Depth Anything V2 model")
    parser.add_argument("--disable-depth-anything", action="store_false", dest="use_depth_anything", help="Disable Depth Anything V2 model")
    parser.add_argument("--depth-model", default="small", choices=["small", "base"], help="Depth Anything V2 model scale")
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        mode=args.mode,
        max_frames=args.max_frames,
        use_depth_anything=args.use_depth_anything,
        depth_model_size=args.depth_model
    )

if __name__ == "__main__":
    main()

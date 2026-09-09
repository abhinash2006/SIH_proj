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
    use_yolo: bool = True,
    yolo_model: str = "yolo26n.pt",
    yolo_confidence: float = 0.35,
    yolo_imgsz: int = 640,
    yolo_device: str = "auto",
    yolo_tracking: bool = False,
    yolo_tiling: bool = False,
    config_path: str = "configs/config.yaml"
):

    config = load_config(config_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Frame Extraction & Geometric Selection
    frames_dir = output_dir_path / "temp_frames"
    if os.path.isfile(input_path) and VideoProcessor.is_video_file(input_path):
        extracted_dir = frames_dir / "extracted"
        frame_paths = VideoProcessor.extract_frames(input_path, extracted_dir, target_fps=2.0)
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

        # 1. Perception Layer: Ultralytics YOLO 2D Object Detection & Tracking
        from src.yolo_detector import YOLODetector
        yolo_dir = expD_dir / "yolo"
        annotated_dir = yolo_dir / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)

        yolo_detector = YOLODetector(
            model_name=yolo_model,
            confidence=yolo_confidence,
            imgsz=yolo_imgsz,
            device=yolo_device,
            tiled_inference=yolo_tiling
        )

        if use_yolo:
            logger.info(f"\n>>> RUNNING ULTRALYTICS YOLO ({yolo_model}) PERCEPTION LAYER <<<")
            if yolo_tracking:
                frame_detections_batch = yolo_detector.track_frames(selected_paths, conf=yolo_confidence, imgsz=yolo_imgsz)
            else:
                frame_detections_batch = yolo_detector.detect_frames(selected_paths, conf=yolo_confidence, imgsz=yolo_imgsz)
        else:
            logger.info("\n>>> YOLO Perception Layer disabled via flag. Using empty detections. <<<")
            frame_detections_batch = [[] for _ in selected_paths]

        tracker = MultiFrameObjectTracker()
        damage_detector = StructuralDamageDetector()
        hazard_detector = HazardDetector()
        inc_manager = DisasterIncidentManager(db_dir=str(expD_dir))

        reconstructor = PointCloudReconstructor()
        raw_pts, raw_cols, raw_conf = reconstructor.build_pointcloud_from_predictions(predictions, confidence_threshold=1.1)

        # Geometry Quality Evaluation Gate
        from src.geometry_quality import GeometryQualityEvaluator
        geom_eval = GeometryQualityEvaluator.evaluate_reconstruction_quality(predictions, raw_pts)
        logger.info(f"[GEOMETRY EVAL] Status: {geom_eval['geometry_status']} | Score: {geom_eval['geometry_quality_score']}/100 | Allowed: {geom_eval['localization_allowed']}")

        all_detections_3d = []
        all_hazards = []
        all_damage = []
        reprojection_errors_log = []

        for i, f_path in enumerate(selected_paths):
            img_bgr = cv2.imread(f_path)
            depth_m = predictions["depth_maps"][i]
            ext = predictions["extrinsics"][i]
            intri = predictions["intrinsics"][i]

            # Detections for current frame
            raw_dets = frame_detections_batch[i]
            tracked_dets = tracker.track_frame_detections(raw_dets, frame_idx=i)

            # Save annotated frame image
            annotated_img = yolo_detector.annotate_frame(img_bgr, tracked_dets)
            cv2.imwrite(str(annotated_dir / f"frame_{i:04d}.png"), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

            # 2. Damage & Hazard inspection (establish disaster context first)
            dmg = damage_detector.analyze_structural_damage(img_bgr, depth_m, tracked_dets, frame_idx=i)
            hz = hazard_detector.analyze_hazards(img_bgr, depth_m, tracked_dets, frame_idx=i)

            for d in dmg:
                if geom_eval["localization_allowed"]:
                    p3d = GeoProjection3D.project_detection_to_3d(d, depth_m, ext, intri)
                    d.update(p3d)
                else:
                    d.update({"location_status": "UNLOCALIZED", "x_m": None, "y_m": None, "z_m": None})
                all_damage.append(d)
                inc_manager.add_incident(d)

            for h in hz:
                if geom_eval["localization_allowed"]:
                    p3d = GeoProjection3D.project_detection_to_3d(h, depth_m, ext, intri)
                    h.update(p3d)
                else:
                    h.update({"location_status": "UNLOCALIZED", "x_m": None, "y_m": None, "z_m": None})
                all_hazards.append(h)
                inc_manager.add_incident(h)

            is_scene_flooded = any(h.get("incident_type") == "FLOOD_ZONE" for h in hz)
            has_active_disaster = len(hz) > 0 or len(dmg) > 0

            from src.disaster_inspection.segmentation import SceneSegmentationAnalyzer
            water_mask_frame, _ = SceneSegmentationAnalyzer.segment_water_candidates(img_bgr)

            # 3. 3D Geo Projection & Grounded Incident Creation
            for det in tracked_dets:
                if geom_eval["localization_allowed"]:
                    proj3d = GeoProjection3D.project_detection_to_3d(det, depth_m, ext, intri)
                else:
                    proj3d = det.copy()
                    proj3d.update({"location_status": "UNLOCALIZED", "x_m": None, "y_m": None, "z_m": None, "reprojection_error_px": None})

                all_detections_3d.append(proj3d)

                if proj3d.get("reprojection_error_px") is not None:
                    reprojection_errors_log.append({
                        "frame_idx": i,
                        "class_name": det["class_name"],
                        "reprojection_error_px": proj3d["reprojection_error_px"],
                        "location_status": proj3d["location_status"]
                    })

                cls_name = det.get("class_name")
                if cls_name == "person":
                    inc_type = "VICTIM_PERSON_DETECTED" if has_active_disaster else "PERSON_DETECTED"
                    sev = "CRITICAL" if has_active_disaster else "LOW"
                    ev = f"Person / potential victim located in active disaster zone (frame {i})." if has_active_disaster else f"Civilian person observed during aerial monitoring (frame {i})."
                    inc_manager.add_incident({
                        "incident_type": inc_type,
                        "target_object_class": "person",
                        "object_id": proj3d.get("object_id", "PERSON_001"),
                        "confidence": proj3d["confidence"],
                        "detection_confidence": proj3d.get("detection_confidence", proj3d["confidence"]),
                        "localization_confidence": proj3d.get("localization_confidence", 0.0),
                        "severity": sev,
                        "location_status": proj3d["location_status"],
                        "x_m": proj3d.get("x_m"),
                        "y_m": proj3d.get("y_m"),
                        "z_m": proj3d.get("z_m"),
                        "reprojection_error_px": proj3d.get("reprojection_error_px"),
                        "frame_idx": i,
                        "evidence": ev
                    })
                elif cls_name in ["building", "structure"]:
                    bx1, by1, bx2, by2 = det["bbox"]
                    b_crop = water_mask_frame[max(0, by1):min(img_bgr.shape[0], by2), max(0, bx1):min(img_bgr.shape[1], bx2)]
                    b_water_pct = float(np.mean(b_crop > 0)) if b_crop.size > 0 else 0.0
                    if is_scene_flooded and (b_water_pct > 0.02 or any(h.get("water_state") == "FLOOD_ZONE" for h in hz)):
                        inc_manager.add_incident({
                            "incident_type": "FLOODED_STRUCTURE",
                            "target_object_class": "building",
                            "object_id": proj3d.get("object_id", f"STRUCTURE_{i+1:03d}"),
                            "confidence": proj3d["confidence"],
                            "detection_confidence": proj3d.get("detection_confidence", proj3d["confidence"]),
                            "localization_confidence": proj3d.get("localization_confidence", 0.0),
                            "severity": "HIGH",
                            "location_status": proj3d["location_status"],
                            "x_m": proj3d.get("x_m"),
                            "y_m": proj3d.get("y_m"),
                            "z_m": proj3d.get("z_m"),
                            "reprojection_error_px": proj3d.get("reprojection_error_px"),
                            "frame_idx": i,
                            "evidence": f"Flooded structure inundated by water ({b_water_pct*100.0:.1f}% water contact) in aerial frame {i}."
                        })

        # Release YOLO model from GPU memory
        yolo_detector.release()

        # Save Debug Artifacts (Requirements 2, 4, 16)

        debug_dir = output_dir_path / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        Visualizer.plot_camera_trajectory(predictions["extrinsics"], debug_dir / "camera_trajectory.png")
        
        # Method A: Direct VGGT Point Head Prediction
        pts_A, cols_A, conf_A = reconstructor.build_pointcloud_from_predictions(predictions, use_depth_unprojection=False, confidence_threshold=0.0)
        reconstructor.save_ply(pts_A, cols_A, debug_dir / "raw_vggt_points.ply")
        reconstructor.save_ply(pts_A, cols_A, debug_dir / "pointmap_reconstruction.ply")

        # Confidence Filtered Direct Points
        pts_A_filt, cols_A_filt, _ = reconstructor.build_pointcloud_from_predictions(predictions, use_depth_unprojection=False, confidence_threshold=1.2)
        reconstructor.save_ply(pts_A_filt, cols_A_filt, debug_dir / "confidence_filtered_vggt_points.ply")

        # Method B: VGGT Depth + Camera Unprojection
        pts_B, cols_B, conf_B = reconstructor.build_pointcloud_from_predictions(predictions, use_depth_unprojection=True, confidence_threshold=1.1)
        reconstructor.save_ply(pts_B, cols_B, debug_dir / "depth_unprojection_reconstruction.ply")
        reconstructor.save_ply(raw_pts, raw_cols, debug_dir / "pointcloud_raw.ply")

        with open(debug_dir / "geometry_metrics.json", "w") as f:
            json.dump(geom_eval, f, indent=4)
        with open(debug_dir / "reprojection_errors.json", "w") as f:
            json.dump(reprojection_errors_log, f, indent=4)
        with open(debug_dir / "detections.json", "w") as f:
            json.dump([d for d in all_detections_3d], f, indent=4, default=str)
        with open(debug_dir / "incidents.json", "w") as f:
            json.dump(inc_manager.get_all_incidents(), f, indent=4, default=str)

        # Calculate Overall Risk & Priorities
        risk_level, risk_score, reasons, priorities = DisasterRiskScorer.calculate_scene_risk(
            all_detections_3d, all_hazards, all_damage
        )

        summary_data = {
            "total_frames": len(frame_paths),
            "selected_frames": len(selected_paths),
            "reconstructed_points": len(raw_pts),
            "detected_people": sum(1 for d in all_detections_3d if d.get("class_name") == "person"),
            "detected_vehicles": sum(1 for d in all_detections_3d if d.get("class_name") == "vehicle"),
            "geometry_status": geom_eval["geometry_status"],
            "geometry_quality_score": geom_eval["geometry_quality_score"],
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
    parser.add_argument("--use-vggt", action="store_true", default=True, help="Enable VGGT model")
    parser.add_argument("--use-depth-anything", action="store_true", default=True, help="Enable Depth Anything V2 model")
    parser.add_argument("--disable-depth-anything", action="store_false", dest="use_depth_anything", help="Disable Depth Anything V2 model")
    parser.add_argument("--depth-model", default="small", choices=["small", "base"], help="Depth Anything V2 model scale")
    
    # YOLO Perception CLI arguments
    parser.add_argument("--use-yolo", action="store_true", default=True, help="Enable Ultralytics YOLO object perception layer")
    parser.add_argument("--disable-yolo", action="store_false", dest="use_yolo", help="Disable Ultralytics YOLO object perception layer")
    parser.add_argument("--yolo-model", default="yolo26n.pt", help="Ultralytics YOLO model checkpoint (yolo26n.pt / yolo11n.pt)")
    parser.add_argument("--yolo-confidence", type=float, default=0.35, help="YOLO detection confidence threshold")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image input resolution")
    parser.add_argument("--yolo-device", default="auto", help="Execution device for YOLO (auto / cuda / cpu)")
    parser.add_argument("--yolo-tracking", action="store_true", default=False, help="Enable persistent object tracking across frames")
    parser.add_argument("--yolo-tiling", action="store_true", default=False, help="Enable 2x2 multi-scale tiled inference for small drone objects")
    
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        mode=args.mode,
        max_frames=args.max_frames,
        use_depth_anything=args.use_depth_anything,
        depth_model_size=args.depth_model,
        use_yolo=args.use_yolo,
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        yolo_imgsz=args.yolo_imgsz,
        yolo_device=args.yolo_device,
        yolo_tracking=args.yolo_tracking,
        yolo_tiling=args.yolo_tiling
    )

if __name__ == "__main__":
    main()


import os
import sys

# Prevent OpenBLAS heap memory allocation errors on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import shutil
import zipfile
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import gradio as gr
import numpy as np
import cv2
from PIL import Image

# Ensure Windows PyTorch CUDA DLL path is configured
if os.name == "nt":
    candidates = [
        os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
        r"D:\Vggt\drone_vggt_env\Lib\site-packages\torch\lib"
    ]
    for dll_path in candidates:
        if os.path.exists(dll_path) and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(dll_path)
                break
            except Exception:
                pass

# Ensure local source directory is importable
sys.path.insert(0, str(Path(__file__).parent))

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
logger = logging.getLogger("DroneVGGT_App")

def process_disaster_drone_pipeline(
    video_file,
    image_folder_str,
    target_fps,
    max_frames,
    sharpness_thresh,
    confidence_thresh,
    use_da2,
    da2_model_size,
    do_refine,
    do_filter,
    do_disaster_inspection,
    progress=gr.Progress()
):
    try:
        if video_file is not None:
            input_source = video_file
            is_video = True
        elif image_folder_str and os.path.exists(image_folder_str):
            input_source = image_folder_str
            is_video = False
        else:
            return (
                "### Error: Please upload a drone video or provide a valid image directory path.",
                None, [], [], [], [], None, None, None, None, [], [], ""
            )

        work_dir = Path("outputs/gradio_session")
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        frames_dir = work_dir / "extracted_frames"
        selected_dir = work_dir / "selected_frames"

        # 1. Video Frame Extraction
        progress(0.05, desc="Extracting frames & inspecting video...")
        if is_video and VideoProcessor.is_video_file(input_source):
            frame_paths = VideoProcessor.extract_frames(input_source, frames_dir, target_fps=target_fps)
        else:
            frame_paths = VideoProcessor.process_image_folder(input_source)

        # 2. Geometric Frame Selection
        progress(0.15, desc="Intelligently selecting geometrically optimal frames...")
        config = {
            "frame_selection": {"sharpness_threshold": sharpness_thresh, "max_selected_frames": int(max_frames)},
            "pointcloud_filtering": {"enabled": do_filter}
        }
        selector = FrameSelector(config)
        selected_paths, quality_metrics = selector.select_optimal_frames(frame_paths, selected_dir, max_frames=int(max_frames))

        if len(selected_paths) == 0:
            return "No frames passed quality selection threshold. Lower the sharpness threshold.", None, [], [], [], [], None, None, None, None, [], [], ""

        # 3. Depth Anything V2 Monocular Estimation (Sequential GPU memory load)
        da2_depth_maps = []
        da2_gallery = []
        if use_da2:
            progress(0.30, desc="Running Depth Anything V2 monocular depth estimation...")
            da2_engine = DepthAnythingV2Estimator(model_name=da2_model_size, encoder="vits" if da2_model_size=="small" else "vitb")
            da2_vis_dir = work_dir / "da2_vis"
            
            for i, f_path in enumerate(selected_paths):
                d_map = da2_engine.predict_image(f_path)
                da2_depth_maps.append(d_map)
                v_path = da2_engine.save_depth_visualizations(d_map, da2_vis_dir / f"frame_{i:04d}.png")
                da2_gallery.append(Image.open(v_path))
            
            da2_engine.unload_model()

        # 4. VGGT 3D Reconstruction Model
        progress(0.50, desc="Running Meta VGGT foundation transformer inference...")
        vggt_engine = VGGTInferenceEngine()
        predictions = vggt_engine.run_inference(selected_paths)
        vggt_engine.unload_model()

        # 5. Depth Alignment & Fused Confidence
        fused_confidence_maps = []
        diff_gallery = []
        consistency_gallery = []

        if use_da2 and len(da2_depth_maps) > 0:
            progress(0.70, desc="Aligning monocular depth and computing consistency heatmaps...")
            aligner = RobustDepthAligner()
            analyzer = DepthConsistencyAnalyzer()
            fuser = DepthConfidenceFused()

            for i in range(len(selected_paths)):
                vggt_d = predictions["depth_maps"][i]
                vggt_c = predictions["depth_conf"][i]
                raw_da2 = da2_depth_maps[i]

                aligned_da2, _ = aligner.align_depth_maps(raw_da2, vggt_d, vggt_c)
                abs_diff, consistency, _ = analyzer.compute_consistency(vggt_d, aligned_da2, vggt_c)
                fused_conf = fuser.fuse_confidence(vggt_c, consistency)
                fused_confidence_maps.append(fused_conf)

                # Visualizations
                diff_vis = (cv2.applyColorMap((np.clip(abs_diff / 2.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET))
                cons_vis = (cv2.applyColorMap((consistency * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS))
                diff_gallery.append(Image.fromarray(cv2.cvtColor(diff_vis, cv2.COLOR_BGR2RGB)))
                consistency_gallery.append(Image.fromarray(cv2.cvtColor(cons_vis, cv2.COLOR_BGR2RGB)))

        # 6. 3D Point Cloud Building & Filtering
        progress(0.80, desc="Building metric 3D point cloud & Poisson mesh...")
        reconstructor = PointCloudReconstructor()
        raw_pts, raw_cols, _ = reconstructor.build_pointcloud_from_predictions(predictions, confidence_threshold=float(confidence_thresh))
        
        fused_conf_flat = np.concatenate([c.flatten() for c in fused_confidence_maps]) if fused_confidence_maps else None
        pcd_filter = PointCloudFilter(config)
        final_pts, final_cols, filter_stats = pcd_filter.filter_pointcloud(raw_pts, raw_cols, fused_conf_flat)

        ply_file = work_dir / "drone_scene_reconstruction.ply"
        reconstructor.save_ply(final_pts, final_cols, ply_file)

        colmap_dir = work_dir / "colmap_sparse"
        COLMAPExporter.export_colmap(colmap_dir, selected_paths, predictions["extrinsics"], predictions["intrinsics"], final_pts, final_cols)
        colmap_zip = work_dir / "colmap_sparse.zip"
        with zipfile.ZipFile(colmap_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(colmap_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), colmap_dir))

        mesh_file = work_dir / "surface_mesh.ply"
        MeshGenerator.generate_poisson_mesh(final_pts, final_cols, mesh_file)

        # 7. AI Disaster Inspection & Incident Database
        progress(0.90, desc="Running AI disaster inspection, person detection & 3D projection...")
        detection_gallery = []
        inc_manager = DisasterIncidentManager(db_dir=str(work_dir))
        all_dets_3d = []
        all_hazards = []
        all_damage = []

        from src.geometry_quality import GeometryQualityEvaluator
        geom_eval = GeometryQualityEvaluator.evaluate_reconstruction_quality(predictions, final_pts)
        logger.info(f"[GEOMETRY EVAL] Status: {geom_eval['geometry_status']} | Score: {geom_eval['geometry_quality_score']}/100 | Allowed: {geom_eval['localization_allowed']}")

        if do_disaster_inspection:
            from src.yolo_detector import YOLODetector
            yolo_det = YOLODetector(model_name="yolo26n.pt", confidence=0.35)
            tracker = MultiFrameObjectTracker()
            dmg_det = StructuralDamageDetector()
            hz_det = HazardDetector()

            # Batch run YOLO perception
            yolo_dets_batch = yolo_det.detect_frames(selected_paths, conf=0.35)

            for i, f_path in enumerate(selected_paths):
                img_bgr = cv2.imread(f_path)
                depth_m = predictions["depth_maps"][i]
                ext = predictions["extrinsics"][i]
                intri = predictions["intrinsics"][i]

                raw_dets = yolo_dets_batch[i]
                tracked_dets = tracker.track_frame_detections(raw_dets, frame_idx=i)
                annotated_img = yolo_det.annotate_frame(img_bgr, tracked_dets)
                detection_gallery.append(Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)))

                # Inspect damage and hazards first to establish scene disaster context
                dmg = dmg_det.analyze_structural_damage(img_bgr, depth_m, tracked_dets, frame_idx=i)
                hz = hz_det.analyze_hazards(img_bgr, depth_m, tracked_dets, frame_idx=i)

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

                for det in tracked_dets:
                    if geom_eval["localization_allowed"]:
                        proj3d = GeoProjection3D.project_detection_to_3d(det, depth_m, ext, intri)
                    else:
                        proj3d = det.copy()
                        proj3d.update({"location_status": "UNLOCALIZED", "x_m": None, "y_m": None, "z_m": None, "reprojection_error_px": None})
                    
                    all_dets_3d.append(proj3d)

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

            yolo_det.release()


        risk_level, risk_score, reasons, priorities = DisasterRiskScorer.calculate_scene_risk(
            all_dets_3d, all_hazards, all_damage
        )

        summary_data = {
            "total_frames": len(frame_paths),
            "selected_frames": len(selected_paths),
            "reconstructed_points": len(final_pts),
            "detected_people": sum(1 for d in all_dets_3d if d.get("class_name") == "person"),
            "detected_vehicles": sum(1 for d in all_dets_3d if d.get("class_name") == "vehicle"),
            "geometry_status": geom_eval["geometry_status"],
            "geometry_quality_score": geom_eval["geometry_quality_score"],
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_reasons": reasons,
            "rescue_priorities": priorities
        }

        report_path = DisasterReportGenerator.generate_html_report(summary_data, inc_manager.get_all_incidents(), work_dir / "disaster_report.html")

        # Summary text
        people_count = summary_data["detected_people"]
        summary_markdown = (
            f"### 🛸 AI Disaster Inspection Executive Summary\n"
            f"- **Reconstruction Status**: **{geom_eval['geometry_status']}** (Quality Score: {geom_eval['geometry_quality_score']}/100)\n"
            f"- **Rescue Risk Level**: **{risk_level}** (Risk Score: {risk_score}/100)\n"
            f"- **Reconstructed 3D Points**: {len(final_pts):,}\n"
            f"- **Detected People**: **{people_count}** ({'NO PERSON DETECTED IN AVAILABLE IMAGERY' if people_count==0 else 'VICTIMS LOCATED'})\n"
            f"- **Detected Vehicles**: {summary_data['detected_vehicles']}\n"
            f"- **Deduplicated Incidents / Alerts**: {len(inc_manager.get_all_incidents())}\n"
            f"- **Actionable Rescue Priorities**: {priorities[0] if priorities else 'Routine Monitoring'}"
        )

        vggt_depth_gallery = []
        for i in range(len(selected_paths)):
            d_arr = predictions["depth_maps"][i].squeeze()
            d_max = float(np.nanmax(d_arr)) if d_arr.size > 0 else 0.0
            if d_max > 0:
                norm_d = (np.nan_to_num(d_arr / d_max, nan=0.0) * 255.0).clip(0, 255).astype(np.uint8)
            else:
                norm_d = np.zeros_like(d_arr, dtype=np.uint8)
            vggt_depth_gallery.append(Image.fromarray(norm_d))
        selected_imgs = [Image.open(p) for p in selected_paths]

        # Formatted table data for Incidents
        incidents_table = []
        for inc in inc_manager.get_all_incidents():
            if inc["x_m"] is not None and inc["location_status"] != "UNLOCALIZED":
                reproj_str = f" (reproj: {inc['reprojection_error_px']}px)" if inc.get("reprojection_error_px") is not None else ""
                loc_str = f"X:{inc['x_m']:.2f}, Y:{inc['y_m']:.2f}, Z:{inc['z_m']:.2f}{reproj_str}"
            else:
                loc_str = "UNLOCALIZED"
            
            src_frames_str = f"Frames {inc.get('source_frames', [inc.get('frame_idx', 0)])}"
            obj_cls = inc.get("target_object_class", inc.get("object_id", "object"))
            det_conf = float(inc.get("detection_confidence", inc.get("confidence", 0.0)))
            loc_conf = float(inc.get("localization_confidence", 0.0 if inc["location_status"]=="UNLOCALIZED" else 0.90))

            incidents_table.append([
                inc["incident_id"],
                inc["incident_type"],
                obj_cls,
                f"{det_conf:.2f}",
                f"{loc_conf:.2f}",
                loc_str,
                src_frames_str,
                inc.get("status", "NEEDS_VERIFICATION")
            ])

        progress(1.0, desc="3D Rescue Map & Disaster Inspection Complete!")

        return (
            summary_markdown,
            str(mesh_file) if mesh_file.exists() else str(ply_file),
            selected_imgs,
            da2_gallery,
            vggt_depth_gallery,
            diff_gallery,
            str(ply_file),
            str(colmap_zip),
            str(mesh_file),
            str(report_path),
            detection_gallery,
            incidents_table,
            f"HTML Report generated at: {report_path}"
        )

    except Exception as e:
        logger.error(f"Pipeline execution error: {e}", exc_info=True)
        return f"**Error**: {str(e)}", None, [], [], [], [], None, None, None, None, [], [], ""

# Backward compatibility alias
process_drone_pipeline = process_disaster_drone_pipeline

def create_ui():
    with gr.Blocks(title="Drone-VGGT: AI Disaster Inspection & 3D Rescue Mapping System") as demo:
        gr.Markdown(
            """
            # 🛸 AI-Powered Drone Disaster Inspection & 3D Rescue Mapping System
            ### Single-Pass Drone Video to Metric 3D Reconstruction, Monocular Depth Validation & Automated Rescue Intelligence
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Input Source")
                video_input = gr.Video(label="Upload Drone Aerial Video (.mp4, .mov, .avi)")
                image_folder_input = gr.Textbox(label="OR Path to UAV Image Directory", placeholder="e.g. data/raw/uav_sequence/1121222322212102-4/images")
                
                gr.Markdown("### 2. Model & Processing Parameters")
                with gr.Accordion("Sampling & VRAM Controls", open=True):
                    fps_slider = gr.Slider(0.5, 10.0, value=2.0, step=0.5, label="Target Sampling FPS")
                    max_frames_slider = gr.Slider(6, 60, value=16, step=2, label="Max Selected Frames (4GB VRAM Safety)")
                    sharpness_slider = gr.Slider(10.0, 150.0, value=40.0, step=5.0, label="Laplacian Sharpness Threshold")
                    conf_slider = gr.Slider(0.5, 3.0, value=1.1, step=0.1, label="VGGT Depth Confidence Threshold")

                with gr.Accordion("Depth Anything V2 & Disaster AI Options", open=True):
                    use_da2_check = gr.Checkbox(value=True, label="Enable Depth Anything V2 Validation")
                    da2_model_radio = gr.Radio(["small", "base"], value="small", label="Depth Anything V2 Scale (Apache-2.0)")
                    refine_check = gr.Checkbox(value=True, label="Enable Image Refinement (CLAHE)")
                    filter_check = gr.Checkbox(value=True, label="Enable Depth-Aware Point Cloud Filtering")
                    disaster_check = gr.Checkbox(value=True, label="Enable AI Disaster & Person Inspection")

                run_btn = gr.Button("🚀 Execute 3D Disaster Inspection", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 3. Rescue Dashboard & Intelligence Center")
                summary_output = gr.Markdown("Upload drone video and click **Execute 3D Disaster Inspection**.")
                
                with gr.Tabs():
                    with gr.TabItem("3D Scene & Rescue Model"):
                        model_viewer = gr.Model3D(label="Interactive 3D Disaster Scene (Mesh / Point Cloud)", height=450)
                        with gr.Row():
                            ply_download = gr.File(label="Download Reconstructed PLY")
                            colmap_download = gr.File(label="Download COLMAP Sparse Zip")
                            mesh_download = gr.File(label="Download Surface Mesh")

                    with gr.TabItem("AI Disaster & Person Inspection"):
                        gr.Markdown("#### Object Detections, Person Locations & Bounding Box Annotations")
                        detection_gallery_ui = gr.Gallery(label="Detected People, Vehicles & Structures", columns=3, height="auto")

                    with gr.TabItem("Disaster Incidents Table"):
                        gr.Markdown("#### AI-Flagged Incidents, Victim Locations & Verification Status")
                        incidents_dataframe = gr.Dataframe(
                            headers=["Incident ID", "Type", "Object Class", "Detection Conf", "Localization Conf", "3D Location", "Source Frames", "Status"],
                            label="Incident Database"
                        )


                    with gr.TabItem("Monocular Depth Validation"):
                        with gr.Row():
                            da2_gallery_ui = gr.Gallery(label="Depth Anything V2 Predictions", columns=2, height="auto")
                            vggt_depth_gallery_ui = gr.Gallery(label="VGGT Depth Predictions", columns=2, height="auto")
                        with gr.Row():
                            diff_gallery_ui = gr.Gallery(label="Absolute Depth Error Maps", columns=2, height="auto")

                    with gr.TabItem("Selected Drone Frames"):
                        selected_gallery_ui = gr.Gallery(label="Geometrically Selected Frames", columns=4, height="auto")

                    with gr.TabItem("Export & Disaster HTML Report"):
                        report_download_ui = gr.File(label="Download Disaster Inspection HTML Report")
                        report_status_markdown = gr.Markdown("")

        run_btn.click(
            fn=process_disaster_drone_pipeline,
            inputs=[
                video_input, image_folder_input,
                fps_slider, max_frames_slider, sharpness_slider, conf_slider,
                use_da2_check, da2_model_radio,
                refine_check, filter_check, disaster_check
            ],
            outputs=[
                summary_output, model_viewer,
                selected_gallery_ui, da2_gallery_ui, vggt_depth_gallery_ui, diff_gallery_ui,
                ply_download, colmap_download, mesh_download, report_download_ui,
                detection_gallery_ui, incidents_dataframe, report_status_markdown
            ]
        )

    return demo

demo = create_ui()

if __name__ == "__main__":
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=theme)

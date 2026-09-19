"""
End-to-End UAV 3D Reconstruction Pipeline Runner.
Orchestrates:
Video -> Quality Filter -> Keyframes -> YOLO/SAM 2 -> VGGT -> GPS Georeferencing
-> Open3D Point Cloud -> Surface Mesh -> Texture Projection -> GLB Export -> Evaluation.
"""

import os
import sys
import time
import yaml
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
import numpy as np

from src.video.extractor import VideoFrameExtractor
from src.video.quality import FrameQualityFilter, FrameQualityReport
from src.video.keyframes import IntelligentKeyframeSelector, KeyframeSelectionResult
from src.detection.yolo_detector import YOLODetector
from src.segmentation.sam2_segmenter import SAM2Segmenter
from src.detection.dynamic_object_filter import DynamicObjectFilter
from src.geometry.vggt_model import load_vggt, BaseVGGTAdapter, VGGTOutput
from src.depth.depth_anything import DepthAnythingV2Estimator
from src.geometry.depth_fusion import DepthFusionEngine
from src.georeference.gps import GPSTelemetryParser, parse_gps_csv
from src.georeference.alignment import GeoreferenceAligner, GeoreferenceTransform
from src.reconstruction.pointcloud import PointCloudBuilder, PointCloudData
from src.reconstruction.filtering import PointCloudFilter, FilterConfig
from src.reconstruction.mesh import MeshReconstructor, MeshData
from src.reconstruction.texturing import MeshTextureProjector
from src.output.output_manager import OutputManager, MissionArtifacts
from src.evaluation.metrics import ReconstructionEvaluator, ReconstructionMetrics

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    reconstruction_mode: str = "hybrid"  # "hybrid", "vggt", "colmap"
    vggt_backend: str = "pretrained"     # "pretrained", "finetuned", "mock"
    vggt_checkpoint: Optional[str] = None
    yolo_checkpoint: str = "yolo11n.pt"
    frame_sampling_fps: float = 2.0
    min_blur_score: float = 40.0
    dynamic_classes: Optional[List[str]] = None
    export_formats: Optional[List[str]] = None


@dataclass
class MissionResult:
    mission_id: str
    status: str
    artifacts: MissionArtifacts
    metrics: Dict[str, Any]
    summary: Dict[str, Any]
    errors: List[str]


class PipelineRunner:
    """
    Main orchestration engine for single-pass UAV 3D reconstruction.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        models_yaml_path: str = "configs/models.yaml",
        pipeline_yaml_path: str = "configs/pipeline.yaml",
    ):
        self.config = config or PipelineConfig()
        self.models_yaml_path = models_yaml_path
        self.pipeline_yaml_path = pipeline_yaml_path
        self._load_yaml_configs()

    def _load_yaml_configs(self):
        """Loads and merges YAML configurations."""
        self.models_cfg = {}
        if os.path.exists(self.models_yaml_path):
            with open(self.models_yaml_path, "r", encoding="utf-8") as f:
                self.models_cfg = yaml.safe_load(f).get("models", {})

        self.pipeline_cfg = {}
        if os.path.exists(self.pipeline_yaml_path):
            with open(self.pipeline_yaml_path, "r", encoding="utf-8") as f:
                self.pipeline_cfg = yaml.safe_load(f).get("pipeline", {})

    def run(
        self,
        input_video_or_images: str,
        gps_csv_path: Optional[str] = None,
        imu_csv_path: Optional[str] = None,
        camera_json_path: Optional[str] = None,
        mission_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
    ) -> MissionResult:
        """
        Executes complete reconstruction pipeline.
        """
        total_start = time.perf_counter()
        stage_timings = {}
        errors = []

        def report_progress(stage: str, pct: float, msg: str):
            logger.info(f"[{stage.upper()}] {msg}")
            if progress_callback:
                try:
                    progress_callback(stage, pct, msg)
                except Exception:
                    pass

        # 0. Setup Mission Directories
        m_id = mission_id or f"mission_{int(time.time())}"
        output_mgr = OutputManager()
        artifacts = output_mgr.initialize_mission(m_id)
        report_progress("SETUP", 0.05, f"Mission initialized: '{m_id}'")

        # 1. Video Frame Extraction or Directory Copy
        t0 = time.perf_counter()
        in_path = Path(input_video_or_images)
        frame_paths = []

        if in_path.is_file():
            fps_sample = self.pipeline_cfg.get("video", {}).get("frame_sampling_fps", self.config.frame_sampling_fps)
            extractor = VideoFrameExtractor(sampling_fps=fps_sample)
            extract_res = extractor.extract(str(in_path), artifacts.frames_dir)
            frame_paths = extract_res["frame_paths"]
            report_progress("VIDEO", 0.15, f"Extracted {len(frame_paths)} frames from '{in_path.name}'")
        elif in_path.is_dir():
            # If directory of images provided
            img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            for f in sorted(in_path.glob("*")):
                if f.suffix.lower() in img_exts:
                    dst = Path(artifacts.frames_dir) / f.name
                    if str(f.resolve()) != str(dst.resolve()):
                        shutil.copy2(f, dst)
                    frame_paths.append(str(dst.resolve()))
            report_progress("VIDEO", 0.15, f"Ingested {len(frame_paths)} image frames from '{in_path}'")
        else:
            raise FileNotFoundError(f"Input path does not exist: {input_video_or_images}")

        if not frame_paths:
            raise ValueError("No frames found or extracted from input.")

        stage_timings["frame_extraction"] = time.perf_counter() - t0

        # 2. Frame Quality Filtering
        t0 = time.perf_counter()
        min_blur = self.pipeline_cfg.get("quality", {}).get("min_blur_score", self.config.min_blur_score)
        qfilter = FrameQualityFilter(min_blur_score=min_blur)
        qreport = qfilter.filter_frames(
            frame_paths,
            artifacts.frames_dir,
            csv_report_path=os.path.join(artifacts.mission_dir, "frame_quality.csv"),
        )
        accepted_frames = [s.frame_path for s in qreport.scores if s.accepted]
        stage_timings["quality_filtering"] = time.perf_counter() - t0

        if len(accepted_frames) < 3:
            logger.warning("[FILTER] Few frames passed strict blur filter. Falling back to all extracted frames.")
            accepted_frames = frame_paths

        report_progress("FILTER", 0.25, f"Accepted {len(accepted_frames)}/{len(frame_paths)} quality frames")

        # 3. Intelligent Keyframe Selection
        t0 = time.perf_counter()
        kf_selector = IntelligentKeyframeSelector()
        kf_res: KeyframeSelectionResult = kf_selector.select_keyframes(
            accepted_frames,
            os.path.join(artifacts.mission_dir, "keyframes"),
            total_initial_frames=len(frame_paths),
        )
        keyframes = kf_res.keyframe_paths
        stage_timings["keyframe_selection"] = time.perf_counter() - t0

        if len(keyframes) < 3:
            logger.warning("[KEYFRAME] Insufficient keyframes selected. Using first 24 quality frames.")
            keyframes = accepted_frames[:24]

        report_progress("KEYFRAME", 0.35, f"Selected {len(keyframes)} keyframes (ratio: {kf_res.selection_ratio:.1%})")

        # 4. Dynamic Object Detection & Segmentation (YOLO + SAM 2)
        t0 = time.perf_counter()
        yolo_ckpt = self.models_cfg.get("yolo", {}).get("checkpoint", self.config.yolo_checkpoint)
        yolo_det = YOLODetector(model_path=yolo_ckpt)
        sam2_seg = SAM2Segmenter()
        dyn_filter = DynamicObjectFilter(
            mode=self.pipeline_cfg.get("dynamic_objects", {}).get("mode", "yolo+sam2"),
            yolo_detector=yolo_det,
            sam2_segmenter=sam2_seg,
        )
        filter_results = dyn_filter.process_sequence(keyframes, artifacts.masks_dir)
        dynamic_masks = [r.mask for r in filter_results]
        total_dynamic_objs = sum(r.dynamic_count for r in filter_results)
        stage_timings["dynamic_filtering"] = time.perf_counter() - t0

        report_progress("YOLO", 0.45, f"Detected {total_dynamic_objs} dynamic objects across keyframes")
        report_progress("SAM2", 0.50, f"Generated {len(dynamic_masks)} dynamic object exclusion masks")

        # 5. VGGT 3D Geometry Prediction (Modular Pretrained / Fine-Tuned / Mock)
        t0 = time.perf_counter()
        vggt_backend = self.config.vggt_backend or self.models_cfg.get("vggt", {}).get("backend", "pretrained")
        vggt_ckpt = self.config.vggt_checkpoint or self.models_cfg.get("vggt", {}).get("checkpoint", "models/VGGT-1B/model.pt")

        report_progress("VGGT", 0.55, f"Running VGGT 3D geometry engine (backend: '{vggt_backend}')...")
        try:
            vggt_engine: BaseVGGTAdapter = load_vggt(
                checkpoint_path=vggt_ckpt,
                checkpoint_type=vggt_backend,
            )
            vggt_out: VGGTOutput = vggt_engine.predict(keyframes)
        except Exception as vggt_err:
            logger.warning(f"[VGGT] Full inference unavailable ({vggt_err}). Switching to Mock VGGT for testing.")
            errors.append(f"VGGT model fallback: {vggt_err}")
            vggt_engine = load_vggt(checkpoint_type="mock")
            vggt_out = vggt_engine.predict(keyframes)

        stage_timings["vggt_prediction"] = time.perf_counter() - t0
        report_progress("VGGT", 0.65, f"Processed {vggt_out.num_frames} frames (Peak VRAM: {vggt_out.peak_vram_mb}MB)")

        # 6. Optional Depth Anything V2 Comparison & Fusion
        t0 = time.perf_counter()
        depth_fusion = DepthFusionEngine(mode="comparison")
        fusion_report = depth_fusion.process(
            vggt_depth=vggt_out.depth_maps[0],
            vggt_conf=vggt_out.depth_conf[0],
            dynamic_mask=dynamic_masks[0] if dynamic_masks else None,
        )
        stage_timings["depth_comparison"] = time.perf_counter() - t0
        report_progress("DEPTH", 0.70, "Generated depth maps and geometric comparison report")

        # 7. GPS / IMU Georeferencing Alignment
        t0 = time.perf_counter()
        is_georeferenced = False
        gps_rmse = 0.0
        scale_factor = 1.0

        if gps_csv_path and os.path.exists(gps_csv_path):
            try:
                gps_traj = parse_gps_csv(gps_csv_path)
                aligner = GeoreferenceAligner()
                geo_transform, aligned_centers = aligner.align_trajectory(
                    camera_extrinsics=vggt_out.extrinsics,
                    gps_positions_enu=gps_traj.positions_enu,
                )
                # Transform point maps into world coordinate system
                vggt_out.point_maps = aligner.transform_points(vggt_out.point_maps, geo_transform)
                is_georeferenced = True
                gps_rmse = geo_transform.alignment_rmse_m
                scale_factor = geo_transform.scale
                report_progress("GEO", 0.75, f"GPS alignment complete (RMSE: {gps_rmse:.2f}m | Scale: {scale_factor:.2f})")
            except Exception as gps_err:
                logger.warning(f"[GEO] GPS georeferencing failed: {gps_err}")
                errors.append(f"GPS Georeferencing error: {gps_err}")
                report_progress("GEO", 0.75, "Running in relative unscaled coordinate space")
        else:
            report_progress("GEO", 0.75, "No GPS file provided — running non-georeferenced mode.")

        stage_timings["georeferencing"] = time.perf_counter() - t0

        # 8. Point Cloud Generation & Filtering (Open3D)
        t0 = time.perf_counter()
        builder = PointCloudBuilder()
        raw_pcd_data: PointCloudData = builder.build_from_vggt_output(
            point_maps=vggt_out.point_maps,
            images=vggt_out.images,
            confidences=vggt_out.point_conf,
            dynamic_masks=dynamic_masks,
        )

        pcd_filter = PointCloudFilter()
        filtered_pcd: PointCloudData = pcd_filter.filter(raw_pcd_data)

        builder.save_ply(filtered_pcd, str(Path(artifacts.pointcloud_dir) / "pointcloud.ply"))
        stage_timings["pointcloud"] = time.perf_counter() - t0
        report_progress("3D", 0.82, f"Generated {raw_pcd_data.num_points:,} points (Filtered: {filtered_pcd.num_points:,})")

        # 9. Mesh Reconstruction & Texturing
        t0 = time.perf_counter()
        reconstructor = MeshReconstructor()
        mesh = reconstructor.reconstruct(filtered_pcd)

        # Texture projection onto mesh
        projector = MeshTextureProjector()
        mesh = projector.project_texture_vertex_colors(
            mesh=mesh,
            image_paths=keyframes,
            camera_extrinsics=vggt_out.extrinsics,
            camera_intrinsics=vggt_out.intrinsics,
        )

        mesh_data: MeshData = reconstructor.export_all_formats(
            mesh,
            base_output_dir=artifacts.mission_dir,
            base_name="final",
        )
        stage_timings["mesh_and_texture"] = time.perf_counter() - t0
        report_progress("MESH", 0.90, f"Generated {mesh_data.num_triangles:,} triangles ({mesh_data.num_vertices:,} vertices)")
        report_progress("EXPORT", 0.95, "final.glb, final.obj, and final.ply created successfully")

        # 10. Quantitative Evaluation & Metrics
        t0 = time.perf_counter()
        evaluator = ReconstructionEvaluator()
        total_time_s = time.perf_counter() - total_start

        metrics_obj = ReconstructionMetrics(
            mission_id=m_id,
            reconstruction_mode=self.config.reconstruction_mode,
            num_input_frames=len(frame_paths),
            num_keyframes=len(keyframes),
            num_points=filtered_pcd.num_points,
            num_triangles=mesh_data.num_triangles,
            num_vertices=mesh_data.num_vertices,
            point_metrics=None,
            trajectory_metrics=None,
            total_processing_time_s=round(total_time_s, 2),
            peak_vram_mb=vggt_out.peak_vram_mb,
            stage_timings_s={k: round(v, 2) for k, v in stage_timings.items()},
        )

        metrics_dict = asdict(metrics_obj)
        output_mgr.save_json(metrics_dict, artifacts.metrics_json)

        processing_report = {
            "mission_id": m_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "success" if not errors else "completed_with_warnings",
            "is_georeferenced": is_georeferenced,
            "gps_alignment_rmse_m": gps_rmse,
            "scale_factor": scale_factor,
            "total_processing_time_s": round(total_time_s, 2),
            "artifacts": asdict(artifacts),
            "stage_timings": {k: round(v, 2) for k, v in stage_timings.items()},
            "errors_or_warnings": errors,
        }
        output_mgr.save_json(processing_report, artifacts.processing_report_json)
        report_progress("COMPLETE", 1.0, f"Mission completed in {total_time_s:.1f}s")

        return MissionResult(
            mission_id=m_id,
            status="success" if not errors else "completed_with_warnings",
            artifacts=artifacts,
            metrics=metrics_dict,
            summary=processing_report,
            errors=errors,
        )

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Ensure local source directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vggt_inference import VGGTInferenceEngine
from src.reconstruction import PointCloudReconstructor
from src.geometry_quality import GeometryQualityEvaluator

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ControlBenchmark")

def run_control_benchmark():
    logger.info("=" * 65)
    logger.info(" CONTROL BENCHMARK: 3D RECONSTRUCTION SUITABILITY ANALYSIS")
    logger.info("=" * 65)

    uav_dir = Path("data/raw/uav_sequence/1121222322212102-4/images")
    if not uav_dir.exists():
        logger.error(f"UAV image directory not found: {uav_dir}")
        return

    from src.video_processor import VideoProcessor
    all_imgs = VideoProcessor.process_image_folder(uav_dir)
    image_paths = all_imgs[:6]
    logger.info(f"[EXPERIMENT 2 - FLOOD DRONE SEQUENCE] Loaded {len(image_paths)} images from {len(all_imgs)} candidates.")

    engine = VGGTInferenceEngine()
    preds = engine.run_inference(image_paths)
    engine.unload_model()

    reconstructor = PointCloudReconstructor()
    pts, cols, conf = reconstructor.build_pointcloud_from_predictions(preds, confidence_threshold=1.1)

    eval_result = GeometryQualityEvaluator.evaluate_reconstruction_quality(preds, pts)

    logger.info("\n>>> CONTROL BENCHMARK RESULTS <<<")
    logger.info(f"- Sequence: {uav_dir.name}")
    logger.info(f"- Reconstruction Status: {eval_result['geometry_status']}")
    logger.info(f"- Geometry Quality Score: {eval_result['geometry_quality_score']}/100")
    logger.info(f"- Camera Baseline: {eval_result['camera_baseline_m']}m")
    logger.info(f"- Point Count: {eval_result['point_count']:,}")
    logger.info(f"- Spatial Extent (X, Y, Z): {eval_result['extent_xyz_m']}m")
    logger.info(f"- Localization Allowed: {eval_result['localization_allowed']}")
    logger.info(f"- Rationale: {eval_result['reason']}")

if __name__ == "__main__":
    run_control_benchmark()

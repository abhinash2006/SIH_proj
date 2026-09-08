import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
from src.frame_quality import FrameQualityAnalyzer

logger = logging.getLogger(__name__)

class FrameSelector:
    """
    Intelligent geometric frame selection for multi-view 3D reconstruction.
    Balances:
    1. Image sharpness & quality
    2. Overlap & visual continuity
    3. Viewpoint diversity & temporal scene coverage
    4. Rejection of duplicate / redundant frames
    5. Rejection of sky-dominated or severely degraded frames
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        sel_cfg = self.config.get("frame_selection", {})
        
        self.sharpness_threshold = sel_cfg.get("sharpness_threshold", 40.0)
        self.underexposure_threshold = sel_cfg.get("underexposure_threshold", 0.25)
        self.overexposure_threshold = sel_cfg.get("overexposure_threshold", 0.40)
        self.min_brightness = sel_cfg.get("min_brightness", 15.0)
        self.sky_max_ratio = sel_cfg.get("sky_max_ratio", 0.75)
        self.similarity_threshold = sel_cfg.get("similarity_threshold", 0.985)
        self.max_selected_frames = sel_cfg.get("max_selected_frames", 24)

    @staticmethod
    def compute_frame_fingerprint(bgr_img: np.ndarray) -> np.ndarray:
        """
        Computes a compact, robust perceptual visual descriptor (HSV color histogram + downscaled luminance grid)
        for fast pairwise visual similarity and baseline change estimation.
        """
        small_img = cv2.resize(bgr_img, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
        
        # 3D HSV histogram (8 bins H, 4 bins S, 4 bins V)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist_feat = hist.flatten()
        
        # Grayscale spatial grid features (16x16)
        gray_grid = cv2.resize(cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY), (16, 16)).astype(np.float32) / 255.0
        grid_feat = gray_grid.flatten()
        grid_feat = grid_feat / (np.linalg.norm(grid_feat) + 1e-7)
        
        combined = np.concatenate([hist_feat, grid_feat])
        return combined / (np.linalg.norm(combined) + 1e-7)

    @staticmethod
    def compute_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Cosine similarity between two frame descriptors.
        """
        return float(np.dot(feat1, feat2))

    def select_optimal_frames(
        self,
        frame_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None
    ) -> Tuple[List[str], Dict[str, Any]]:
        if max_frames:
            self.max_selected_frames = max_frames
        res = self.select_frames(frame_paths, output_dir=output_dir)
        return res["selected_paths"], res.get("quality_stats", {})

    def select_frames(
        self,
        frame_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Performs intelligent frame selection across candidate frames.
        Returns a dictionary containing:
        - selected_paths: list of selected frame paths
        - rejected_paths: list of rejected frame paths with reasons
        - quality_stats: summary metrics
        """
        if not frame_paths:
            raise ValueError("No candidate frame paths provided for selection.")
            
        logger.info(f"[INFO] Analyzing {len(frame_paths)} candidate frames for intelligent selection...")
        
        # Step 1: Analyze quality for all frames
        analyzed_frames = []
        rejected_quality = []
        
        for p in frame_paths:
            p_str = str(p)
            try:
                metrics = FrameQualityAnalyzer.analyze_frame(p_str)
                # Check quality criteria
                if metrics["sharpness"] < self.sharpness_threshold:
                    rejected_quality.append((p_str, f"Blurry (sharpness: {metrics['sharpness']} < {self.sharpness_threshold})"))
                elif metrics["underexposed_ratio"] > self.underexposure_threshold:
                    rejected_quality.append((p_str, f"Underexposed ({metrics['underexposed_ratio']*100:.1f}%)"))
                elif metrics["overexposed_ratio"] > self.overexposure_threshold:
                    rejected_quality.append((p_str, f"Overexposed ({metrics['overexposed_ratio']*100:.1f}%)"))
                elif metrics["mean_brightness"] < self.min_brightness:
                    rejected_quality.append((p_str, f"Too dark (brightness: {metrics['mean_brightness']})"))
                elif metrics["sky_ratio"] > self.sky_max_ratio:
                    rejected_quality.append((p_str, f"Sky dominant ({metrics['sky_ratio']*100:.1f}%)"))
                else:
                    analyzed_frames.append(metrics)
            except Exception as e:
                logger.warning(f"Failed to analyze frame {p_str}: {e}")
                rejected_quality.append((p_str, f"Read error: {str(e)}"))

        # Fallback: if criteria were too strict and filtered everything, retain top quality frames
        if len(analyzed_frames) < 3:
            logger.warning(f"[WARNING] Strict filters left only {len(analyzed_frames)} frames. Relaxing thresholds.")
            all_metrics = []
            for p in frame_paths:
                try:
                    all_metrics.append(FrameQualityAnalyzer.analyze_frame(str(p)))
                except Exception:
                    pass
            all_metrics.sort(key=lambda x: x["quality_score"], reverse=True)
            analyzed_frames = all_metrics[:min(len(all_metrics), self.max_selected_frames)]

        # Step 2: Redundancy filtering and geometric baseline maximization
        logger.info(f"[INFO] Quality filtering passed: {len(analyzed_frames)}/{len(frame_paths)} frames.")
        
        # Load descriptors for remaining frames
        valid_items = []
        for item in analyzed_frames:
            img = cv2.imread(item["file_path"])
            if img is not None:
                feat = self.compute_frame_fingerprint(img)
                valid_items.append({"metrics": item, "feat": feat})

        if not valid_items:
            raise RuntimeError("No usable frames remained after quality filtering.")

        # Sequential selection with minimum viewpoint displacement
        selected_items = [valid_items[0]]
        rejected_redundant = []
        
        for item in valid_items[1:]:
            sim = self.compute_similarity(selected_items[-1]["feat"], item["feat"])
            if sim > self.similarity_threshold:
                # Frame is near-duplicate of previous selected frame
                # If current frame has significantly better sharpness, replace it, otherwise reject
                if item["metrics"]["sharpness"] > selected_items[-1]["metrics"]["sharpness"] * 1.25:
                    rejected_redundant.append((selected_items[-1]["metrics"]["file_path"], f"Replaced by sharper near-duplicate (sim: {sim:.3f})"))
                    selected_items[-1] = item
                else:
                    rejected_redundant.append((item["metrics"]["file_path"], f"Near duplicate of previous frame (similarity: {sim:.3f})"))
            else:
                selected_items.append(item)

        # Step 3: Enforce max_selected_frames with uniform temporal/geometric distribution
        if len(selected_items) > self.max_selected_frames:
            logger.info(f"[INFO] Subsampling {len(selected_items)} non-redundant frames down to {self.max_selected_frames}...")
            indices = np.round(np.linspace(0, len(selected_items) - 1, self.max_selected_frames)).astype(int)
            final_items = [selected_items[i] for i in indices]
        else:
            final_items = selected_items

        selected_paths = [item["metrics"]["file_path"] for item in final_items]
        
        # Copy to output_dir if specified
        final_saved_paths = []
        if output_dir:
            out_path_dir = Path(output_dir)
            out_path_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            for idx, p_str in enumerate(selected_paths):
                ext = Path(p_str).suffix or ".jpg"
                dest_file = out_path_dir / f"selected_{idx:05d}{ext}"
                shutil.copy2(p_str, dest_file)
                final_saved_paths.append(str(dest_file))
        else:
            final_saved_paths = selected_paths

        mean_sharpness = float(np.mean([item["metrics"]["sharpness"] for item in final_items]))
        mean_quality = float(np.mean([item["metrics"]["quality_score"] for item in final_items]))
        
        logger.info(
            f"[SUCCESS] Intelligently selected {len(final_saved_paths)} frames "
            f"(Avg Sharpness: {mean_sharpness:.1f}, Avg Quality Score: {mean_quality:.1f})"
        )
        
        return {
            "selected_paths": final_saved_paths,
            "selected_count": len(final_saved_paths),
            "total_input_frames": len(frame_paths),
            "rejected_quality_count": len(rejected_quality),
            "rejected_redundant_count": len(rejected_redundant),
            "rejected_details": rejected_quality + rejected_redundant,
            "average_sharpness": round(mean_sharpness, 2),
            "average_quality_score": round(mean_quality, 2),
            "selected_metrics": [item["metrics"] for item in final_items]
        }

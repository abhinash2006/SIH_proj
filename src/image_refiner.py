import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

class ImageRefiner:
    """
    Conservative, geometry-preserving image preprocessor for 3D reconstruction.
    Improves contrast, signal-to-noise ratio, and edge consistency without
    hallucinating fake textures, changing scene geometry, or altering object shapes.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        ref_cfg = self.config.get("image_refinement", {})
        
        self.enabled = ref_cfg.get("enabled", True)
        self.do_denoise = ref_cfg.get("denoise", True)
        self.denoise_h = ref_cfg.get("denoise_h", 3.0)
        self.do_exposure = ref_cfg.get("exposure_normalization", True)
        self.clahe_clip_limit = ref_cfg.get("clahe_clip_limit", 1.5)
        self.clahe_grid_size = ref_cfg.get("clahe_grid_size", 8)
        self.do_sharpen = ref_cfg.get("sharpening", True)
        self.sharpen_strength = ref_cfg.get("sharpen_strength", 0.25)
        self.do_sky_masking = ref_cfg.get("sky_masking", False)
        self.target_resolution = self.config.get("video", {}).get("target_resolution", 518)

    def refine_image(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Applies conservative refinement pipeline to a single image.
        """
        if not self.enabled or bgr_img is None:
            return bgr_img

        refined = bgr_img.copy()

        # 1. Edge-preserving Denoising (Bilateral filtering prevents blurring structural edges)
        if self.do_denoise:
            # d=5, sigmaColor=25, sigmaSpace=25
            refined = cv2.bilateralFilter(refined, d=5, sigmaColor=25, sigmaSpace=25)

        # 2. Exposure & Contrast Normalization (CLAHE in LAB color space - L channel only)
        if self.do_exposure:
            lab = cv2.cvtColor(refined, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=(self.clahe_grid_size, self.clahe_grid_size)
            )
            l_clahe = clahe.apply(l)
            
            # Subtle blend with original luminance (70% clahe, 30% original) to prevent overshooting
            l_blended = cv2.addWeighted(l_clahe, 0.75, l, 0.25, 0)
            
            lab_merged = cv2.merge([l_blended, a, b])
            refined = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

        # 3. Mild Sharpening via Unsharp Masking (High-frequency boost)
        if self.do_sharpen and self.sharpen_strength > 0:
            gaussian_blur = cv2.GaussianBlur(refined, (0, 0), sigmaX=1.5)
            # unsharp_mask = original + strength * (original - blur)
            refined = cv2.addWeighted(refined, 1.0 + self.sharpen_strength, gaussian_blur, -self.sharpen_strength, 0)

        # 4. Optional Sky Masking (Mask upper background sky without altering buildings/terrain)
        if self.do_sky_masking:
            hsv = cv2.cvtColor(refined, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            sky_mask = (v > 200) & (s < 50)
            # Only apply to upper 50% of the image
            height = refined.shape[0]
            sky_mask[int(height * 0.5):, :] = False
            # Slightly dim the sky to neutralize false feature matches
            refined[sky_mask] = (refined[sky_mask].astype(np.float32) * 0.2).astype(np.uint8)

        return np.clip(refined, 0, 255).astype(np.uint8)

    def process_frame(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> str:
        """
        Reads frame, refines, resizes/pads to target resolution, and saves to output_path.
        """
        input_path = str(input_path)
        output_path = str(output_path)
        
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not read input image: {input_path}")

        # Apply conservative refinement
        refined = self.refine_image(img)
        
        # Save output
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, refined, [cv2.IMWRITE_JPEG_QUALITY, 98])
        return output_path

    def process_frames_batch(
        self,
        input_paths: List[Union[str, Path]],
        output_dir: Union[str, Path]
    ) -> List[str]:
        """
        Batch processes and refines a list of frame paths.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        refined_paths = []
        logger.info(f"[INFO] Running conservative image refinement on {len(input_paths)} frames...")
        
        for idx, in_p in enumerate(input_paths):
            ext = Path(in_p).suffix or ".jpg"
            out_file = out_dir / f"refined_{idx:05d}{ext}"
            self.process_frame(in_p, out_file)
            refined_paths.append(str(out_file))
            
        logger.info(f"[SUCCESS] Refined {len(refined_paths)} frames into {output_dir}")
        return refined_paths

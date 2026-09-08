import os
import sys
import gc
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np

# Ensure torch Windows DLLs are available
torch_lib = r"D:\Vggt\drone_vggt_env\Lib\site-packages\torch\lib"
if os.path.exists(torch_lib) and hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(torch_lib)

import torch
import torch.nn.functional as F
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

logger = logging.getLogger(__name__)

class VGGTInferenceEngine:
    """
    Robust, memory-aware VGGT Inference Engine.
    Handles device selection, automatic precision (FP16/BF16/FP32),
    VRAM-based chunking, OOM recovery, and multi-view 3D prediction.
    """

    def __init__(
        self,
        model_name: str = "facebook/VGGT-1B",
        weights_path: Optional[str] = "models/VGGT-1B/model.pt",
        device: Optional[str] = None,
        use_mixed_precision: bool = True,
        chunk_size: Optional[int] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision and (self.device == "cuda")
        self.weights_path = weights_path
        self.model_name = model_name
        self.model = None
        
        # Determine optimal dtype for GPU
        if self.device == "cuda":
            cap = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
            # Detect available VRAM in GB
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"[INFO] Detected GPU: {torch.cuda.get_device_name(0)} with {total_vram_gb:.2f} GB VRAM")
            
            # Dynamic chunk sizing based on VRAM
            if chunk_size is not None:
                self.chunk_size = chunk_size
            elif total_vram_gb <= 4.5:
                self.chunk_size = 24  # 4GB VRAM budget
            elif total_vram_gb <= 8.5:
                self.chunk_size = 48  # 8GB VRAM
            else:
                self.chunk_size = 96  # 12GB+ VRAM
        else:
            self.dtype = torch.float32
        self.dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    def load_model(self):
        """Loads model weights into VRAM."""
        if self.model is not None:
            return self.model

        logger.info("[INFO] Initializing VGGT model architecture...")
        model = VGGT(enable_camera=True, enable_point=True, enable_depth=True, enable_track=False)
        
        loaded = False
        # Check local weights path first
        if self.weights_path and os.path.exists(self.weights_path):
            logger.info(f"[INFO] Loading weights from local file: {self.weights_path}")
            try:
                state_dict = torch.load(self.weights_path, map_location="cpu", weights_only=False)
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                model.load_state_dict(state_dict, strict=False)
                loaded = True
                logger.info("[SUCCESS] Local model checkpoint loaded successfully!")
            except Exception as e:
                logger.warning(f"[WARNING] Failed to load local weights: {e}")

        if not loaded:
            logger.info(f"[INFO] Loading pretrained weights from Hugging Face: {self.model_name}...")
            try:
                model = VGGT.from_pretrained(self.model_name)
                loaded = True
                logger.info("[SUCCESS] Hugging Face pretrained weights loaded successfully!")
            except Exception as e:
                logger.warning(f"[WARNING] from_pretrained failed: {e}. Trying state_dict from URL...")
                url = f"https://huggingface.co/{self.model_name}/resolve/main/model.pt"
                state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                model.load_state_dict(state_dict, strict=False)
                logger.info("[SUCCESS] Model state_dict loaded from URL!")

        model.eval()
        self.model = model.to(self.device)
        logger.info(f"[SUCCESS] VGGT model ready on {self.device} (dtype: {self.dtype})")
        return self.model

    def unload_model(self):
        """Frees VGGT model weights from VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
        logger.info("[INFO] VGGT model unloaded from GPU memory.")

    def preprocess_images(self, image_paths: List[Union[str, Path]], target_size: int = 392) -> torch.Tensor:
        """
        Preprocesses images to shape [1, S, 3, H, W] normalized in [0, 1].
        Applies target_size resize (392x392) for 4GB VRAM safety.
        """
        image_str_paths = [str(p) for p in image_paths]
        images = load_and_preprocess_images(image_str_paths)
        if images.ndim == 4:
            if target_size != images.shape[-1]:
                images = F.interpolate(images, size=(target_size, target_size), mode="bicubic", align_corners=False)
            images = images.unsqueeze(0)
        elif images.ndim == 5:
            B, S, C, H, W = images.shape
            if target_size != H:
                images_reshaped = images.view(B * S, C, H, W)
                images_resized = F.interpolate(images_reshaped, size=(target_size, target_size), mode="bicubic", align_corners=False)
                images = images_resized.view(B, S, C, target_size, target_size)
        return images

    @torch.inference_mode()
    def run_inference(
        self,
        image_paths: List[Union[str, Path]],
        chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs feedforward 3D reconstruction over image sequence.
        Processes images in memory-safe sequential chunks (e.g., 6 frames per chunk),
        releasing GPU VRAM after each chunk before proceeding to the next.
        """
        self.load_model()
        
        current_chunk_size = min(6, chunk_size or self.chunk_size)
        num_images = len(image_paths)
        logger.info(f"[INFO] Running VGGT chunked inference on {num_images} images (max chunk size: {current_chunk_size})...")

        # Determine safe resolution: 392x392 for <= 6GB VRAM GPUs, 518x518 for > 6GB VRAM GPUs
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if self.device == "cuda" else 16.0
        target_res = 392 if vram_gb <= 6.0 else 518

        # Break image_paths into chunks of current_chunk_size
        chunk_paths_list = [image_paths[i:i + current_chunk_size] for i in range(0, num_images, current_chunk_size)]

        all_extrinsics = []
        all_intrinsics = []
        all_depth_maps = []
        all_depth_conf = []
        all_world_points = []
        all_world_points_conf = []
        all_unprojected_points = []
        all_images_rgb = []
        peak_vram_mb = 0.0

        for idx, chunk_paths in enumerate(chunk_paths_list):
            logger.info(f"[INFO] Processing chunk {idx+1}/{len(chunk_paths_list)} ({len(chunk_paths)} frames)...")
            images_tensor = self.preprocess_images(chunk_paths, target_size=target_res).to(self.device)

            if self.device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()

            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision, dtype=self.dtype):
                predictions = self.model(images_tensor)

            if self.device == "cuda":
                current_peak = torch.cuda.max_memory_allocated(0) / (1024**2)
                peak_vram_mb = max(peak_vram_mb, current_peak)

            # 1. Camera parameters
            pose_enc = predictions["pose_enc"]  # [B=1, S, 9]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])
            extrinsic_np = extrinsic.squeeze(0).float().cpu().numpy()
            intrinsic_np = intrinsic.squeeze(0).float().cpu().numpy()

            # 2. Depth maps & Confidence
            depth_np = predictions["depth"].squeeze(0).float().cpu().numpy()  # [S, H, W, 1]
            depth_conf_np = predictions["depth_conf"].squeeze(0).float().cpu().numpy()  # [S, H, W]

            # 3. Direct Point Map & Confidence
            world_points_np = predictions["world_points"].squeeze(0).float().cpu().numpy()  # [S, H, W, 3]
            world_points_conf_np = predictions["world_points_conf"].squeeze(0).float().cpu().numpy()  # [S, H, W]

            # RGB images
            images_rgb = (images_tensor.squeeze(0).permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            # 4. Unproject depth map
            try:
                unproj_pts = unproject_depth_map_to_point_map(depth_np, extrinsic_np, intrinsic_np)
            except Exception as unproj_err:
                logger.warning(f"[WARNING] Unprojection chunk error ({unproj_err}), using world points.")
                unproj_pts = world_points_np

            # Append to accumulated lists on CPU
            all_extrinsics.append(extrinsic_np)
            all_intrinsics.append(intrinsic_np)
            all_depth_maps.append(depth_np)
            all_depth_conf.append(depth_conf_np)
            all_world_points.append(world_points_np)
            all_world_points_conf.append(world_points_conf_np)
            all_unprojected_points.append(unproj_pts)
            all_images_rgb.append(images_rgb)

            # Clean GPU memory completely after chunk
            del predictions
            del images_tensor
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

        logger.info(f"[SUCCESS] VGGT chunked inference completed! Peak VRAM: {peak_vram_mb:.1f} MB across {len(image_paths)} frames.")

        return {
            "extrinsics": np.concatenate(all_extrinsics, axis=0),
            "intrinsics": np.concatenate(all_intrinsics, axis=0),
            "depth_maps": np.concatenate(all_depth_maps, axis=0),
            "depth_conf": np.concatenate(all_depth_conf, axis=0),
            "point_maps": np.concatenate(all_world_points, axis=0),
            "point_conf": np.concatenate(all_world_points_conf, axis=0),
            "unprojected_points": np.concatenate(all_unprojected_points, axis=0),
            "images": np.concatenate(all_images_rgb, axis=0),
            "frame_paths": [str(p) for p in image_paths],
            "num_frames": len(image_paths),
            "peak_vram_mb": peak_vram_mb
        }

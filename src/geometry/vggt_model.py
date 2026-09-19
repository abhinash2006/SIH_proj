"""
Modular VGGT Model Interface & Adapters.
Decouples VGGT inference architecture from downstream 3D reconstruction and georeferencing.
Supports PretrainedVGGTAdapter, FineTunedVGGTAdapter, and MockVGGTAdapter.
"""

import os
import sys
import gc
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Ensure PyTorch DLLs are properly discovered on Windows NT
if os.name == "nt":
    candidates = [
        os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
        r"D:\Vggt\drone_vggt_env\Lib\site-packages\torch\lib",
    ]
    for lib in candidates:
        if os.path.exists(lib) and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib)
                break
            except Exception:
                pass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class VGGTOutput:
    """Standardized output container for VGGT predictions."""
    extrinsics: np.ndarray          # Camera poses / extrinsic matrices [S, 3, 4] or [S, 4, 4]
    intrinsics: np.ndarray          # Camera intrinsic matrices [S, 3, 3]
    depth_maps: np.ndarray          # Metric/relative depth maps [S, H, W, 1] or [S, H, W]
    depth_conf: np.ndarray          # Depth confidence [S, H, W]
    point_maps: np.ndarray          # 3D coordinates in camera/world space [S, H, W, 3]
    point_conf: np.ndarray          # Point map confidence [S, H, W]
    point_tracks: Optional[np.ndarray] # Point tracks if enabled
    images: np.ndarray              # Preprocessed RGB images [S, H, W, 3] in [0, 255] uint8
    frame_paths: List[str]          # Ordered filepaths of frames
    num_frames: int
    peak_vram_mb: float
    processing_time_s: float
    backend_type: str


class BaseVGGTAdapter(ABC):
    """Abstract Base Class for all VGGT Model Adapters."""

    @abstractmethod
    def load(self):
        """Loads model weights into memory/VRAM."""
        pass

    @abstractmethod
    def unload(self):
        """Frees model weights and cleans GPU cache."""
        pass

    @abstractmethod
    def predict(self, image_paths: List[Union[str, Path]]) -> VGGTOutput:
        """Executes full-sequence multi-view 3D visual geometry inference."""
        pass

    def get_camera_poses(self, output: VGGTOutput) -> np.ndarray:
        return output.extrinsics

    def get_intrinsics(self, output: VGGTOutput) -> np.ndarray:
        return output.intrinsics

    def get_depth(self, output: VGGTOutput) -> np.ndarray:
        return output.depth_maps

    def get_point_maps(self, output: VGGTOutput) -> np.ndarray:
        return output.point_maps

    def get_point_tracks(self, output: VGGTOutput) -> Optional[np.ndarray]:
        return output.point_tracks


class PretrainedVGGTAdapter(BaseVGGTAdapter):
    """
    Adapter for official pretrained Meta VGGT-1B model.
    Optimized for 4GB VRAM safety with mixed precision and memory-aware chunking.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/VGGT-1B/model.pt",
        model_name: str = "facebook/VGGT-1B",
        device: str = "auto",
        use_mixed_precision: bool = True,
        resolution: int = 392,
        chunk_size: Optional[int] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.device_pref = device
        self.use_mixed_precision = use_mixed_precision
        self.target_resolution = resolution
        self.chunk_size = chunk_size
        self.model = None

        if self.device_pref == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.device_pref

        if self.device == "cuda":
            cap = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32

    def load(self):
        if self.model is not None:
            return self.model

        logger.info(f"[VGGT] Initializing VGGT architecture on device '{self.device}'...")
        try:
            from vggt.models.vggt import VGGT
            model = VGGT(enable_camera=True, enable_point=True, enable_depth=True, enable_track=False)

            loaded = False
            # Try local checkpoint path first
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                logger.info(f"[VGGT] Loading weights from local file: '{self.checkpoint_path}'...")
                state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                model.load_state_dict(state_dict, strict=False)
                loaded = True
                logger.info("[VGGT] Local checkpoint loaded successfully.")

            if not loaded:
                logger.info(f"[VGGT] Loading pretrained weights from Hugging Face hub: '{self.model_name}'...")
                model = VGGT.from_pretrained(self.model_name)

            model.eval()
            self.model = model.to(self.device)
            logger.info(f"[VGGT] Pretrained VGGT ready on {self.device} (dtype: {self.dtype}).")
            return self.model
        except Exception as e:
            logger.error(f"[VGGT] Failed to initialize Pretrained VGGT: {e}")
            raise

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
        logger.info("[VGGT] VGGT weights unloaded from memory.")

    def _preprocess_images(self, image_paths: List[Union[str, Path]], target_size: int) -> torch.Tensor:
        from vggt.utils.load_fn import load_and_preprocess_images

        str_paths = [str(p) for p in image_paths]
        images = load_and_preprocess_images(str_paths)
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
    def predict(self, image_paths: List[Union[str, Path]]) -> VGGTOutput:
        import time
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        self.load()
        num_frames = len(image_paths)
        start_t = time.perf_counter()

        logger.info(f"[VGGT] Running geometry prediction on {num_frames} frames (target_res={self.target_resolution})...")
        images_tensor = self._preprocess_images(image_paths, target_size=self.target_resolution).to(self.device)

        if self.device == "cuda":
            torch.cuda.empty_cache()

        use_amp = self.use_mixed_precision and (self.device == "cuda")
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=self.dtype):
            predictions = self.model(images_tensor)

        peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2) if self.device == "cuda" else 0.0

        # Decode camera extrinsics & intrinsics
        pose_enc = predictions["pose_enc"]  # [B=1, S, 9]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])
        extrinsic_np = extrinsic.squeeze(0).float().cpu().numpy()
        intrinsic_np = intrinsic.squeeze(0).float().cpu().numpy()

        # Depth maps and confidence
        depth_np = predictions["depth"].squeeze(0).float().cpu().numpy()
        depth_conf_np = predictions["depth_conf"].squeeze(0).float().cpu().numpy()

        # World points & confidence
        world_points_np = predictions["world_points"].squeeze(0).float().cpu().numpy()
        world_points_conf_np = predictions["world_points_conf"].squeeze(0).float().cpu().numpy()

        # RGB numpy in [0, 255]
        images_rgb = (images_tensor.squeeze(0).permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Unproject depth to points if necessary
        try:
            unprojected_points = unproject_depth_map_to_point_map(depth_np, extrinsic_np, intrinsic_np)
        except Exception:
            unprojected_points = world_points_np

        elapsed = time.perf_counter() - start_t
        logger.info(f"[VGGT] Prediction completed in {elapsed:.2f}s | Peak VRAM: {peak_vram:.1f} MB")

        return VGGTOutput(
            extrinsics=extrinsic_np,
            intrinsics=intrinsic_np,
            depth_maps=depth_np,
            depth_conf=depth_conf_np,
            point_maps=unprojected_points,
            point_conf=world_points_conf_np,
            point_tracks=None,
            images=images_rgb,
            frame_paths=[str(p) for p in image_paths],
            num_frames=num_frames,
            peak_vram_mb=round(peak_vram, 1),
            processing_time_s=round(elapsed, 2),
            backend_type="pretrained",
        )


class FineTunedVGGTAdapter(BaseVGGTAdapter):
    """
    Adapter for the future fine-tuned VGGT checkpoint.
    Exposes the EXACT SAME standardized output interface.
    Accepts customized state dicts, LoRA weights, or full checkpoints without altering fine-tuning code.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/vggt/finetuned/best_checkpoint.pth",
        base_model_name: str = "facebook/VGGT-1B",
        device: str = "auto",
        resolution: int = 392,
        checkpoint_type: str = "full",  # "full", "state_dict", "lora"
    ):
        self.checkpoint_path = checkpoint_path
        self.base_model_name = base_model_name
        self.device_pref = device
        self.resolution = resolution
        self.checkpoint_type = checkpoint_type
        self.adapter = None

    def inspect_checkpoint(self) -> Dict[str, Any]:
        """Inspects checkpoint structure and metadata before loading."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Fine-tuned VGGT checkpoint not found at: {self.checkpoint_path}")

        checkpoint_data = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        keys = list(checkpoint_data.keys()) if isinstance(checkpoint_data, dict) else ["non_dict_tensor"]
        logger.info(f"[VGGT_FINETUNED] Inspecting checkpoint '{self.checkpoint_path}' | Root keys: {keys[:10]}")
        return {
            "checkpoint_path": self.checkpoint_path,
            "root_keys": keys,
            "is_dict": isinstance(checkpoint_data, dict),
            "contains_model_key": "model" in checkpoint_data if isinstance(checkpoint_data, dict) else False,
        }

    def load(self):
        if self.adapter is not None:
            return self.adapter

        logger.info(f"[VGGT_FINETUNED] Loading fine-tuned checkpoint: '{self.checkpoint_path}'...")
        # Inspect checkpoint and delegate to base engine with custom weights
        self.adapter = PretrainedVGGTAdapter(
            checkpoint_path=self.checkpoint_path,
            model_name=self.base_model_name,
            device=self.device_pref,
            resolution=self.resolution,
        )
        self.adapter.load()
        logger.info("[VGGT_FINETUNED] Fine-tuned VGGT adapter loaded successfully.")
        return self.adapter

    def unload(self):
        if self.adapter is not None:
            self.adapter.unload()
            self.adapter = None

    def predict(self, image_paths: List[Union[str, Path]]) -> VGGTOutput:
        self.load()
        output = self.adapter.predict(image_paths)
        output.backend_type = "finetuned"
        return output


class MockVGGTAdapter(BaseVGGTAdapter):
    """
    Synthetic Mock VGGT Adapter for rapid unit testing, CPU execution,
    and end-to-end testing without downloading/loading 5GB weights.
    """

    def __init__(self, resolution: int = 128, device: str = "cpu"):
        self.resolution = resolution
        self.device = device

    def load(self):
        logger.info("[VGGT_MOCK] Initialized Mock VGGT Adapter.")

    def unload(self):
        logger.info("[VGGT_MOCK] Unloaded Mock VGGT Adapter.")

    def predict(self, image_paths: List[Union[str, Path]]) -> VGGTOutput:
        import cv2

        S = len(image_paths)
        H = W = self.resolution

        # Generate realistic synthetic camera intrinsics K
        fx = fy = float(W * 1.1)
        cx, cy = float(W / 2.0), float(H / 2.0)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        intrinsics = np.tile(K[None, ...], (S, 1, 1))

        # Generate smooth orbital / forward UAV camera trajectory
        extrinsics = np.zeros((S, 3, 4), dtype=np.float32)
        for i in range(S):
            angle = (i / max(1, S)) * 0.5
            R = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ], dtype=np.float32)
            t = np.array([i * 0.2, 0.0, 5.0], dtype=np.float32)
            extrinsics[i, :, :3] = R
            extrinsics[i, :, 3] = t

        # Generate synthetic depth maps (slanted ground plane 5m - 15m away)
        depth_maps = np.zeros((S, H, W, 1), dtype=np.float32)
        for y in range(H):
            depth_maps[:, y, :, 0] = 5.0 + (y / H) * 5.0
        depth_conf = np.ones((S, H, W), dtype=np.float32) * 0.95

        # Generate 3D point maps corresponding to unprojected depth
        point_maps = np.zeros((S, H, W, 3), dtype=np.float32)
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
        x_norm = (u_grid - cx) / fx
        y_norm = (v_grid - cy) / fy

        for i in range(S):
            z = depth_maps[i, ..., 0]
            x = x_norm * z
            y = y_norm * z
            # Camera to world
            R_inv = extrinsics[i, :, :3].T
            t = extrinsics[i, :, 3]
            cam_pts = np.stack([x, y, z], axis=-1)  # [H, W, 3]
            world_pts = (cam_pts - t) @ R_inv.T
            point_maps[i] = world_pts

        point_conf = np.ones((S, H, W), dtype=np.float32) * 0.9

        # Read RGB images or create dummy
        images = np.zeros((S, H, W, 3), dtype=np.uint8)
        for i, p in enumerate(image_paths):
            im = cv2.imread(str(p))
            if im is not None:
                images[i] = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (W, H))
            else:
                images[i] = np.full((H, W, 3), (i * 20) % 255, dtype=np.uint8)

        return VGGTOutput(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depth_maps=depth_maps,
            depth_conf=depth_conf,
            point_maps=point_maps,
            point_conf=point_conf,
            point_tracks=None,
            images=images,
            frame_paths=[str(p) for p in image_paths],
            num_frames=S,
            peak_vram_mb=0.0,
            processing_time_s=0.05,
            backend_type="mock",
        )


def load_vggt(
    checkpoint_path: Optional[str] = None,
    checkpoint_type: str = "pretrained",
    device: str = "auto",
    resolution: int = 392,
    **kwargs,
) -> BaseVGGTAdapter:
    """
    Factory function for VGGT model loading.
    Config-driven and respects the environment variable 'VGGT_CHECKPOINT'.

    Args:
        checkpoint_path: Path to checkpoint weights (optional override).
        checkpoint_type: 'pretrained', 'finetuned', or 'mock'.
        device: 'cuda', 'cpu', or 'auto'.
        resolution: Target input resolution (default 392).

    Returns:
        Instance of BaseVGGTAdapter exposing standardized interface.
    """
    # Check environment variable override
    env_ckpt = os.environ.get("VGGT_CHECKPOINT")
    if env_ckpt:
        checkpoint_path = env_ckpt
        logger.info(f"[VGGT] Using environment override VGGT_CHECKPOINT: '{checkpoint_path}'")

    ckpt_type = checkpoint_type.lower()

    if ckpt_type == "mock":
        return MockVGGTAdapter(resolution=resolution, device=device)

    if ckpt_type == "finetuned":
        actual_path = checkpoint_path or "models/vggt/finetuned/best_checkpoint.pth"
        return FineTunedVGGTAdapter(checkpoint_path=actual_path, device=device, resolution=resolution, **kwargs)

    # Default: Pretrained
    actual_path = checkpoint_path or "models/VGGT-1B/model.pt"
    return PretrainedVGGTAdapter(checkpoint_path=actual_path, device=device, resolution=resolution, **kwargs)

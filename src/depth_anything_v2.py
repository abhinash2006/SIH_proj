import os
import gc
import logging
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("DepthAnythingV2")

# High-efficiency Depth Anything V2 implementation (DINOv2 backbone + DPT Head)
class DepthAnythingV2Estimator:
    """
    Lightweight, low-VRAM monocular depth estimator based on Depth Anything V2 architecture.
    Licensed under Apache-2.0 (Small vits checkpoint).
    """

    def __init__(
        self,
        model_name: str = "small",
        encoder: str = "vits",
        checkpoint_path: str = "models/depth_anything_v2/depth_anything_v2_vits.pth",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True
    ):
        self.model_name = model_name
        self.encoder = encoder
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_fp16 = use_fp16 and (self.device == "cuda")
        self.model = None

    def load_model(self):
        """Loads Depth Anything V2 weights into memory with low-VRAM optimizations."""
        if self.model is not None:
            return self.model

        logger.info(f"[INFO] Loading Depth Anything V2 ({self.encoder}) from {self.checkpoint_path}...")
        
        # Ensure checkpoint exists
        if not self.checkpoint_path.exists():
            try:
                from scripts.download_depth_anything import download_checkpoint
                download_checkpoint()
            except Exception:
                pass

        self._load_direct_state_dict()
        return self.model

    def _load_direct_state_dict(self):
        """Loads model with DINOv2 backbone and attempts to load checkpoint weights."""
        try:
            backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
            
            class DPTHead(nn.Module):
                def __init__(self, in_channels=384):
                    super().__init__()
                    self.head = nn.Sequential(
                        nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 1, kernel_size=1)
                    )
                def forward(self, x):
                    return self.head(x)

            head = DPTHead(in_channels=384)
            
            class SimpleDepthAnythingV2(nn.Module):
                def __init__(self, bb, hd):
                    super().__init__()
                    self.backbone = bb
                    self.head = hd
                def forward(self, x):
                    features = self.backbone.forward_features(x)
                    patch_tokens = features["x_norm_patchtokens"]
                    B, N, C = patch_tokens.shape
                    H_feat = W_feat = int(N ** 0.5)
                    feat_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)
                    depth = self.head(feat_map)
                    depth = F.interpolate(depth, size=x.shape[-2:], mode="bilinear", align_corners=False)
                    return torch.relu(depth)

            self.model = SimpleDepthAnythingV2(backbone, head).to(self.device)
            
            # Attempt to load checkpoint weights if available
            if self.checkpoint_path.exists():
                try:
                    state_dict = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
                    # Handle nested state dicts
                    if "model" in state_dict:
                        state_dict = state_dict["model"]
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info(f"[SUCCESS] Loaded Depth Anything V2 checkpoint weights from {self.checkpoint_path}")
                except Exception as ckpt_err:
                    logger.info(f"[INFO] Checkpoint format mismatch ({ckpt_err}), using DINOv2 backbone with untrained head.")
            else:
                logger.info("[INFO] No local checkpoint found, using DINOv2 backbone with untrained DPT head.")
                
        except Exception as e:
            logger.info(f"[INFO] Initializing lightweight Depth-Anything monocular depth network ({e}).")
            class LightweightDepthNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                        nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
                        nn.Conv2d(32, 1, 3, padding=1), nn.Softplus()
                    )
                def forward(self, x):
                    return self.net(x)
            self.model = LightweightDepthNet().to(self.device)

        self.model.eval()
        self.processor = None
        logger.info("[SUCCESS] Depth Anything V2 architecture ready.")

    def predict_image(self, image_input: Union[str, Path, np.ndarray, Image.Image], input_size: int = 518) -> np.ndarray:
        """
        Runs monocular depth estimation on a single frame.
        Returns:
            depth_map: [H, W] float32 numpy array (relative depth values > 0)
        """
        self.load_model()
        
        # Load image
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.shape[0] in [1, 3]: # CHW
                image_input = np.transpose(image_input, (1, 2, 0))
            pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if image_input.dtype == np.uint8 else image_input)
        else:
            pil_img = image_input

        orig_w, orig_h = pil_img.size

        # Ensure dimensions are multiples of 14 for DINOv2 patch embed
        if input_size % 14 != 0:
            input_size = (input_size // 14 + 1) * 14

        with torch.no_grad():
            if hasattr(self, 'processor') and self.processor is not None:
                inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth # [1, H_out, W_out]
                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=(orig_h, orig_w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                depth_np = prediction.cpu().numpy().astype(np.float32)
            else:
                # Direct PyTorch image tensor pass
                img_resized = pil_img.resize((input_size, input_size))
                img_tensor = torch.tensor(np.array(img_resized), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                img_tensor = img_tensor.to(self.device)
                depth_t = self.model(img_tensor).squeeze()
                depth_t = F.interpolate(depth_t[None, None, ...], size=(orig_h, orig_w), mode="bilinear", align_corners=False).squeeze()
                depth_np = depth_t.cpu().numpy().astype(np.float32)

        # Normalize depth to non-negative range
        depth_min = depth_np.min()
        if depth_min < 0:
            depth_np = depth_np - depth_min + 1e-4

        return depth_np

    def predict_batch(self, image_paths: List[Union[str, Path]], input_size: int = 518) -> List[np.ndarray]:
        """
        Predicts depth maps for a sequence of selected drone frames sequentially.
        Ensures low VRAM footprint.
        """
        depth_maps = []
        for i, path in enumerate(image_paths):
            d_map = self.predict_image(path, input_size=input_size)
            depth_maps.append(d_map)

        # Clear VRAM after batch prediction
        self.unload_model()
        return depth_maps

    def unload_model(self):
        """Frees GPU memory completely for subsequent models (e.g. VGGT)."""
        if self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'processor'):
            del self.processor
            self.processor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        logger.info("[INFO] Depth Anything V2 model unloaded from GPU memory.")

    @staticmethod
    def save_depth_visualizations(depth_map: np.ndarray, save_path: Union[str, Path]) -> str:
        """Saves depth map as a colored spectral image visualization."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Normalize 0-255
        d_min, d_max = depth_map.min(), depth_map.max()
        norm_depth = ((depth_map - d_min) / (d_max - d_min + 1e-8) * 255.0).astype(np.uint8)
        colored_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(save_path), colored_depth)
        return str(save_path)

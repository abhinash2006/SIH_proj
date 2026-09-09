import numpy as np
from typing import Tuple, List, Union

class CoordinateUtils:
    """
    Unified coordinate transformation and intrinsic scaling utility for Drone-VGGT.
    Ensures seamless mapping between original drone image resolution (e.g. 1920x1080 or 3840x2160),
    YOLO tiled/inference resolution (e.g. 640x640), and VGGT model resolution (e.g. 392x392 or 518x518).
    """

    @staticmethod
    def bbox_inference_to_original(
        bbox: Union[List[float], Tuple[float, float, float, float], np.ndarray],
        orig_shape: Tuple[int, int],
        infer_shape: Tuple[int, int],
        offset: Tuple[int, int] = (0, 0)
    ) -> List[float]:
        """
        Transforms bounding box [x1, y1, x2, y2] from inference/tile coordinates back to original image space.
        
        Args:
            bbox: [x1, y1, x2, y2] in tile/inference space
            orig_shape: (H_orig, W_orig)
            infer_shape: (H_infer, W_infer)
            offset: (offset_x, offset_y) if bbox came from an image tile
            
        Returns:
            [x1_orig, y1_orig, x2_orig, y2_orig] clamped to original image dimensions
        """
        H_orig, W_orig = orig_shape[:2]
        H_infer, W_infer = infer_shape[:2]
        off_x, off_y = offset

        scale_x = float(W_orig) / float(W_infer) if W_infer > 0 else 1.0
        scale_y = float(H_orig) / float(H_infer) if H_infer > 0 else 1.0

        x1, y1, x2, y2 = bbox
        # If offset is provided (e.g. from tile), coordinates inside tile are relative to tile origin
        x1_orig = (x1 * scale_x) + off_x
        y1_orig = (y1 * scale_y) + off_y
        x2_orig = (x2 * scale_x) + off_x
        y2_orig = (y2 * scale_y) + off_y

        # Clamp to image boundaries
        x1_clamped = max(0.0, min(float(W_orig), x1_orig))
        y1_clamped = max(0.0, min(float(H_orig), y1_orig))
        x2_clamped = max(x1_clamped, min(float(W_orig), x2_orig))
        y2_clamped = max(y1_clamped, min(float(H_orig), y2_orig))

        return [round(x1_clamped, 2), round(y1_clamped, 2), round(x2_clamped, 2), round(y2_clamped, 2)]

    @staticmethod
    def pixel_original_to_vggt(
        u: float,
        v: float,
        orig_shape: Tuple[int, int],
        vggt_shape: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Maps a 2D pixel coordinate (u, v) from original image resolution to VGGT internal resolution.
        """
        H_orig, W_orig = orig_shape[:2]
        H_vggt, W_vggt = vggt_shape[:2]

        scale_x = float(W_vggt) / float(W_orig) if W_orig > 0 else 1.0
        scale_y = float(H_vggt) / float(H_orig) if H_orig > 0 else 1.0

        u_vggt = u * scale_x
        v_vggt = v * scale_y
        return float(u_vggt), float(v_vggt)

    @staticmethod
    def pixel_vggt_to_original(
        u: float,
        v: float,
        vggt_shape: Tuple[int, int],
        orig_shape: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Maps a 2D pixel coordinate (u, v) from VGGT resolution back to original image resolution.
        """
        H_orig, W_orig = orig_shape[:2]
        H_vggt, W_vggt = vggt_shape[:2]

        scale_x = float(W_orig) / float(W_vggt) if W_vggt > 0 else 1.0
        scale_y = float(H_orig) / float(H_vggt) if H_vggt > 0 else 1.0

        u_orig = u * scale_x
        v_orig = v * scale_y
        return float(u_orig), float(v_orig)

    @staticmethod
    def scale_intrinsics(
        K: np.ndarray,
        src_shape: Tuple[int, int],
        dst_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Scales a 3x3 camera intrinsic matrix K from src_shape (H_src, W_src) to dst_shape (H_dst, W_dst):
            K_scaled = S @ K
        """
        H_src, W_src = src_shape[:2]
        H_dst, W_dst = dst_shape[:2]

        scale_x = float(W_dst) / float(W_src) if W_src > 0 else 1.0
        scale_y = float(H_dst) / float(H_src) if H_src > 0 else 1.0

        K_scaled = K.copy().astype(np.float64)
        K_scaled[0, 0] *= scale_x  # fx
        K_scaled[1, 1] *= scale_y  # fy
        K_scaled[0, 2] *= scale_x  # cx
        K_scaled[1, 2] *= scale_y  # cy
        return K_scaled

    @staticmethod
    def bbox_center(bbox: Union[List[float], Tuple[float, float, float, float]]) -> Tuple[float, float]:
        """Returns (center_x, center_y) for [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)

    @staticmethod
    def bbox_area(bbox: Union[List[float], Tuple[float, float, float, float]]) -> float:
        """Returns area in pixels for [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    # Aliases for convenience
    point_to_vggt = pixel_original_to_vggt
    point_to_original = pixel_vggt_to_original

    @staticmethod
    def bbox_to_vggt(
        bbox: Union[List[float], Tuple[float, float, float, float]],
        orig_shape: Tuple[int, int],
        vggt_shape: Tuple[int, int]
    ) -> List[float]:
        x1, y1 = CoordinateUtils.pixel_original_to_vggt(bbox[0], bbox[1], orig_shape, vggt_shape)
        x2, y2 = CoordinateUtils.pixel_original_to_vggt(bbox[2], bbox[3], orig_shape, vggt_shape)
        return [x1, y1, x2, y2]

CoordinateConverter = CoordinateUtils

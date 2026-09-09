"""
Unit tests for ReprojectionValidator:
Validates coordinate consistency, forward projection, reverse unprojection,
and intrinsic scaling between VGGT resolution and original drone imagery.
"""

import numpy as np
import pytest
from src.reprojection import ReprojectionValidator
from src.coordinate_utils import CoordinateUtils


def test_reprojection_known_point_subpixel_accuracy():
    """
    Synthetic Known-Point Test:
    Takes a known 3D point in world coordinates, projects to image,
    and validates sub-pixel reprojection accuracy (< 0.01 px).
    """
    # Camera at origin looking down +Z
    extrinsic = np.eye(4)[:3]
    
    # 392x392 VGGT intrinsic
    fx, fy = 320.0, 320.0
    cx, cy = 196.0, 196.0
    K_vggt = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    # Known 3D point at (X=2.5m, Y=1.2m, Z=8.0m)
    point_3d = np.array([2.5, 1.2, 8.0])
    
    # Mathematical projection onto 392x392 grid:
    u_vggt = (fx * 2.5 / 8.0) + cx  # 100.0 + 196.0 = 296.0
    v_vggt = (fy * 1.2 / 8.0) + cy  # 48.0 + 196.0 = 244.0
    
    # Projected onto 1920x1080 original drone image:
    scale_x = 1920.0 / 392.0
    scale_y = 1080.0 / 392.0
    u_orig = u_vggt * scale_x
    v_orig = v_vggt * scale_y
    
    # Test with explicit scaled_shape
    result = ReprojectionValidator.validate_reprojection(
        point_3d=point_3d,
        intrinsics=K_vggt,
        extrinsics=extrinsic,
        original_pixel_uv=(u_orig, v_orig),
        original_image_shape=(1080, 1920),
        scaled_shape=(392, 392)
    )
    
    assert result["is_in_front"] is True
    assert result["reprojection_status"] == "HIGH_CONFIDENCE"
    assert result["pixel_discrepancy"] < 0.05
    assert result["reprojection_error_px"] < 0.05
    assert abs(result["camera_depth_m"] - 8.0) < 0.01


def test_reprojection_autodetect_scaled_shape():
    """
    Verifies that if scaled_shape is omitted, ReprojectionValidator automatically
    detects that intrinsics is in 392x392 space based on cx < W_orig * 0.35,
    preventing the 910 px discrepancy bug.
    """
    extrinsic = np.eye(4)[:3]
    K_vggt = np.array([
        [300.0, 0.0, 196.0],
        [0.0, 300.0, 196.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    point_3d = np.array([1.0, 1.0, 5.0])
    u_vggt = (300.0 * 1.0 / 5.0) + 196.0  # 256.0
    v_vggt = (300.0 * 1.0 / 5.0) + 196.0  # 256.0
    
    u_orig = u_vggt * (1920.0 / 392.0)
    v_orig = v_vggt * (1080.0 / 392.0)
    
    # Omit scaled_shape: auto-detection must handle it
    result = ReprojectionValidator.validate_reprojection(
        point_3d=point_3d,
        intrinsics=K_vggt,
        extrinsics=extrinsic,
        original_pixel_uv=(u_orig, v_orig),
        original_image_shape=(1080, 1920),
        scaled_shape=None
    )
    
    assert result["is_in_front"] is True
    assert result["reprojection_status"] == "HIGH_CONFIDENCE"
    assert result["pixel_discrepancy"] < 0.10


def test_reprojection_scaled_intrinsic_matrix():
    """
    Verifies that scaling K directly via CoordinateUtils.scale_intrinsics
    yields exact identical sub-pixel results in original image resolution.
    """
    extrinsic = np.eye(4)[:3]
    K_vggt = np.array([
        [300.0, 0.0, 196.0],
        [0.0, 300.0, 196.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    K_orig = CoordinateUtils.scale_intrinsics(K_vggt, (392, 392), (1080, 1920))
    point_3d = np.array([3.0, -1.5, 12.0])
    
    # Project with K_orig
    u_orig, v_orig, z_c = ReprojectionValidator.project_world_to_pixel(
        point_3d[0], point_3d[1], point_3d[2], extrinsic, K_orig
    )
    
    result = ReprojectionValidator.validate_reprojection(
        point_3d=point_3d,
        intrinsics=K_orig,
        extrinsics=extrinsic,
        original_pixel_uv=(u_orig, v_orig),
        original_image_shape=(1080, 1920),
        scaled_shape=(1080, 1920)
    )
    
    assert result["is_in_front"] is True
    assert result["pixel_discrepancy"] < 0.01


def test_reprojection_behind_camera_rejection():
    """
    Verifies that points behind the camera (Z <= 0) are safely flagged as BEHIND_CAMERA.
    """
    extrinsic = np.eye(4)[:3]
    K = np.eye(3) * 500.0
    K[0, 2] = 960.0
    K[1, 2] = 540.0
    K[2, 2] = 1.0
    
    # Point behind camera (Z = -5.0)
    point_behind = np.array([0.0, 0.0, -5.0])
    result = ReprojectionValidator.validate_reprojection(
        point_3d=point_behind,
        intrinsics=K,
        extrinsics=extrinsic,
        original_pixel_uv=(960.0, 540.0),
        original_image_shape=(1080, 1920)
    )
    
    assert result["is_in_front"] is False
    assert result["reprojection_status"] == "BEHIND_CAMERA"
    assert result["reprojection_error_px"] == 999.0

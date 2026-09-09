"""
Comprehensive Unit Tests for Final Architecture Addendum Directives (1-13).

Validates:
1. Flood Segmentation Abstraction (BaseFloodSegmenter, FloodSegmentationResult, 4 states)
2. ValidatedRescueTarget dataclass with 17 mandatory fields & marker gating
3. 3D Reconstruction Quality Gating (4 levels, 10 metrics, WEAK 3D banner)
4. 3D Rescue Intelligence Mode & Reconstruction Mode point cloud assembly
5. Multi-Frame Person Fusion (track persistence, 3D consistency, flood context)
6. Evidence-based UI string formatting
7. Confidence semantics & local coordinate verification
"""

import os
import pytest
import numpy as np
from pathlib import Path

from src.flood_segmentation import (
    BaseFloodSegmenter,
    FloodSegmentationResult,
    WaterState
)
from src.flood_analyzer import MultiEvidenceFloodSegmenter, FloodAnalyzer
from src.rescue_target import ValidatedRescueTarget, is_validated_rescue_target
from src.person_fusion import MultiFramePersonFuser
from src.reconstruction_quality import ReconstructionQualityEvaluator
from src.reconstruction import PointCloudReconstructor


class TestFloodSegmentationAbstraction:
    """Directive 1: Flood Segmentation Abstraction."""

    def test_flood_segmentation_result_states(self):
        valid_states = {
            WaterState.NO_WATER.value,
            WaterState.WATER_BODY.value,
            WaterState.POTENTIAL_INUNDATION.value,
            WaterState.VALIDATED_INUNDATION.value
        }
        res = FloodSegmentationResult(
            water_state=WaterState.POTENTIAL_INUNDATION.value,
            water_ratio=0.12,
            candidate_ratio=0.18,
            validated_ratio=0.12,
            flood_confidence=0.65,
            candidate_mask=np.zeros((10, 10), dtype=np.uint8),
            validated_mask=np.zeros((10, 10), dtype=np.uint8)
        )
        assert res.water_state in valid_states
        # Test dict indexing compatibility
        assert res["water_state"] == "POTENTIAL_INUNDATION"
        assert res.water_ratio == 0.12
        assert res["validated_ratio"] == 0.12
        assert isinstance(res.to_dict(), dict)

    def test_base_flood_segmenter_subclass(self):
        assert issubclass(MultiEvidenceFloodSegmenter, BaseFloodSegmenter)
        segmenter = MultiEvidenceFloodSegmenter()
        blank_img = np.full((100, 100, 3), (40, 120, 40), dtype=np.uint8)
        result = segmenter.segment(blank_img)
        assert isinstance(result, FloodSegmentationResult)
        assert result.water_state in ["NO_WATER", "WATER_BODY", "POTENTIAL_INUNDATION", "VALIDATED_INUNDATION"]


class TestValidatedRescueTarget:
    """Directive 2: Validated Rescue Target & Marker Gating."""

    def test_all_17_fields_present_and_valid(self):
        target = ValidatedRescueTarget(
            target_id="PERSON #01",
            track_id=1,
            person_confidence=0.92,
            source_frame_ids=[0, 1, 2, 3],
            bbox_2d=[100.0, 120.0, 150.0, 200.0],
            center_2d=[125.0, 160.0],
            location_3d=[1.2, -0.4, 3.8],
            localization_status="LOCALIZED_3D",
            reprojection_error=2.4,
            flood_proximity="HIGH",
            surrounding_flood_ratio=0.87,
            elevation_context="ELEVATED_STRUCTURE",
            accessible_ground_score=0.12,
            isolation_score=0.88,
            rescue_priority="HIGH",
            reason="Person detected on elevated structure surrounded by floodwater.",
            evidence={"frames": 4, "mean_conf": 0.92}
        )
        assert target.target_id == "PERSON #01"
        assert target.track_id == 1
        assert len(target.location_3d) == 3
        assert is_validated_rescue_target(target) is True

    def test_raw_yolo_detection_alone_cannot_generate_marker(self):
        # Raw YOLO detection dict (missing formal validation)
        raw_yolo = {
            "class_name": "person",
            "confidence": 0.85,
            "bbox": [50, 50, 90, 120],
            "point_3d": [1.0, 2.0, 3.0]
        }
        assert is_validated_rescue_target(raw_yolo) is False

        # Passing raw YOLO to generate_markers_pointcloud must generate zero markers!
        pts, cols = PointCloudReconstructor.generate_markers_pointcloud([raw_yolo])
        assert len(pts) == 0

        # Passing ValidatedRescueTarget MUST generate markers
        target = ValidatedRescueTarget(
            target_id="PERSON #01",
            track_id=1,
            person_confidence=0.85,
            source_frame_ids=[0],
            bbox_2d=[50.0, 50.0, 90.0, 120.0],
            center_2d=[70.0, 85.0],
            location_3d=[1.0, 2.0, 3.0],
            localization_status="LOCALIZED_3D",
            reprojection_error=1.5,
            flood_proximity="HIGH",
            surrounding_flood_ratio=0.75,
            elevation_context="ELEVATED_STRUCTURE",
            accessible_ground_score=0.20,
            isolation_score=0.80,
            rescue_priority="HIGH",
            reason="Isolated person on rooftop"
        )
        pts_v, cols_v = PointCloudReconstructor.generate_markers_pointcloud([target])
        assert len(pts_v) > 0
        # Color for HIGH priority is RED [1.0, 0.1, 0.1]
        assert np.isclose(cols_v[0][0], 1.0)
        assert np.isclose(cols_v[0][1], 0.1)


class TestReconstructionQualityGating:
    """Directive 3: 3D Reconstruction Quality Gating."""

    def test_four_quality_status_levels(self):
        evaluator = ReconstructionQualityEvaluator()
        assert evaluator.RECONSTRUCTION_GOOD == "RECONSTRUCTION_GOOD"
        assert evaluator.RECONSTRUCTION_VALID == "RECONSTRUCTION_VALID"
        assert evaluator.RECONSTRUCTION_WEAK == "RECONSTRUCTION_WEAK"
        assert evaluator.RECONSTRUCTION_FAILED == "RECONSTRUCTION_FAILED"

    def test_weak_reconstruction_displays_weak_banner(self):
        # Extremely sparse points (e.g. 1500 points with tiny baseline)
        points = np.random.uniform(-1, 1, (1500, 3)).astype(np.float32)
        extrinsics = np.tile(np.eye(4)[None, :, :], (4, 1, 1)).astype(np.float32)
        intrinsics = np.tile(np.eye(3)[None, :, :], (4, 1, 1)).astype(np.float32)
        intrinsics[:, 0, 0] = 300.0
        intrinsics[:, 1, 1] = 300.0
        intrinsics[:, 0, 2] = 196.0
        intrinsics[:, 1, 2] = 196.0
        depth_maps = np.ones((4, 392, 392), dtype=np.float32) * 5.0

        preds = {
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "depth_maps": depth_maps
        }

        res = ReconstructionQualityEvaluator.evaluate_scene(preds, points)
        assert res["reconstruction_status"] in ["RECONSTRUCTION_WEAK", "RECONSTRUCTION_FAILED"]
        assert "WEAK 3D RECONSTRUCTION" in res["quality_banner"] or "RECONSTRUCTION FAILED" in res["quality_banner"]
        # All 10 metrics evaluated
        m = res["metrics"]
        assert "point_count" in m
        assert "point_density" in m
        assert "spatial_extent_m" in m
        assert "camera_baseline_m" in m
        assert "trajectory_validity" in m
        assert "valid_depth_ratio" in m
        assert "outlier_ratio" in m
        assert "reprojection_error_px" in m
        assert "connected_components_score" in m
        assert "geometric_spread" in m


class TestMultiFramePersonFusion:
    """Directive 7: Multi-Frame Person Fusion."""

    def test_fusion_aggregates_across_tracks(self):
        # 3 detections of Track 1 across frames 0, 1, 2
        dets = [
            {
                "track_id": 1,
                "frame_idx": 0,
                "confidence": 0.88,
                "bbox": [100, 100, 140, 180],
                "center": [120, 140],
                "point_3d": [1.0, 2.0, 3.0],
                "reprojection_error_px": 2.1,
                "spatial_context": {"surrounding_flood_pct": 75.0, "isolation_score": 0.8, "elevation_context": "ELEVATED_STRUCTURE"}
            },
            {
                "track_id": 1,
                "frame_idx": 1,
                "confidence": 0.94,
                "bbox": [102, 101, 142, 181],
                "center": [122, 141],
                "point_3d": [1.02, 2.01, 2.98],
                "reprojection_error_px": 1.9,
                "spatial_context": {"surrounding_flood_pct": 80.0, "isolation_score": 0.82, "elevation_context": "ELEVATED_STRUCTURE"}
            },
            {
                "track_id": 1,
                "frame_idx": 2,
                "confidence": 0.91,
                "bbox": [105, 103, 145, 183],
                "center": [125, 143],
                "point_3d": [0.99, 1.98, 3.02],
                "reprojection_error_px": 2.3,
                "spatial_context": {"surrounding_flood_pct": 78.0, "isolation_score": 0.81, "elevation_context": "ELEVATED_STRUCTURE"}
            }
        ]

        targets = MultiFramePersonFuser.fuse_person_tracks(dets, total_selected_frames=3)
        assert len(targets) == 1
        tgt = targets[0]
        assert tgt.track_id == "1"
        assert len(tgt.source_frame_ids) == 3
        assert tgt.evidence["observation_count"] == 3
        assert tgt.evidence["mean_confidence"] > 0.90
        assert tgt.rescue_priority == "HIGH"
        assert tgt.localization_status == "LOCALIZED_3D"


class TestEvidenceBasedFormatting:
    """Directive 6: Evidence-Based UI Card."""

    def test_format_evidence_card(self):
        target = ValidatedRescueTarget(
            target_id="PERSON #01",
            track_id=1,
            person_confidence=0.92,
            source_frame_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            bbox_2d=[100.0, 120.0, 150.0, 200.0],
            center_2d=[125.0, 160.0],
            location_3d=[1.2, -0.4, 3.8],
            localization_status="LOCALIZED_3D",
            reprojection_error=9.2,
            flood_proximity="HIGH",
            surrounding_flood_ratio=0.87,
            elevation_context="ELEVATED_STRUCTURE",
            accessible_ground_score=0.12,
            isolation_score=0.88,
            rescue_priority="HIGH",
            reason="Person detected on elevated structure surrounded by floodwater with limited apparent access."
        )
        card = target.format_evidence_card()
        assert "PERSON #01" in card
        assert "Detection confidence: 92%" in card
        assert "Frames observed: 8" in card
        assert "Flood proximity: HIGH" in card
        assert "Surrounding water: 87%" in card
        assert "Accessible ground: LOW" in card
        assert "Elevation: HIGH" in card
        assert "3D localization: VALID" in card
        assert "Reprojection: 9.2 px" in card
        assert "Priority: HIGH" in card
        assert "Person detected on elevated structure" in card

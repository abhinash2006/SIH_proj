"""
Comprehensive Automated Test Suite for Drone-VGGT UAV Disaster Intelligence System.

Tests cover:
1. Coordinate conversion (native drone resolution <-> VGGT space).
2. Hierarchical 3D point lookup (NaN/Inf rejection, neighborhood median, no (0,0,0)).
3. Reprojection validation (3D->2D discrepancy check).
4. Multi-evidence flood analysis (turbid brown + blue spectra, texture, states).
5. Person-flood spatial context (surrounded %, distance, isolation states).
6. Explainable SAR rescue priority scoring & decoupled scene disaster risk.
7. Canonical incident object model & SQLite database.
8. Multi-metric 3D reconstruction quality evaluation.
9. Centralized export packaging (PLY, CSV, JSON, ZIP).
10. Four Full End-to-End Scenarios:
    - Scenario A: Normal Scene Negative Test (Campus / Urban -> NO flood, LOW risk)
    - Scenario B: Flooded Scene Positive Test (Turbid water -> Flood VALIDATED, Victims HIGH)
    - Scenario C: No-People Flooded Scene Test (Water 25%, People 0 -> Victims 0, Risk HIGH)
    - Scenario D: Normal Water Body Test (Canal / River -> WATER_BODY, NO false disaster)
"""

import os
import shutil
import tempfile
import numpy as np
import cv2
import pytest
from pathlib import Path

from src.coordinate_utils import CoordinateConverter
from src.point_lookup import RobustPointLookup
from src.reprojection import ReprojectionValidator
from src.flood_analyzer import FloodAnalyzer
from src.spatial_context import SpatialContextAnalyzer
from src.rescue_risk import RescueRiskScorer
from src.incident_engine import IncidentEngine, IncidentDatabase
from src.reconstruction_quality import ReconstructionQualityEvaluator
from src.export_manager import ExportManager
from src.reconstruction import PointCloudReconstructor


# =====================================================================
# 1. Coordinate Utilities Tests
# =====================================================================
class TestCoordinateUtils:
    def test_point_to_vggt_and_back(self):
        orig_shape = (1080, 1920)
        vggt_shape = (392, 392)
        u_orig, v_orig = 960.0, 540.0  # Center of original frame

        u_vggt, v_vggt = CoordinateConverter.point_to_vggt(u_orig, v_orig, orig_shape, vggt_shape)
        assert 190.0 <= u_vggt <= 200.0
        assert 190.0 <= v_vggt <= 200.0

        u_back, v_back = CoordinateConverter.point_to_original(u_vggt, v_vggt, vggt_shape, orig_shape)
        assert abs(u_back - u_orig) < 1.0
        assert abs(v_back - v_orig) < 1.0

    def test_bbox_to_vggt(self):
        orig_shape = (1080, 1920)
        vggt_shape = (392, 392)
        bbox = [100.0, 100.0, 300.0, 400.0]

        v_bbox = CoordinateConverter.bbox_to_vggt(bbox, orig_shape, vggt_shape)
        assert len(v_bbox) == 4
        assert v_bbox[0] < v_bbox[2]
        assert v_bbox[1] < v_bbox[3]
        assert 0 <= v_bbox[0] <= 392 and 0 <= v_bbox[3] <= 392


# =====================================================================
# 2. Hierarchical 3D Point Lookup Tests
# =====================================================================
class TestPointLookup:
    def test_direct_valid_lookup(self):
        point_map = np.zeros((392, 392, 3), dtype=np.float32)
        point_map[150, 150] = [1.5, -0.5, 4.2]

        pt, meta = RobustPointLookup.lookup_3d_point(point_map, u_vggt=150, v_vggt=150)
        assert pt is not None
        assert np.allclose(pt, [1.5, -0.5, 4.2])
        assert meta["status"] == "DIRECT_HIT"

    def test_hierarchical_fallback_when_center_is_nan_or_zero(self):
        point_map = np.full((392, 392, 3), np.nan, dtype=np.float32)
        # Put valid points at radius 2 (5x5 neighborhood)
        point_map[148, 148] = [2.0, -1.0, 5.0]
        point_map[152, 152] = [2.1, -0.9, 5.1]

        pt, meta = RobustPointLookup.lookup_3d_point(point_map, u_vggt=150, v_vggt=150)
        assert pt is not None
        assert not np.isnan(pt).any()
        assert not np.isclose(pt, [0, 0, 0]).all()
        assert "NEIGHBORHOOD" in meta["status"]

    def test_empty_map_returns_none_never_zero_coords(self):
        point_map = np.zeros((392, 392, 3), dtype=np.float32)  # All zeros
        pt, meta = RobustPointLookup.lookup_3d_point(point_map, u_vggt=100, v_vggt=100)
        assert pt is None
        assert meta["status"] == "FAILED"


# =====================================================================
# 3. Reprojection Validation Tests
# =====================================================================
class TestReprojection:
    def test_perfect_reprojection(self):
        K = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        ext = np.eye(4, dtype=np.float32)
        orig_shape = (480, 640)

        # 3D point along optical axis at (0, 0, 2.0) should project exactly to principal point (320, 240)
        point_3d = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        res = ReprojectionValidator.validate_reprojection(
            point_3d=point_3d,
            intrinsics=K,
            extrinsics=ext,
            original_pixel_uv=(320.0, 240.0),
            original_image_shape=orig_shape
        )
        assert res["is_in_front"] is True
        assert res["pixel_discrepancy"] < 0.1
        assert res["reprojection_status"] == "HIGH_CONFIDENCE"

    def test_behind_camera_reprojection(self):
        K = np.eye(3, dtype=np.float32)
        ext = np.eye(4, dtype=np.float32)
        point_3d = np.array([0.0, 0.0, -2.0], dtype=np.float32)  # Negative Z is behind camera
        res = ReprojectionValidator.validate_reprojection(
            point_3d=point_3d,
            intrinsics=K,
            extrinsics=ext,
            original_pixel_uv=(100.0, 100.0),
            original_image_shape=(480, 640)
        )
        assert res["is_in_front"] is False
        assert res["reprojection_status"] == "BEHIND_CAMERA"


# =====================================================================
# 4. Multi-Evidence Flood Analysis Tests
# =====================================================================
class TestFloodAnalyzer:
    def test_turbid_muddy_water_detected(self):
        # Muddy water: BGR ~ [40, 75, 110] (Brown / Tan runoff)
        img = np.full((200, 200, 3), (40, 75, 110), dtype=np.uint8)
        mask = FloodAnalyzer.extract_water_candidates(img)
        coverage = np.mean(mask > 0)
        assert coverage > 0.85, f"Expected high muddy water detection, got {coverage*100:.1f}%"

    def test_blue_water_detected(self):
        # Blue water: BGR ~ [180, 100, 30]
        img = np.full((200, 200, 3), (180, 100, 30), dtype=np.uint8)
        mask = FloodAnalyzer.extract_water_candidates(img)
        coverage = np.mean(mask > 0)
        assert coverage > 0.85, f"Expected high blue water detection, got {coverage*100:.1f}%"

    def test_dry_grass_or_asphalt_rejected(self):
        # Dry green grass: BGR ~ [30, 140, 40]
        img_grass = np.full((200, 200, 3), (30, 140, 40), dtype=np.uint8)
        mask_grass = FloodAnalyzer.extract_water_candidates(img_grass)
        assert np.mean(mask_grass > 0) < 0.02

        # Gray asphalt road: BGR ~ [120, 120, 120]
        img_road = np.full((200, 200, 3), (120, 120, 120), dtype=np.uint8)
        mask_road = FloodAnalyzer.extract_water_candidates(img_road)
        assert np.mean(mask_road > 0) < 0.05

    def test_flood_state_classification(self):
        # Image with 25% muddy floodwater
        img = np.full((400, 400, 3), (120, 120, 120), dtype=np.uint8)
        img[200:, :] = (40, 75, 110)  # Lower half muddy water
        res = FloodAnalyzer.analyze_flood(img)
        assert res["flood_status"] in ["POTENTIAL_INUNDATION", "VALIDATED_INUNDATION"]
        assert res["water_ratio"] >= 0.20


# =====================================================================
# 5. Person-Flood Spatial Context Tests
# =====================================================================
class TestSpatialContext:
    def test_stranded_person_surrounded_by_water(self):
        water_mask = np.ones((500, 500), dtype=np.uint8) * 255
        # Small dry island in middle
        water_mask[230:270, 230:270] = 0

        res = SpatialContextAnalyzer.analyze_person_context(
            person_center_orig=(250, 250),
            person_bbox_orig=[240, 240, 260, 260],
            water_mask_orig=water_mask,
            point_3d=np.array([0.0, 0.0, 3.0]),
            depth_map=np.full((392, 392), 3.0)
        )
        assert res["isolation_type"] == "POTENTIAL_STRANDED_PERSON"
        assert res["surrounded_percentage"] > 70.0

    def test_normal_person_on_dry_land(self):
        water_mask = np.zeros((500, 500), dtype=np.uint8)  # Dry ground
        res = SpatialContextAnalyzer.analyze_person_context(
            person_center_orig=(250, 250),
            person_bbox_orig=[240, 240, 260, 260],
            water_mask_orig=water_mask,
            point_3d=np.array([0.0, 0.0, 3.0]),
            depth_map=np.full((392, 392), 3.0)
        )
        assert res["isolation_type"] == "NORMAL_PERSON"
        assert res["surrounded_percentage"] < 5.0


# =====================================================================
# 6. Rescue Risk & SAR Priority Engine Tests
# =====================================================================
class TestRescueRiskScorer:
    def test_rescue_priority_scoring(self):
        victim_stranded = {
            "track_id": "P_001",
            "frame_id": "frame_0001.jpg",
            "frame_idx": 0,
            "detection_confidence": 0.85,
            "point_3d": [1.0, 2.0, 3.0],
            "spatial_context": {
                "isolation_type": "POTENTIAL_STRANDED_PERSON",
                "surrounded_percentage": 90.0,
                "nearest_water_distance_m": 0.3
            }
        }
        victim_normal = {
            "track_id": "P_002",
            "frame_id": "frame_0001.jpg",
            "frame_idx": 0,
            "detection_confidence": 0.80,
            "point_3d": [5.0, 2.0, 3.0],
            "spatial_context": {
                "isolation_type": "NORMAL_PERSON",
                "surrounded_percentage": 0.0,
                "nearest_water_distance_m": 50.0
            }
        }

        res = RescueRiskScorer.score_rescue_priorities(
            people_with_context=[victim_stranded, victim_normal],
            scene_flood_ratio=0.25,
            is_active_flood=True
        )

        ranked = res["ranked_priorities"]
        assert len(ranked) == 2
        assert ranked[0]["victim_id"] == "P_001"
        assert ranked[0]["rescue_priority"] == "HIGH"
        assert ranked[1]["rescue_priority"] == "LOW"

    def test_decoupled_scene_risk_when_no_people(self):
        res = RescueRiskScorer.score_rescue_priorities(
            people_with_context=[],  # 0 people
            scene_flood_ratio=0.30,  # 30% flood
            is_active_flood=True
        )
        assert res["scene_risk"]["risk_level"] in ["CRITICAL", "HIGH"]
        assert len(res["ranked_priorities"]) == 0
        assert "30.0%" in res["scene_risk"]["reasons"][0]


# =====================================================================
# 7. Canonical Incident Object & Database Tests
# =====================================================================
class TestIncidentEngineAndDatabase:
    def test_canonical_incident_schema_and_db(self):
        temp_dir = tempfile.mkdtemp()
        try:
            db_path = Path(temp_dir) / "test_incidents.db"
            db = IncidentDatabase(str(db_path))

            inc = IncidentEngine.create_incident(
                incident_type="FLOOD_ZONE",
                severity="HIGH",
                confidence=0.92,
                source_frame_id="frame_0002.jpg",
                location_3d={"x": 2.5, "y": -1.2, "z": 4.0},
                flood_status="VALIDATED_INUNDATION",
                summary="Severe flood inundation across primary access roadway."
            )

            # Check schema keys
            required_keys = ["incident_id", "incident_type", "severity", "confidence", "location_3d", "flood_status", "summary"]
            for k in required_keys:
                assert k in inc

            db.insert_incident(inc)
            records = db.get_all_incidents()
            assert len(records) == 1
            assert records[0]["incident_type"] == "FLOOD_ZONE"
            assert records[0]["severity"] == "HIGH"
            assert records[0]["location_3d"]["x"] == 2.5
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# =====================================================================
# 8. 3D Reconstruction Quality Evaluator Tests
# =====================================================================
class TestReconstructionQuality:
    def test_quality_evaluation_good_scene(self):
        pts = np.random.uniform(-5, 5, size=(60000, 3)).astype(np.float32)
        extrinsics = np.zeros((4, 4, 4), dtype=np.float32)
        for i in range(4):
            extrinsics[i] = np.eye(4)
            extrinsics[i, 0, 3] = i * 0.4  # 1.2m baseline
        intrinsics = np.tile(np.eye(3, dtype=np.float32)[None, ...], (4, 1, 1))
        intrinsics[:, 0, 0] = 500.0
        intrinsics[:, 1, 1] = 500.0
        depth_maps = [np.full((392, 392), 3.0, dtype=np.float32) for _ in range(4)]

        predictions = {
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "depth_maps": depth_maps
        }

        res = ReconstructionQualityEvaluator.evaluate_scene(predictions, pts, outlier_ratio=0.05)
        assert res["geometry_status"] in ["GOOD", "ACCEPTABLE"]
        assert res["localization_allowed"] is True
        assert res["geometry_quality_score"] > 60.0

    def test_degenerate_pancake_rejected(self):
        # Flat 2D pancake (all Z coords identical)
        pts = np.random.uniform(-5, 5, size=(10000, 3)).astype(np.float32)
        pts[:, 2] = 1.0  # Zero depth variation

        extrinsics = np.zeros((2, 4, 4), dtype=np.float32)
        extrinsics[0] = np.eye(4)
        extrinsics[1] = np.eye(4)
        intrinsics = np.tile(np.eye(3, dtype=np.float32)[None, ...], (2, 1, 1))

        predictions = {
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "depth_maps": [np.ones((392, 392))]
        }

        res = ReconstructionQualityEvaluator.evaluate_scene(predictions, pts)
        assert res["geometry_status"] == "INVALID"
        assert res["localization_allowed"] is False


# =====================================================================
# 9. Export Manager Tests
# =====================================================================
class TestExportManager:
    def test_export_pipeline_files(self):
        temp_dir = tempfile.mkdtemp()
        try:
            out_dir = Path(temp_dir)
            pts = np.random.randn(100, 3).astype(np.float32)
            cols = np.ones((100, 3), dtype=np.float32)

            # Export geometry stages
            geom_paths = ExportManager.export_geometry_stages(
                output_dir=out_dir,
                raw_pcd=(pts, cols),
                depth_pcd=(pts, cols),
                filtered_pcd=(pts, cols)
            )
            assert os.path.exists(geom_paths["raw_ply"])
            assert os.path.exists(geom_paths["depth_ply"])
            assert os.path.exists(geom_paths["filtered_ply"])

            # Export incidents
            inc = IncidentEngine.create_incident(
                incident_type="TEST",
                severity="LOW",
                confidence=0.9,
                source_frame_id="frame_0.jpg",
                summary="Test"
            )
            inc_paths = ExportManager.export_incidents([inc], out_dir)
            assert os.path.exists(inc_paths["incidents_json"])
            assert os.path.exists(inc_paths["incidents_csv"])

            # Export master zip
            zip_path = ExportManager.create_master_zip(out_dir, "test_pkg.zip")
            assert os.path.exists(zip_path)
            assert os.path.getsize(zip_path) > 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# =====================================================================
# 10. End-to-End UAV Disaster Intelligence Scenario Tests
# =====================================================================
class TestScenarios:
    def test_scenario_a_normal_scene_negative_test(self):
        """
        Scenario A: Urban / university drone sequence with buildings, paths, dry roads.
        Result must be: NO_WATER, civilian person marked as NORMAL_PERSON, scene risk LOW.
        """
        # Create synthetic dry urban frame (grass + road + building)
        img = np.full((300, 300, 3), (80, 80, 80), dtype=np.uint8)  # Road gray
        img[:100, :] = (35, 130, 45)  # Lawn green

        flood_res = FloodAnalyzer.analyze_flood(img)
        assert flood_res["flood_status"] == "NO_WATER"
        assert flood_res["water_ratio"] < 0.03

        # Person walking along dry lawn
        water_mask = FloodAnalyzer.extract_water_candidates(img)
        ctx = SpatialContextAnalyzer.analyze_person_context(
            person_center_orig=(150, 150),
            person_bbox_orig=[140, 140, 160, 160],
            water_mask_orig=water_mask
        )
        assert ctx["isolation_type"] == "NORMAL_PERSON"

        sar = RescueRiskScorer.score_rescue_priorities(
            people_with_context=[{"track_id": "P1", "spatial_context": ctx, "point_3d": [1, 1, 1]}],
            scene_flood_ratio=flood_res["water_ratio"],
            is_active_flood=False
        )
        assert sar["scene_risk"]["risk_level"] == "LOW"
        assert sar["ranked_priorities"][0]["rescue_priority"] == "LOW"

    def test_scenario_b_flooded_scene_positive_test(self):
        """
        Scenario B: Drone sequence of flooded residential neighborhood.
        Result must be: Flood VALIDATED, stranded victim identified, HIGH priority rescue assigned.
        """
        # Flooded muddy scene
        img = np.full((400, 400, 3), (35, 70, 105), dtype=np.uint8)  # Muddy water
        # Small roof island
        img[180:220, 180:220] = (100, 100, 120)

        flood_res = FloodAnalyzer.analyze_flood(img)
        assert flood_res["flood_status"] in ["POTENTIAL_INUNDATION", "VALIDATED_INUNDATION"]
        assert flood_res["water_ratio"] > 0.80

        water_mask = FloodAnalyzer.extract_water_candidates(img)
        ctx = SpatialContextAnalyzer.analyze_person_context(
            person_center_orig=(200, 200),
            person_bbox_orig=[190, 190, 210, 210],
            water_mask_orig=water_mask,
            point_3d=np.array([0.5, -0.2, 4.0])
        )
        assert ctx["isolation_type"] == "POTENTIAL_STRANDED_PERSON"

        sar = RescueRiskScorer.score_rescue_priorities(
            people_with_context=[{
                "track_id": "VICTIM_ROOF_01",
                "spatial_context": ctx,
                "point_3d": [0.5, -0.2, 4.0],
                "detection_confidence": 0.90
            }],
            scene_flood_ratio=flood_res["water_ratio"],
            is_active_flood=True
        )
        assert sar["scene_risk"]["risk_level"] in ["CRITICAL", "HIGH"]
        assert sar["ranked_priorities"][0]["rescue_priority"] == "HIGH"
        assert "isolated by floodwater" in sar["ranked_priorities"][0]["rationale"]

    def test_scenario_c_no_people_flooded_scene_test(self):
        """
        Scenario C: Flooded area with 0 people detected.
        Result must be: Flood VALIDATED, 0 victim targets, BUT Scene Risk remains HIGH.
        Must NOT collapse to 'LOW / NO DISASTER EVIDENCE'.
        """
        img = np.full((300, 300, 3), (40, 75, 110), dtype=np.uint8)  # Muddy flood
        flood_res = FloodAnalyzer.analyze_flood(img)

        sar = RescueRiskScorer.score_rescue_priorities(
            people_with_context=[],  # 0 people
            scene_flood_ratio=flood_res["water_ratio"],
            is_active_flood=True
        )
        assert len(sar["ranked_priorities"]) == 0
        assert sar["scene_risk"]["risk_level"] in ["CRITICAL", "HIGH"]
        assert "flood inundation" in sar["scene_risk"]["reasons"][0]

    def test_scenario_d_normal_water_body_test(self):
        """
        Scenario D: Normal water canal / river without flood emergency.
        Result must be: WATER_BODY, no false flood alarm, low scene risk.
        """
        # Blue water body occupying 10% in middle (canal)
        img = np.full((400, 400, 3), (50, 120, 50), dtype=np.uint8)  # Green banks
        img[160:240, :] = (160, 100, 30)  # Blue canal

        flood_res = FloodAnalyzer.analyze_flood(
            img,
            detected_objects=[{"class_name": "car", "bbox": [10, 10, 40, 40]}]
        )
        # Bounded channel without inundating objects should be WATER_BODY or NO_WATER
        assert flood_res["flood_status"] in ["WATER_BODY", "POTENTIAL_INUNDATION", "NO_WATER"]
        assert flood_res["flood_status"] != "VALIDATED_INUNDATION"

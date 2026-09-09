import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("IncidentEngine")

class IncidentEngine:
    """
    Canonical Incident Store & Deduplication Engine.
    Enforces Phase 19 Canonical Incident Object schema across the entire application:
    incident = {
        "id": "INC-001",
        "type": "POTENTIAL_STRANDED_PERSON",
        "severity": "HIGH",
        "confidence": 0.91,
        "evidence": "...",
        "track_id": 1,
        "frame_ids": [1, 2],
        "bbox_2d": [x1, y1, x2, y2],
        "location_3d": {"x": 1.2, "y": 3.4, "z": 5.6, "status": "LOCALIZED_3D"},
        "reprojection_error_px": 8.4,
        "flood_context": "HIGH",
        "isolation_score": 0.88,
        "rescue_priority": "HIGH",
        "validation_status": "VALIDATED"
    }
    """

    def __init__(self, db_dir: Optional[str] = None):
        self.incidents: List[Dict[str, Any]] = []
        self.db_path = None
        if db_dir:
            out_dir = Path(db_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = out_dir / "incidents.db"
            self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS incidents (
                        id TEXT PRIMARY KEY,
                        type TEXT,
                        severity TEXT,
                        confidence REAL,
                        evidence TEXT,
                        track_id INTEGER,
                        frame_ids TEXT,
                        bbox_2d TEXT,
                        location_3d TEXT,
                        reprojection_error_px REAL,
                        flood_context TEXT,
                        isolation_score REAL,
                        rescue_priority TEXT,
                        validation_status TEXT,
                        data_json TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to initialize SQLite incident database: {e}")

    def add_incident(self, incident: Dict[str, Any]) -> str:
        """
        Adds or merges an incident into the canonical store.
        Deduplicates by track_id or spatial proximity.
        """
        track_id = incident.get("track_id")
        inc_type = incident.get("type", "INCIDENT")

        # 1. Check for existing incident with same track_id
        if track_id is not None:
            for existing in self.incidents:
                if existing.get("track_id") == track_id and existing.get("type") == inc_type:
                    # Merge frames
                    new_frames = incident.get("frame_ids", [incident.get("frame_idx", 0)])
                    existing_frames = set(existing.get("frame_ids", []))
                    existing_frames.update(new_frames)
                    existing["frame_ids"] = sorted(list(existing_frames))
                    # Update confidence if higher
                    if incident.get("confidence", 0.0) > existing.get("confidence", 0.0):
                        existing["confidence"] = incident["confidence"]
                    # Update 3D location if previous was unavailable
                    if existing["location_3d"]["status"] != "LOCALIZED_3D" and incident["location_3d"]["status"] == "LOCALIZED_3D":
                        existing["location_3d"] = incident["location_3d"]
                        existing["reprojection_error_px"] = incident.get("reprojection_error_px")
                    # Update DB if active
                    self._persist_incident(existing)
                    return existing["id"]

        # 2. Check for spatial deduplication (if both localized within 2.0m radius)
        loc = incident.get("location_3d", {})
        if loc.get("status") == "LOCALIZED_3D" and loc.get("x") is not None:
            x1, y1, z1 = loc["x"], loc["y"], loc["z"]
            for existing in self.incidents:
                if existing.get("type") == inc_type:
                    e_loc = existing.get("location_3d", {})
                    if e_loc.get("status") == "LOCALIZED_3D" and e_loc.get("x") is not None:
                        dist = ((x1 - e_loc["x"])**2 + (y1 - e_loc["y"])**2 + (z1 - e_loc["z"])**2)**0.5
                        if dist < 2.0:
                            # Merge into existing incident
                            new_frames = incident.get("frame_ids", [incident.get("frame_idx", 0)])
                            existing_frames = set(existing.get("frame_ids", []))
                            existing_frames.update(new_frames)
                            existing["frame_ids"] = sorted(list(existing_frames))
                            self._persist_incident(existing)
                            return existing["id"]

        # 3. Create new canonical incident ID
        inc_num = len(self.incidents) + 1
        inc_id = f"INC-{inc_num:03d}"
        incident["id"] = inc_id

        inc_type = incident.get("incident_type", incident.get("type", "INCIDENT"))
        summary = incident.get("summary", incident.get("evidence", "No evidence summary provided."))

        # Normalize schema fields
        normalized = {
            "id": inc_id,
            "incident_id": inc_id,
            "type": inc_type,
            "incident_type": inc_type,
            "severity": incident.get("severity", "MEDIUM"),
            "confidence": round(float(incident.get("confidence", 0.80)), 2),
            "evidence": summary,
            "summary": summary,
            "track_id": incident.get("track_id"),
            "frame_ids": incident.get("frame_ids", [incident.get("frame_idx", 0)]),
            "source_frame_id": incident.get("source_frame_id", f"frame_{incident.get('frame_idx', 0):04d}.jpg"),
            "bbox_2d": incident.get("bbox_2d", [0, 0, 0, 0]),
            "location_3d": incident.get("location_3d", {"x": None, "y": None, "z": None, "status": "UNAVAILABLE"}),
            "location_status": incident.get("location_status", "LOCALIZED" if incident.get("location_3d", {}).get("status") == "LOCALIZED_3D" else "UNLOCALIZED"),
            "reprojection_error_px": incident.get("reprojection_error_px"),
            "flood_context": incident.get("flood_context", incident.get("flood_status", "NONE")),
            "flood_status": incident.get("flood_status", incident.get("flood_context", "NONE")),
            "isolation_score": round(float(incident.get("isolation_score", 0.0)), 2),
            "rescue_priority": incident.get("rescue_priority", "LOW"),
            "validation_status": incident.get("validation_status", "VALIDATED")
        }

        self.incidents.append(normalized)
        self._persist_incident(normalized)
        return inc_id

    def _persist_incident(self, inc: Dict[str, Any]):
        if not self.db_path:
            return
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO incidents (
                        id, type, severity, confidence, evidence, track_id, frame_ids,
                        bbox_2d, location_3d, reprojection_error_px, flood_context,
                        isolation_score, rescue_priority, validation_status, data_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    inc["id"],
                    inc["type"],
                    inc["severity"],
                    inc["confidence"],
                    inc["evidence"],
                    inc.get("track_id"),
                    json.dumps(inc.get("frame_ids", [])),
                    json.dumps(inc.get("bbox_2d", [])),
                    json.dumps(inc.get("location_3d", {})),
                    inc.get("reprojection_error_px"),
                    inc.get("flood_context"),
                    inc.get("isolation_score"),
                    inc.get("rescue_priority"),
                    inc.get("validation_status"),
                    json.dumps(inc)
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Error persisting incident {inc.get('id')}: {e}")

    def get_all_incidents(self) -> List[Dict[str, Any]]:
        return self.incidents

    def get_table_rows(self) -> List[List[Any]]:
        """Formats canonical incidents into UI DataFrame table rows."""
        rows = []
        for inc in self.incidents:
            loc = inc["location_3d"]
            if loc.get("status") == "LOCALIZED_3D" and loc.get("x") is not None:
                reproj_str = f" ({inc['reprojection_error_px']}px)" if inc.get("reprojection_error_px") is not None else ""
                loc_str = f"X:{loc['x']:.2f}, Y:{loc['y']:.2f}, Z:{loc['z']:.2f}{reproj_str}"
            else:
                loc_str = "UNAVAILABLE"

            src_frames = f"Frames {inc.get('frame_ids', [])}"
            trk_str = f"TRK-{inc['track_id']}" if inc.get("track_id") is not None else "N/A"

            rows.append([
                inc["id"],
                inc["type"],
                inc["severity"],
                f"{inc['confidence']:.2f}",
                trk_str,
                loc_str,
                inc.get("flood_context", "NONE"),
                f"{inc.get('isolation_score', 0.0):.2f}",
                inc.get("rescue_priority", "LOW"),
                inc.get("validation_status", "VALIDATED"),
                src_frames
            ])
        return rows

    # Helper alias
    insert_incident = add_incident

    @staticmethod
    def create_incident(
        incident_type: str,
        severity: str = "MEDIUM",
        confidence: float = 0.85,
        source_frame_id: str = "frame_0000.jpg",
        location_3d: Optional[Dict[str, Any]] = None,
        flood_status: str = "NONE",
        summary: str = "",
        track_id: Optional[int] = None,
        bbox: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        inc_id = f"INC-{np.random.randint(100, 999)}"
        loc = location_3d if location_3d else {"x": None, "y": None, "z": None, "status": "UNAVAILABLE"}
        if "status" not in loc:
            loc["status"] = "LOCALIZED_3D" if loc.get("x") is not None else "UNAVAILABLE"

        return {
            "incident_id": inc_id,
            "id": inc_id,
            "incident_type": incident_type,
            "type": incident_type,
            "severity": severity,
            "confidence": round(float(confidence), 2),
            "location_3d": loc,
            "location_status": "LOCALIZED" if loc.get("status") == "LOCALIZED_3D" else "UNLOCALIZED",
            "flood_status": flood_status,
            "source_frame_id": source_frame_id,
            "summary": summary,
            "evidence": summary,
            "track_id": track_id,
            "bbox_2d": bbox or [0, 0, 0, 0]
        }

    @staticmethod
    def create_flood_incident(
        flood_eval: Dict[str, Any],
        source_frame_id: str = "frame_0000.jpg",
        water_mask: Optional[np.ndarray] = None,
        point_map: Optional[np.ndarray] = None,
        vggt_shape: Tuple[int, int] = (392, 392),
        orig_shape: Tuple[int, int] = (1080, 1920),
        frame_idx: int = 0
    ) -> Dict[str, Any]:
        water_ratio = flood_eval.get("water_ratio", 0.0)
        status = flood_eval.get("flood_status", "POTENTIAL_INUNDATION")
        severity = "CRITICAL" if water_ratio > 0.30 else "HIGH"

        loc_3d = {"x": None, "y": None, "z": None, "status": "UNAVAILABLE"}
        if point_map is not None and water_mask is not None and np.any(water_mask > 0):
            # Centroid of water
            ys, xs = np.where(water_mask > 0)
            if len(xs) > 0:
                mx, my = int(np.mean(xs)), int(np.mean(ys))
                # Map to point_map
                px = int(mx * vggt_shape[1] / orig_shape[1])
                py = int(my * vggt_shape[0] / orig_shape[0])
                px = min(vggt_shape[1] - 1, max(0, px))
                py = min(vggt_shape[0] - 1, max(0, py))
                p = point_map[py, px]
                if not np.isnan(p).any() and not np.isclose(p, [0, 0, 0]).all():
                    loc_3d = {"x": round(float(p[0]), 2), "y": round(float(p[1]), 2), "z": round(float(p[2]), 2), "status": "LOCALIZED_3D"}

        return IncidentEngine.create_incident(
            incident_type="FLOOD_ZONE",
            severity=severity,
            confidence=0.90,
            source_frame_id=source_frame_id,
            location_3d=loc_3d,
            flood_status=status,
            summary=f"Inundation confirmed with {water_ratio*100:.1f}% water surface coverage."
        )

    @staticmethod
    def create_victim_incident(
        victim_data: Dict[str, Any],
        source_frame_id: str = "frame_0000.jpg"
    ) -> Dict[str, Any]:
        priority = victim_data.get("rescue_priority", "LOW")
        severity = "CRITICAL" if priority == "HIGH" else ("HIGH" if priority == "MEDIUM" else "LOW")
        v_id = victim_data.get("victim_id", "VICTIM")
        pos = victim_data.get("world_position")
        loc_3d = {"x": pos["x"], "y": pos["y"], "z": pos["z"], "status": "LOCALIZED_3D"} if pos else None

        ctx = victim_data.get("spatial_context") or {}
        iso_type = ctx.get("isolation_type", "NORMAL_PERSON")
        inc_type = "STRANDED_PERSON" if "STRANDED" in iso_type else ("ISOLATED_PERSON" if "ISOLATED" in iso_type else "PERSON_IN_DISTRESS")

        return IncidentEngine.create_incident(
            incident_type=inc_type,
            severity=severity,
            confidence=0.88,
            source_frame_id=source_frame_id,
            location_3d=loc_3d,
            flood_status=iso_type,
            summary=victim_data.get("rationale", f"Victim {v_id} triage status {priority}")
        )

    @staticmethod
    def create_vehicle_incident(
        vehicle_data: Dict[str, Any],
        source_frame_id: str = "frame_0000.jpg",
        flood_status: str = "NO_WATER"
    ) -> Dict[str, Any]:
        val_cls = str(vehicle_data.get("final_class", vehicle_data.get("class_name", "VEHICLE"))).upper()
        if "UNCERTAIN" in val_cls:
            logger.info("[INCIDENT GATED] UNCERTAIN_OBJECT excluded from canonical vehicle incidents.")
            return None

        pos = vehicle_data.get("point_3d")
        loc_3d = {"x": round(float(pos[0]), 2), "y": round(float(pos[1]), 2), "z": round(float(pos[2]), 2), "status": "LOCALIZED_3D"} if pos is not None else None
        is_flooded = "INUNDATION" in flood_status
        inc_type = "VEHICLE_IN_FLOOD" if is_flooded else "VEHICLE_DETECTED"
        severity = "HIGH" if is_flooded else "LOW"

        return IncidentEngine.create_incident(
            incident_type=inc_type,
            severity=severity,
            confidence=float(vehicle_data.get("validation_confidence", vehicle_data.get("detection_confidence", 0.85))),
            source_frame_id=source_frame_id,
            location_3d=loc_3d,
            flood_status=flood_status,
            summary=f"Vehicle located in aerial frame ({flood_status})."
        )


IncidentDatabase = IncidentEngine

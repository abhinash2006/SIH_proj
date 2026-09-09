import os
import json
import sqlite3
import logging
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("IncidentManager")

class DisasterIncidentManager:
    """
    Manages disaster incident reports, detected victims/objects, damage alerts, and hazard maps.
    Performs Spatial & Temporal Deduplication to merge multi-frame observations into single unique incidents.
    Persists incident database into JSON and SQLite formats.
    """

    def __init__(self, db_dir: str = "outputs"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.db_dir / "incidents.json"
        self.db_path = self.db_dir / "incidents.db"
        self.incidents: List[Dict[str, Any]] = []
        self.incident_counter = 0
        self._init_sqlite()

    def _init_sqlite(self):
        """Initializes SQLite database table for disaster incidents."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS incidents")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id TEXT PRIMARY KEY,
                    incident_type TEXT,
                    object_id TEXT,
                    confidence REAL,
                    severity TEXT,
                    location_status TEXT,
                    x_m REAL,
                    y_m REAL,
                    z_m REAL,
                    reprojection_error_px REAL,
                    latitude REAL,
                    longitude REAL,
                    source_frames TEXT,
                    observation_count INTEGER,
                    evidence TEXT,
                    status TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as err:
            logger.warning(f"[WARNING] SQLite init notice: {err}")

    def add_incident(self, incident_data: Dict[str, Any]) -> str:
        """
        Adds a disaster incident. Performs deduplication against existing incidents
        using object_id, 3D spatial proximity (<3m), and semantic class.
        """
        inc_type = incident_data.get("incident_type", "HAZARD")
        obj_id = incident_data.get("object_id", incident_data.get("target_object_id", "N/A"))
        frame_idx = int(incident_data.get("frame_idx", 0))
        
        raw_x = incident_data.get("x_m")
        raw_y = incident_data.get("y_m")
        raw_z = incident_data.get("z_m")
        loc_status = incident_data.get("location_status")

        if raw_x is None or raw_y is None or raw_z is None or loc_status == "UNLOCALIZED":
            x_m, y_m, z_m = None, None, None
            location_status = "UNLOCALIZED"
        else:
            x_m, y_m, z_m = float(raw_x), float(raw_y), float(raw_z)
            location_status = loc_status or "LOCALIZED_3D"

        # Check for duplicate existing incident
        for existing in self.incidents:
            if existing["incident_type"] != inc_type:
                continue

            same_object = (obj_id != "N/A" and existing["object_id"] == obj_id)
            
            spatial_near = False
            if x_m is not None and existing["x_m"] is not None:
                dist_3d = math.sqrt((x_m - existing["x_m"])**2 + (y_m - existing["y_m"])**2 + (z_m - existing["z_m"])**2)
                if dist_3d < 3.0: # 3 meters threshold
                    spatial_near = True

            if same_object or spatial_near:
                # Merge observation into existing incident
                if frame_idx not in existing["source_frames"]:
                    existing["source_frames"].append(frame_idx)
                    existing["source_frames"].sort()
                
                existing["observation_count"] += 1
                base_conf = float(incident_data.get("confidence", existing["confidence"]))
                
                # Temporal boost: boost confidence with repeated observations
                existing["confidence"] = round(min(0.98, max(existing["confidence"], base_conf) + 0.04), 3)

                # Update 3D coordinates to higher confidence sample
                if x_m is not None and (existing["x_m"] is None or base_conf > existing["confidence"] - 0.05):
                    existing["x_m"], existing["y_m"], existing["z_m"] = x_m, y_m, z_m
                    existing["location_status"] = location_status
                    existing["reprojection_error_px"] = incident_data.get("reprojection_error_px")

                logger.info(f"[DEDUPLICATION] Merged observation frame {frame_idx} into {existing['incident_id']} (Obs Count: {existing['observation_count']})")
                self._save_to_sqlite(existing)
                self._save_to_json()
                return existing["incident_id"]

        # Create new unique incident
        self.incident_counter += 1
        incident_id = f"INCIDENT_{self.incident_counter:03d}"
        
        record = {
            "incident_id": incident_id,
            "incident_type": inc_type,
            "object_id": obj_id,
            "target_object_class": incident_data.get("target_object_class", incident_data.get("hazard_class", incident_data.get("class_name", "object"))),
            "confidence": round(float(incident_data.get("confidence", 0.70)), 3),
            "detection_confidence": round(float(incident_data.get("detection_confidence", incident_data.get("confidence", 0.70))), 3),
            "localization_confidence": round(float(incident_data.get("localization_confidence", 0.0 if location_status == "UNLOCALIZED" else 0.80)), 3),
            "severity": incident_data.get("severity", "HIGH"),
            "location_status": location_status,
            "x_m": x_m,
            "y_m": y_m,
            "z_m": z_m,
            "reprojection_error_px": incident_data.get("reprojection_error_px"),
            "latitude": incident_data.get("latitude"),
            "longitude": incident_data.get("longitude"),
            "coordinate_system": incident_data.get("coordinate_system", "VGGT Local World Coordinates" if x_m is not None else "UNLOCALIZED"),
            "source_frames": [frame_idx],
            "observation_count": 1,
            "evidence": incident_data.get("evidence", "Visual anomaly detected in drone imagery."),
            "status": incident_data.get("status", "NEEDS_VERIFICATION")
        }

        self.incidents.append(record)
        self._save_to_sqlite(record)
        self._save_to_json()
        logger.info(f"[NEW INCIDENT] Created {incident_id}: {inc_type} at frame {frame_idx} (Loc Status: {location_status})")
        return incident_id

    def _save_to_sqlite(self, record: Dict[str, Any]):
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO incidents VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                record["incident_id"], record["incident_type"], record["object_id"],
                record["confidence"], record["severity"], record["location_status"],
                record["x_m"], record["y_m"], record["z_m"], record["reprojection_error_px"],
                record["latitude"], record["longitude"], json.dumps(record["source_frames"]),
                record["observation_count"], record["evidence"], record["status"]
            ))
            conn.commit()
            conn.close()
        except Exception as err:
            logger.warning(f"[WARNING] Could not insert incident to SQLite: {err}")

    def _save_to_json(self):
        with open(self.json_path, "w") as f:
            json.dump(self.incidents, f, indent=4)

    def get_all_incidents(self) -> List[Dict[str, Any]]:
        return self.incidents

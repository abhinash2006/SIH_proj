import os
import json
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("IncidentManager")

class DisasterIncidentManager:
    """
    Manages disaster incident reports, detected victims/objects, damage alerts, and hazard maps.
    Persists incident database into JSON and SQLite formats for search-and-rescue decision support.
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
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id TEXT PRIMARY KEY,
                    incident_type TEXT,
                    object_id TEXT,
                    confidence REAL,
                    severity TEXT,
                    x_m REAL,
                    y_m REAL,
                    z_m REAL,
                    latitude REAL,
                    longitude REAL,
                    frame_idx INTEGER,
                    evidence TEXT,
                    status TEXT,
                    ai_certainty TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as err:
            logger.warning(f"[WARNING] SQLite init notice: {err}")

    def add_incident(self, incident_data: Dict[str, Any]) -> str:
        """Adds a new disaster incident record."""
        self.incident_counter += 1
        incident_id = f"INCIDENT_{self.incident_counter:03d}"
        
        record = {
            "incident_id": incident_id,
            "incident_type": incident_data.get("incident_type", "HAZARD"),
            "object_id": incident_data.get("object_id", incident_data.get("target_object_id", "N/A")),
            "confidence": float(incident_data.get("confidence", 0.70)),
            "severity": incident_data.get("severity", "HIGH"),
            "x_m": float(incident_data.get("x_m", 0.0)),
            "y_m": float(incident_data.get("y_m", 0.0)),
            "z_m": float(incident_data.get("z_m", 0.0)),
            "latitude": incident_data.get("latitude", None),
            "longitude": incident_data.get("longitude", None),
            "coordinate_system": incident_data.get("coordinate_system", "Local reconstruction coordinates"),
            "frame_idx": int(incident_data.get("frame_idx", 0)),
            "evidence": incident_data.get("evidence", "Visual anomaly detected in aerial imagery."),
            "status": incident_data.get("status", "NEEDS_VERIFICATION"),
            "ai_certainty": incident_data.get("ai_certainty", "DETECTED" if incident_data.get("confidence", 0.7) > 0.75 else "INFERRED")
        }

        self.incidents.append(record)
        self._save_to_sqlite(record)
        self._save_to_json()
        return incident_id

    def _save_to_sqlite(self, record: Dict[str, Any]):
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO incidents VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                record["incident_id"], record["incident_type"], record["object_id"],
                record["confidence"], record["severity"], record["x_m"], record["y_m"], record["z_m"],
                record["latitude"], record["longitude"], record["frame_idx"],
                record["evidence"], record["status"], record["ai_certainty"]
            ))
            conn.commit()
            conn.close()
        except Exception as err:
            logger.warning(f"[WARNING] Could not insert incident to SQLite: {err}")

    def _save_to_json(self):
        with open(self.json_path, "w") as f:
            json.dump(self.incidents, f, indent=4)

    def update_incident_status(self, incident_id: str, new_status: str) -> bool:
        """Updates responder verification status (CONFIRMED, REJECTED, NEEDS_VERIFICATION)."""
        for inc in self.incidents:
            if inc["incident_id"] == incident_id:
                inc["status"] = new_status
                self._save_to_json()
                self._save_to_sqlite(inc)
                return True
        return False

    def get_all_incidents(self) -> List[Dict[str, Any]]:
        return self.incidents

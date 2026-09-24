"""
Mission Output Manager for UAV 3D Reconstruction System.
Standardizes directory structures, artifact serialization, and structured JSON reports.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MissionArtifacts:
    mission_id: str
    mission_dir: str
    frames_dir: str
    depth_dir: str
    masks_dir: str
    poses_dir: str
    pointcloud_dir: str
    mesh_dir: str
    textures_dir: str
    final_glb: str
    final_obj: str
    final_ply: str
    metrics_json: str
    processing_report_json: str


class OutputManager:
    """
    Creates and maintains standardized mission artifact structures.
    """

    def __init__(self, base_output_dir: str = "output"):
        self.base_dir = Path(base_output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def initialize_mission(self, mission_id: str, custom_dir: Optional[str] = None) -> MissionArtifacts:
        """Sets up all subdirectories for a new mission."""
        if custom_dir:
            m_dir = Path(custom_dir)
        else:
            m_dir = self.base_dir / mission_id

        subdirs = {
            "frames": m_dir / "frames",
            "depth": m_dir / "depth",
            "masks": m_dir / "masks",
            "poses": m_dir / "poses",
            "pointcloud": m_dir / "pointcloud",
            "mesh": m_dir / "mesh",
            "textures": m_dir / "textures",
        }

        for d in subdirs.values():
            d.mkdir(parents=True, exist_ok=True)

        return MissionArtifacts(
            mission_id=mission_id,
            mission_dir=str(m_dir.resolve()),
            frames_dir=str(subdirs["frames"].resolve()),
            depth_dir=str(subdirs["depth"].resolve()),
            masks_dir=str(subdirs["masks"].resolve()),
            poses_dir=str(subdirs["poses"].resolve()),
            pointcloud_dir=str(subdirs["pointcloud"].resolve()),
            mesh_dir=str(subdirs["mesh"].resolve()),
            textures_dir=str(subdirs["textures"].resolve()),
            final_glb=str((m_dir / "final.glb").resolve()),
            final_obj=str((m_dir / "final.obj").resolve()),
            final_ply=str((m_dir / "final.ply").resolve()),
            metrics_json=str((m_dir / "metrics.json").resolve()),
            processing_report_json=str((m_dir / "processing_report.json").resolve()),
        )

    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str):
        """Saves dictionary to formatted JSON file."""
        fp = Path(filepath)
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"[OUTPUT] Saved JSON report to '{fp}'.")

"""
Optional COLMAP Photogrammetry Baseline Wrapper for UAV 3D Reconstruction.
Enables classical Structure-from-Motion (SfM) feature matching and sparse reconstruction.
"""

import os
import shutil
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class COLMAPWrapper:
    """
    Wraps COLMAP CLI for feature extraction, matching, and sparse reconstruction.
    Selectable through configuration: reconstruction_backend: 'colmap' or 'hybrid'.
    """

    def __init__(self, colmap_bin: str = "colmap"):
        self.colmap_bin = colmap_bin
        self.is_available = shutil.which(colmap_bin) is not None

    def run_sparse_reconstruction(
        self,
        image_dir: str,
        output_dir: str,
    ) -> Dict[str, Any]:
        """Runs COLMAP feature extraction, sequential/exhaustive matcher, and mapper."""
        if not self.is_available:
            logger.warning("[COLMAP] COLMAP binary not found in system PATH. Classical baseline unavailable.")
            return {"status": "unavailable", "message": "COLMAP binary not found in PATH."}

        out_path = Path(output_dir)
        db_path = out_path / "database.db"
        sparse_path = out_path / "sparse"
        sparse_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"[COLMAP] Running sparse reconstruction on images in '{image_dir}'...")
        try:
            # 1. Feature extractor
            subprocess.run([
                self.colmap_bin, "feature_extractor",
                "--database_path", str(db_path),
                "--image_path", str(image_dir),
            ], check=True, capture_output=True)

            # 2. Sequential matcher (optimal for aerial drone sequences)
            subprocess.run([
                self.colmap_bin, "sequential_matcher",
                "--database_path", str(db_path),
            ], check=True, capture_output=True)

            # 3. Mapper
            subprocess.run([
                self.colmap_bin, "mapper",
                "--database_path", str(db_path),
                "--image_path", str(image_dir),
                "--output_path", str(sparse_path),
            ], check=True, capture_output=True)

            logger.info("[COLMAP] Reconstruction finished successfully.")
            return {"status": "success", "sparse_dir": str(sparse_path)}
        except Exception as e:
            logger.error(f"[COLMAP] Execution failed: {e}")
            return {"status": "failed", "error": str(e)}

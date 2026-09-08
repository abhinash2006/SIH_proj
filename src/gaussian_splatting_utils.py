import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Union

logger = logging.getLogger(__name__)

class GaussianSplattingUtils:
    """
    Utilities for preparing VGGT outputs for 3D Gaussian Splatting (gsplat / 3DGS).
    """

    @staticmethod
    def prepare_gsplat_scene(
        scene_dir: Union[str, Path],
        frame_paths: List[str],
        colmap_sparse_dir: Union[str, Path]
    ) -> str:
        """
        Organizes scene directory for gsplat:
        scene_dir/
        ├── images/
        └── sparse/0/
            ├── cameras.bin
            ├── images.bin
            └── points3D.bin
        """
        scene_dir = Path(scene_dir)
        images_dir = scene_dir / "images"
        sparse_dir = scene_dir / "sparse" / "0"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy input frames
        for f in frame_paths:
            src_f = Path(f)
            dest_f = images_dir / src_f.name
            if not dest_f.exists():
                shutil.copy2(src_f, dest_f)
                
        # Copy COLMAP binary files
        colmap_src = Path(colmap_sparse_dir)
        for bin_file in ["cameras.bin", "images.bin", "points3D.bin"]:
            src_b = colmap_src / bin_file
            if src_b.exists():
                shutil.copy2(src_b, sparse_dir / bin_file)

        logger.info(f"[SUCCESS] Prepared Gaussian Splatting scene folder at: {scene_dir}")
        return str(scene_dir)

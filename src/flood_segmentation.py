"""
Modular Flood Segmentation Abstraction Layer for Drone Aerial Imagery.

Defines the abstract base class and standard result container for aerial flood segmentation.
Supports heuristic/spectral methods, classical ML, and future deep learning models
(e.g., specialized aerial flood segmentation models) without breaking downstream pipelines.

States:
- NO_WATER: Water candidate coverage is negligible (< 2% of frame).
- WATER_BODY: Normal river, canal, lake, or pond confined in natural/engineered banks.
- POTENTIAL_INUNDATION: Elevated water expanse in contact with terrain or proximity to structures.
- VALIDATED_INUNDATION: Confirmed hazardous flood spill crossing into infrastructure or dry land.

CRITICAL RULE:
Never equate raw water candidate with flood inundation. Water must be validated through
continuity, structural context, and elevation/depth evidence before being classified as a flood.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class WaterState(str, Enum):
    NO_WATER = "NO_WATER"
    WATER_BODY = "WATER_BODY"
    POTENTIAL_INUNDATION = "POTENTIAL_INUNDATION"
    VALIDATED_INUNDATION = "VALIDATED_INUNDATION"


@dataclass
class FloodSegmentationResult:
    """
    Standardized, modular output from any aerial flood segmentation model.
    Implements dict-like access for full backward compatibility.
    """
    water_state: str  # NO_WATER, WATER_BODY, POTENTIAL_INUNDATION, VALIDATED_INUNDATION
    water_ratio: float  # Fraction of frame covered by validated floodwater [0.0 - 1.0]
    candidate_ratio: float  # Fraction of frame covered by raw water candidates [0.0 - 1.0]
    validated_ratio: float  # Fraction of frame covered by verified flood components [0.0 - 1.0]
    flood_confidence: float  # Confidence in flood classification [0.0 - 1.0]
    candidate_mask: np.ndarray  # Binary mask [H, W] of candidate water pixels
    validated_mask: np.ndarray  # Binary mask [H, W] of confirmed floodwater pixels
    depth_flatness_score: float = 0.50  # 3D surface flatness score [0.0 - 1.0]
    building_overlap: float = 0.0  # Overlap ratio with detected structures
    road_overlap: float = 0.0  # Overlap ratio with detected roadways/vehicles
    largest_component_ratio: float = 0.0  # Area ratio of the largest water body
    rejection_reason: Optional[str] = None  # Explanation if flood gate is rejected
    summary_evidence: str = ""  # Human-readable summary of physical evidence
    incident_gate: str = "PASS"  # "PASS" if hazardous flood incident, "FAIL" otherwise
    frame_idx: int = 0  # Video or sequence frame index
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Dictionary compatibility helpers
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return [
            "water_state", "flood_status", "status", "water_ratio", "candidate_ratio",
            "validated_ratio", "flood_confidence", "candidate_mask", "validated_mask",
            "depth_flatness_score", "building_overlap", "road_overlap",
            "largest_component_ratio", "rejection_reason", "summary_evidence",
            "incident_gate", "frame_idx", "metadata"
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to a dictionary representation."""
        return {
            "water_state": self.water_state,
            "flood_status": self.water_state,
            "status": self.water_state,
            "water_ratio": round(self.water_ratio, 4),
            "candidate_ratio": round(self.candidate_ratio, 4),
            "validated_ratio": round(self.validated_ratio, 4),
            "flood_confidence": round(self.flood_confidence, 3),
            "candidate_mask": self.candidate_mask,
            "validated_mask": self.validated_mask,
            "depth_flatness_score": round(self.depth_flatness_score, 3),
            "building_overlap": round(self.building_overlap, 3),
            "road_overlap": round(self.road_overlap, 3),
            "largest_component_ratio": round(self.largest_component_ratio, 4),
            "rejection_reason": self.rejection_reason,
            "summary_evidence": self.summary_evidence,
            "incident_gate": self.incident_gate,
            "frame_idx": self.frame_idx,
            "metadata": self.metadata
        }

    @property
    def flood_status(self) -> str:
        return self.water_state

    @property
    def status(self) -> str:
        return self.water_state


class BaseFloodSegmenter(ABC):
    """
    Abstract Base Class for UAV aerial flood segmentation models.
    Any future deep learning model (e.g. SegFormer, FloodNet, SAM-aerial)
    must implement this interface.
    """

    @abstractmethod
    def segment(
        self,
        bgr_img: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        detected_objects: Optional[List[Dict[str, Any]]] = None,
        frame_idx: int = 0
    ) -> FloodSegmentationResult:
        """
        Segments water and evaluates inundation evidence for a drone viewpoint.

        Args:
            bgr_img: [H, W, 3] BGR input drone image.
            depth_map: Optional [H, W] metric depth map from VGGT / DA2.
            detected_objects: Optional list of 2D detected objects (structures, vehicles, people).
            frame_idx: Frame index for temporal sequence tracking.

        Returns:
            FloodSegmentationResult object.
        """
        pass

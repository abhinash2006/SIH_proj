"""
Validated Rescue Target Data Structure and Enforcement Layer for UAV Search & Rescue.

Directive 2:
Only ValidatedRescueTarget objects may generate rescue markers in 3D visualizations and operational triage.
Raw YOLO detections alone must NEVER create a rescue marker.

Mandatory 17 fields:
- target_id: Unique SAR identifier (e.g. TARGET_01)
- track_id: Persistent track identifier across frames
- person_confidence: Detection confidence [0.0 - 1.0]
- source_frame_ids: List of video/keyframe indices where target was observed
- bbox_2d: Representative 2D bounding box [x1, y1, x2, y2]
- center_2d: Representative 2D center [u, v]
- location_3d: Reconstructed 3D coordinates [X, Y, Z] in local coordinate frame
- localization_status: "LOCALIZED_3D", "LOCALIZED_2D_PROJECTED", "UNLOCALIZED"
- reprojection_error: Mean reprojection error in pixels (or None)
- flood_proximity: "HIGH", "MEDIUM", "LOW", "SURROUNDED", "NONE"
- surrounding_flood_ratio: Surrounding water percentage [0.0 - 100.0] or ratio [0.0 - 1.0]
- elevation_context: "ELEVATED_STRUCTURE", "GROUND_LEVEL", "UNKNOWN"
- accessible_ground_score: Dry ground egress score [0.0 - 1.0] (low means trapped)
- isolation_score: Spatial isolation score [0.0 - 1.0] (high means stranded)
- rescue_priority: "HIGH", "MEDIUM", "LOW"
- reason: Comprehensive explainable rationale for priority assignment
- evidence: Dictionary containing detailed empirical sensor & geometric metrics
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union


@dataclass
class ValidatedRescueTarget:
    """
    Dedicated validated human rescue target for drone disaster SAR triage.
    Enforces the complete set of 17 physical and algorithmic evidence fields.
    """
    target_id: str
    track_id: Union[int, str]
    person_confidence: float
    source_frame_ids: List[Union[int, str]]
    bbox_2d: List[float]
    center_2d: List[float]
    location_3d: List[float]
    localization_status: str
    reprojection_error: Optional[float]
    flood_proximity: str
    surrounding_flood_ratio: float
    elevation_context: str
    accessible_ground_score: float
    isolation_score: float
    rescue_priority: str
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate priority enumeration
        valid_priorities = {"HIGH", "MEDIUM", "LOW"}
        if self.rescue_priority not in valid_priorities:
            self.rescue_priority = "LOW"

        # Validate 3D location length
        if len(self.location_3d) != 3:
            raise ValueError(f"location_3d must have exactly 3 coordinates [x, y, z], got {self.location_3d}")

    # Dict-like access for backward compatibility
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Converts to serializable dictionary."""
        return {
            "target_id": self.target_id,
            "track_id": self.track_id,
            "person_confidence": round(float(self.person_confidence), 3),
            "source_frame_ids": self.source_frame_ids,
            "bbox_2d": [round(float(v), 2) for v in self.bbox_2d],
            "center_2d": [round(float(v), 2) for v in self.center_2d],
            "location_3d": [round(float(v), 3) for v in self.location_3d],
            "localization_status": self.localization_status,
            "reprojection_error": round(float(self.reprojection_error), 2) if self.reprojection_error is not None else None,
            "flood_proximity": self.flood_proximity,
            "surrounding_flood_ratio": round(float(self.surrounding_flood_ratio), 3),
            "elevation_context": self.elevation_context,
            "accessible_ground_score": round(float(self.accessible_ground_score), 3),
            "isolation_score": round(float(self.isolation_score), 3),
            "rescue_priority": self.rescue_priority,
            "reason": self.reason,
            "evidence": self.evidence
        }

    def format_evidence_card(self) -> str:
        """
        Formats evidence according to Directive 6:
        PERSON #01
        Detection confidence: 92%
        Frames observed: 8
        Flood proximity: HIGH
        Surrounding water: 87%
        Accessible ground: LOW
        Elevation: HIGH
        3D localization: VALID
        Reprojection: 9.2 px
        Priority: HIGH
        Reason: "Person detected on elevated structure surrounded by floodwater with limited apparent access."
        """
        obs_count = len(self.source_frame_ids)
        reproj_str = f"{self.reprojection_error:.1f} px" if self.reprojection_error is not None else "N/A"
        ground_str = "LOW" if self.accessible_ground_score < 0.35 else ("MEDIUM" if self.accessible_ground_score < 0.70 else "HIGH")
        elev_str = "HIGH" if "ELEVATED" in self.elevation_context else ("LOW" if "GROUND" in self.elevation_context else "UNKNOWN")
        loc_str = "VALID" if "LOCALIZED" in self.localization_status else "UNCERTAIN"
        flood_pct = self.surrounding_flood_ratio * 100.0 if self.surrounding_flood_ratio <= 1.0 else self.surrounding_flood_ratio

        return f"""### {self.target_id.upper()} (Track #{self.track_id})

Detection confidence: {self.person_confidence * 100.0:.0f}%
Frames observed: {obs_count}
Flood proximity: {self.flood_proximity}
Surrounding water: {flood_pct:.0f}%
Accessible ground: {ground_str} ({self.accessible_ground_score:.2f})
Elevation: {elev_str} ({self.elevation_context})
3D localization: {loc_str} (Status: {self.localization_status})
Reprojection: {reproj_str}

Priority: {self.rescue_priority}

Reason:
"{self.reason}"
"""


def is_validated_rescue_target(obj: Any) -> bool:
    """
    Validation gate: Returns True ONLY if obj is a ValidatedRescueTarget or
    has been formally verified through the SAR validation pipeline with required fields.
    Raw YOLO detections will return False.
    """
    if isinstance(obj, ValidatedRescueTarget):
        return True
    if isinstance(obj, dict):
        required = [
            "target_id", "track_id", "person_confidence", "source_frame_ids",
            "bbox_2d", "center_2d", "location_3d", "localization_status",
            "rescue_priority", "reason"
        ]
        return all(k in obj for k in required) and obj.get("is_validated_rescue_target", False)
    return False

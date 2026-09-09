"""
Multi-Frame Person Fusion and Track Evidence Aggregator for UAV SAR.

Directive 7:
Do not make rescue decisions from a single frame.
Aggregate evidence across the person's entire track.
Store:
- observation count
- mean confidence
- maximum confidence
- persistence
- 3D consistency
- flood-context consistency

A person consistently detected across multiple frames has significantly stronger
evidence than a one-frame transient detection.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

from src.rescue_target import ValidatedRescueTarget
from src.rescue_risk import RescueRiskEngine

logger = logging.getLogger("PersonFusion")


class MultiFramePersonFuser:
    """
    Fuses multi-view and multi-frame person detections into robust,
    multi-frame validated rescue targets.
    """

    @classmethod
    def fuse_person_tracks(
        cls,
        person_detections: List[Dict[str, Any]],
        total_selected_frames: int = 1,
        flood_evaluations: Optional[List[Dict[str, Any]]] = None
    ) -> List[ValidatedRescueTarget]:
        """
        Aggregates person detection records across frames into ValidatedRescueTarget instances.

        Args:
            person_detections: List of detection dictionaries from each frame containing:
                - track_id or id
                - frame_idx, frame_id
                - confidence
                - bbox [x1, y1, x2, y2]
                - center [u, v]
                - point_3d (optional [x, y, z])
                - reprojection_error_px (optional)
                - spatial_context (optional dict)
            total_selected_frames: Total number of frames in the active reconstruction sequence.
            flood_evaluations: Optional list of FloodSegmentationResult for each frame.

        Returns:
            List of ValidatedRescueTarget objects.
        """
        if not person_detections:
            return []

        # 1. Group detections by persistent track identifier
        tracks_map: Dict[Union[int, str], List[Dict[str, Any]]] = {}
        for det in person_detections:
            trk_id = det.get("track_id")
            if trk_id is None:
                trk_id = det.get("id", f"UNTRACKED_{det.get('frame_idx', 0)}")
            
            # Normalize track ID string/int representation
            trk_key = str(trk_id)
            if trk_key not in tracks_map:
                tracks_map[trk_key] = []
            tracks_map[trk_key].append(det)

        validated_targets: List[ValidatedRescueTarget] = []
        target_idx = 1

        for trk_key, obs_list in tracks_map.items():
            obs_count = len(obs_list)
            confidences = [float(d.get("confidence", d.get("detection_confidence", 0.5))) for d in obs_list]
            mean_conf = float(np.mean(confidences))
            max_conf = float(np.max(confidences))

            frame_indices = sorted(list(set(int(d.get("frame_idx", 0)) for d in obs_list)))
            frame_span = max(frame_indices) - min(frame_indices) + 1
            persistence = round(float(obs_count / max(1, frame_span)), 3)

            # Extract 3D points
            valid_3d_points = []
            reproj_errors = []
            for d in obs_list:
                p3d = d.get("point_3d")
                if p3d is not None and len(p3d) == 3:
                    valid_3d_points.append([float(p3d[0]), float(p3d[1]), float(p3d[2])])
                err = d.get("reprojection_error_px")
                if err is not None:
                    reproj_errors.append(float(err))

            mean_reproj_err = round(float(np.mean(reproj_errors)), 2) if reproj_errors else None

            # Evaluate 3D spatial consistency and reprojection gating
            if len(valid_3d_points) >= 2:
                pts_arr = np.array(valid_3d_points, dtype=np.float32)
                std_xyz = np.std(pts_arr, axis=0)
                dispersion_3d = float(np.linalg.norm(std_xyz))
                consistency_3d_score = float(np.clip(1.0 - (dispersion_3d / 3.0), 0.1, 1.0))
                median_3d = np.median(pts_arr, axis=0).tolist()
                loc_status = "LOCALIZED_3D" if (mean_reproj_err is None or mean_reproj_err <= 25.0) else "REPROJECT_REJECTED"
            elif len(valid_3d_points) == 1:
                dispersion_3d = 0.0
                consistency_3d_score = 0.75
                median_3d = valid_3d_points[0]
                loc_status = "LOCALIZED_3D" if (mean_reproj_err is None or mean_reproj_err <= 25.0) else "REPROJECT_REJECTED"
            else:
                dispersion_3d = 999.0
                consistency_3d_score = 0.10
                median_3d = [0.0, 0.0, 0.0]
                loc_status = "UNLOCALIZED"

            # Spatial flood context aggregation
            flood_pcts = []
            isolation_scores = []
            elevation_types = []
            flood_proximities = []

            for d in obs_list:
                ctx = d.get("spatial_context") or {}
                flood_pcts.append(float(ctx.get("surrounded_percentage", ctx.get("surrounding_flood_pct", 0.0))))
                isolation_scores.append(float(ctx.get("isolation_score", 0.0)))
                elevation_types.append(ctx.get("elevation_context", ctx.get("rooftop_elevation_context", "GROUND_LEVEL")))
                flood_proximities.append(ctx.get("flood_proximity", "LOW"))

            mean_flood_pct = float(np.mean(flood_pcts)) if flood_pcts else 0.0
            mean_isolation = float(np.mean(isolation_scores)) if isolation_scores else 0.0
            flood_std = float(np.std(flood_pcts)) if len(flood_pcts) > 1 else 0.0
            flood_context_consistency = round(float(np.clip(1.0 - (flood_std / 50.0), 0.2, 1.0)), 3)

            # Dominant elevation & proximity
            dom_elevation = max(set(elevation_types), key=elevation_types.count) if elevation_types else "GROUND_LEVEL"
            dom_proximity = max(set(flood_proximities), key=flood_proximities.count) if flood_proximities else ("HIGH" if mean_flood_pct > 30.0 else "LOW")
            accessible_ground_score = round(float(np.clip(1.0 - mean_isolation, 0.0, 1.0)), 3)

            # Representative frame detection (closest to median 3D or highest confidence)
            best_det = max(obs_list, key=lambda d: float(d.get("confidence", d.get("detection_confidence", 0.0))))
            bbox_2d = [float(v) for v in best_det.get("bbox", [0, 0, 0, 0])]
            center_2d = [float(v) for v in best_det.get("center", [(bbox_2d[0]+bbox_2d[2])/2.0, (bbox_2d[1]+bbox_2d[3])/2.0])]

            # SAR Priority calculation via RescueRiskEngine
            person_ctx = {
                "classification": "POTENTIAL_STRANDED_PERSON" if (mean_flood_pct > 25.0 or mean_isolation > 0.5) else "NORMAL_PERSON",
                "isolation_score": mean_isolation,
                "surrounding_flood_pct": mean_flood_pct,
                "rooftop_elevation_context": dom_elevation
            }
            risk_eval = RescueRiskEngine.calculate_person_rescue_priority(
                person_context=person_ctx,
                detection_conf=mean_conf,
                location_status=loc_status,
                reprojection_error_px=mean_reproj_err,
                observation_count=obs_count
            )

            priority = risk_eval["rescue_priority"]

            # Construct comprehensive evidence-based reason (Directive 6)
            if priority == "HIGH":
                elev_desc = "elevated structure" if "ELEVATED" in dom_elevation else "isolated high ground"
                reason_str = f"Person observed across {obs_count} frame(s) on {elev_desc} surrounded by {mean_flood_pct:.0f}% floodwater with limited apparent egress."
            elif priority == "MEDIUM":
                reason_str = f"Person detected in proximity to active flood boundary ({mean_flood_pct:.0f}% surrounding water, isolation {mean_isolation:.2f}). Evacuation monitoring recommended."
            else:
                reason_str = f"Person located on accessible terrain ({accessible_ground_score*100:.0f}% dry ground egress) with minimal flood hazard."

            if loc_status == "LOCALIZED_3D" and priority in ["HIGH", "MEDIUM"]:
                detection_state = "VALIDATED RESCUE TARGET"
            elif priority in ["HIGH", "MEDIUM"] or mean_flood_pct > 25.0:
                detection_state = "POTENTIAL STRANDED PERSON"
            elif loc_status == "LOCALIZED_3D":
                detection_state = "PERSON 3D LOCALIZED"
            elif len(flood_pcts) > 0:
                detection_state = "PERSON CONTEXTUALIZED"
            elif obs_count > 1:
                detection_state = "PERSON TRACKED"
            else:
                detection_state = "PERSON DETECTED"

            evidence_dict = {
                "observation_count": obs_count,
                "mean_confidence": round(mean_conf, 3),
                "max_confidence": round(max_conf, 3),
                "persistence": persistence,
                "dispersion_3d_m": round(dispersion_3d, 3) if dispersion_3d < 100 else None,
                "consistency_3d_score": round(consistency_3d_score, 3),
                "flood_context_consistency": flood_context_consistency,
                "reprojection_error_px": mean_reproj_err,
                "source_frames": frame_indices,
                "elevation_context": dom_elevation,
                "detection_state": detection_state,
                "target_classification": "POTENTIAL STRANDED PERSON" if priority in ["HIGH", "MEDIUM"] else "NORMAL PERSON"
            }

            target = ValidatedRescueTarget(
                target_id=f"PERSON #{target_idx:02d}",
                track_id=trk_key,
                person_confidence=round(mean_conf, 3),
                source_frame_ids=frame_indices,
                bbox_2d=bbox_2d,
                center_2d=center_2d,
                location_3d=median_3d,
                localization_status=loc_status,
                reprojection_error=mean_reproj_err,
                flood_proximity=dom_proximity,
                surrounding_flood_ratio=round(mean_flood_pct / 100.0, 3),
                elevation_context=dom_elevation,
                accessible_ground_score=accessible_ground_score,
                isolation_score=round(mean_isolation, 3),
                rescue_priority=priority,
                reason=reason_str,
                evidence=evidence_dict
            )

            validated_targets.append(target)
            target_idx += 1

        # Sort targets by priority (HIGH -> MEDIUM -> LOW), then by isolation score descending
        prio_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        validated_targets.sort(key=lambda t: (prio_order.get(t.rescue_priority, 3), -t.isolation_score))

        logger.info(f"[PERSON FUSION] Fused {len(person_detections)} detections into {len(validated_targets)} validated rescue targets.")
        return validated_targets

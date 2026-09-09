import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("RescueRiskEngine")

class RescueRiskEngine:
    """
    Explainable SAR (Search and Rescue) Priority & Scene Risk Engine.
    
    1. Evaluates individual victim rescue priority: HIGH, MEDIUM, LOW based on:
       - Detection confidence
       - Flood proximity
       - Spatial isolation & accessibility
       - 3D localization validity & reprojection consistency
       - Multi-frame observation persistence
       
    2. Evaluates scene disaster risk rating: CRITICAL, HIGH, MEDIUM, LOW.
       - Decouples disaster hazard from victim count (e.g. validated flood with 0 people is still HIGH disaster risk).
    """

    @staticmethod
    def calculate_person_rescue_priority(
        person_context: Dict[str, Any],
        detection_conf: float,
        location_status: str,
        reprojection_error_px: float = None,
        observation_count: int = 1
    ) -> Dict[str, Any]:
        """
        Calculates an explainable rescue priority for a single detected person.
        """
        classification = person_context.get("classification", "NORMAL_PERSON")
        isolation_score = float(person_context.get("isolation_score", 0.0))
        surrounding_flood = float(person_context.get("surrounding_flood_pct", 0.0))
        is_elevated = person_context.get("rooftop_elevation_context") == "ELEVATED_STRUCTURE"

        # Localization quality score [0.0 - 1.0]
        if location_status == "LOCALIZED_3D":
            if reprojection_error_px is not None:
                loc_qual = float(max(0.20, 1.0 - (reprojection_error_px / 40.0)))
            else:
                loc_qual = 0.85
        else:
            loc_qual = 0.30

        # Multi-frame persistence boost
        persistence_score = min(1.0, 0.60 + (observation_count * 0.10))

        # Composite numerical rescue score [0.0 - 1.0]
        rescue_score = (
            0.35 * isolation_score +
            0.25 * (surrounding_flood / 100.0) +
            0.15 * detection_conf +
            0.15 * loc_qual +
            0.10 * persistence_score
        )
        rescue_score = round(float(rescue_score), 3)

        # Categorize into HIGH / MEDIUM / LOW
        if classification == "POTENTIAL_STRANDED_PERSON" or (rescue_score >= 0.55 and surrounding_flood > 20.0) or (surrounding_flood > 40.0 and is_elevated):
            priority = "HIGH"
            action = "Urgent aerial/boat rescue required. Target is stranded/isolated by floodwater."
        elif classification == "POTENTIAL_ISOLATED_PERSON" or (rescue_score >= 0.40 and surrounding_flood > 10.0) or surrounding_flood > 15.0:
            priority = "MEDIUM"
            action = "Prioritize monitoring and prepare evacuation route. Person in proximity to advancing water."
        else:
            priority = "LOW"
            action = "Routine monitoring. Person is located on accessible dry ground."

        explanation = (
            f"Priority {priority} (Score: {rescue_score:.2f}) | "
            f"Detection Conf: {detection_conf:.2f} | "
            f"Isolation: {isolation_score:.2f} | "
            f"Surrounding Flood: {surrounding_flood:.0f}% | "
            f"3D Status: {location_status}"
        )

        return {
            "rescue_priority": priority,
            "rescue_score": rescue_score,
            "recommended_action": action,
            "priority_explanation": explanation,
            "target_classification": classification
        }

    @staticmethod
    def calculate_scene_risk(
        validated_incidents: List[Dict[str, Any]],
        flood_analysis_results: List[Dict[str, Any]],
        tracked_people: List[Dict[str, Any]]
    ) -> Tuple[str, float, List[str], List[str]]:
        """
        Evaluates overall scene disaster risk.
        
        Returns:
            risk_level: "CRITICAL", "HIGH", "MEDIUM", or "LOW / NO DISASTER EVIDENCE"
            risk_score: float [0 - 100]
            risk_reasons: list of key evidence bullets
            rescue_priorities: list of actionable rescue priority strings
        """
        reasons = []
        priorities = []

        # Check flood evidence across all frames
        has_validated_inundation = any(
            f.get("water_state") == "VALIDATED_INUNDATION" for f in flood_analysis_results
        )
        has_potential_inundation = any(
            f.get("water_state") == "POTENTIAL_INUNDATION" for f in flood_analysis_results
        )
        max_flood_ratio = max([f.get("validated_ratio", 0.0) for f in flood_analysis_results] + [0.0])

        # Check people and rescue priorities
        high_priority_targets = [
            p for p in tracked_people if p.get("rescue_priority") == "HIGH"
        ]
        medium_priority_targets = [
            p for p in tracked_people if p.get("rescue_priority") == "MEDIUM"
        ]
        low_priority_targets = [
            p for p in tracked_people if p.get("rescue_priority") == "LOW"
        ]

        base_score = 0.0

        if has_validated_inundation:
            base_score += 45.0 + min(30.0, max_flood_ratio * 100.0)
            reasons.append(f"Validated flood inundation confirmed covering {max_flood_ratio*100:.1f}% of aerial scene.")
        elif has_potential_inundation:
            base_score += 25.0 + min(15.0, max_flood_ratio * 100.0)
            reasons.append(f"Potential flood inundation detected covering {max_flood_ratio*100:.1f}% of scene.")
        else:
            reasons.append("No active flood inundation evidence detected across analyzed frames.")

        if len(high_priority_targets) > 0:
            base_score += 25.0 + min(15.0, len(high_priority_targets) * 5.0)
            reasons.append(f"CRITICAL SAR: {len(high_priority_targets)} potentially stranded person(s) isolated by floodwater.")
            priorities.append(f"Dispatch rescue team to {len(high_priority_targets)} high-priority stranded victim location(s).")

        if len(medium_priority_targets) > 0:
            base_score += 10.0 + min(10.0, len(medium_priority_targets) * 3.0)
            reasons.append(f"Monitoring: {len(medium_priority_targets)} person(s) in proximity to active flood boundary.")
            priorities.append(f"Establish evacuation route for {len(medium_priority_targets)} person(s) near flood perimeter.")

        if len(tracked_people) == 0:
            reasons.append("Zero persons detected in available aerial frames (continuous optical search active).")
            if has_validated_inundation:
                priorities.append("Area inundation response active; continue aerial search for potential submerged victims.")
            else:
                priorities.append("Routine aerial patrol. No emergency rescue operations required.")

        risk_score = round(min(100.0, max(0.0, base_score)), 1)

        if len(high_priority_targets) > 0 and has_validated_inundation:
            risk_level = "CRITICAL"
        elif has_validated_inundation or len(high_priority_targets) > 0:
            risk_level = "HIGH"
        elif has_potential_inundation or len(medium_priority_targets) > 0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW / NO DISASTER EVIDENCE"
            risk_score = 0.0

        return risk_level, risk_score, reasons, priorities

    @classmethod
    def score_rescue_priorities(
        cls,
        people_with_context: List[Dict[str, Any]],
        scene_flood_ratio: float = 0.0,
        is_active_flood: bool = False
    ) -> Dict[str, Any]:
        """
        Unified SAR ranking pipeline. Evaluates every detected victim, ranks them,
        and determines the decoupled scene disaster risk.
        """
        ranked = []
        for p in people_with_context:
            ctx = p.get("spatial_context") or {}
            det_conf = float(p.get("detection_confidence", 0.8))
            pos_3d = p.get("point_3d")
            loc_status = "LOCALIZED_3D" if pos_3d is not None else "UNLOCALIZED"
            reproj_err = p.get("reprojection_error_px")

            p_ctx = {
                "classification": ctx.get("isolation_type", "NORMAL_PERSON"),
                "isolation_score": float(ctx.get("surrounded_percentage", 0.0) / 100.0),
                "surrounding_flood_pct": float(ctx.get("surrounded_percentage", 0.0)),
                "rooftop_elevation_context": "ELEVATED_STRUCTURE" if ctx.get("elevation_relative_m", 0.0) > 1.5 else "GROUND_LEVEL"
            }

            p_eval = cls.calculate_person_rescue_priority(
                person_context=p_ctx,
                detection_conf=det_conf,
                location_status=loc_status,
                reprojection_error_px=reproj_err
            )

            # Format world position dict
            pos_dict = None
            if pos_3d is not None:
                pos_dict = {"x": round(float(pos_3d[0]), 2), "y": round(float(pos_3d[1]), 2), "z": round(float(pos_3d[2]), 2)}

            ranked.append({
                "victim_id": str(p.get("track_id", f"VICTIM_{len(ranked)+1:02d}")),
                "track_id": str(p.get("track_id", "")),
                "frame_id": str(p.get("frame_id", "frame_0000.jpg")),
                "frame_idx": int(p.get("frame_idx", 0)),
                "rescue_priority": p_eval["rescue_priority"],
                "priority_score": round(p_eval["rescue_score"] * 100.0, 1),
                "rationale": p_eval["recommended_action"],
                "explanation": p_eval["priority_explanation"],
                "world_position": pos_dict,
                "spatial_context": ctx
            })

        # Sort descending by priority_score
        ranked.sort(key=lambda x: x["priority_score"], reverse=True)

        # Decoupled scene disaster risk
        high_targets = [v for v in ranked if v["rescue_priority"] == "HIGH"]
        med_targets = [v for v in ranked if v["rescue_priority"] == "MEDIUM"]

        reasons = []
        if is_active_flood or scene_flood_ratio >= 0.15:
            base_score = 50.0 + min(40.0, scene_flood_ratio * 100.0)
            reasons.append(f"Validated flood inundation confirmed covering {scene_flood_ratio*100:.1f}% of aerial scene.")
            if high_targets:
                scene_level = "CRITICAL"
                base_score = min(100.0, base_score + 20.0)
                reasons.append(f"CRITICAL SAR: {len(high_targets)} stranded victim(s) isolated by floodwaters.")
            else:
                scene_level = "HIGH"
                if not people_with_context:
                    reasons.append("Zero persons detected in available aerial frames (continuous optical search active).")
        elif scene_flood_ratio >= 0.04:
            scene_level = "MEDIUM"
            base_score = 35.0
            reasons.append(f"Potential flood inundation detected covering {scene_flood_ratio*100:.1f}% of scene.")
        else:
            scene_level = "LOW"
            base_score = 10.0
            reasons.append("Routine aerial monitoring. No active flood inundation evidence detected.")

        return {
            "ranked_priorities": ranked,
            "scene_risk": {
                "risk_level": scene_level,
                "risk_score": round(base_score, 1),
                "reasons": reasons
            }
        }


RescueRiskScorer = RescueRiskEngine

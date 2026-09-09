from typing import List, Dict, Any, Tuple

class DisasterRiskScorer:
    """
    Computes an interpretable, transparent risk score and rescue priority rating for analyzed drone scenes.
    Risk Levels: LOW, MEDIUM, HIGH, CRITICAL
    """

    @staticmethod
    def calculate_scene_risk(
        detected_objects: List[Dict[str, Any]],
        hazards: List[Dict[str, Any]],
        damage_incidents: List[Dict[str, Any]]
    ) -> Tuple[str, float, List[str], List[str]]:
        """
        Calculates total risk score and prioritizes rescue inspection areas.
        Returns:
            risk_level: str ("CRITICAL", "HIGH", "MEDIUM", "LOW")
            numeric_score: float in [0.0, 100.0]
            risk_reasons: List[str]
            rescue_priorities: List[str]
        """
        score = 0.0
        reasons = []
        priorities = []

        # Check if active disaster hazards exist
        has_active_disaster = len(hazards) > 0 or len(damage_incidents) > 0

        # 1. Person Detection (Normal activity vs Disaster Exposure)
        people_count = sum(1 for d in detected_objects if d.get("class_name") in ["person", "PERSON_DETECTED"])
        if people_count > 0:
            if has_active_disaster:
                pts = min(40.0, people_count * 20.0)
                score += pts
                reasons.append(f"+ {people_count} Person(s) detected in active disaster zone (+{pts:.0f} pts)")
                priorities.append(f"Priority 1: Immediate life safety inspection for {people_count} person(s) exposed to disaster.")
            else:
                reasons.append(f"Information: {people_count} Person(s) detected in normal environment (No emergency exposure).")
        else:
            reasons.append("No person detected in analyzed drone imagery.")

        # 2. Flood / Water Hazard (+50 pts max for severe flood inundation)
        flood_hazards = [h for h in hazards if h.get("incident_type") == "FLOOD_ZONE"]
        if flood_hazards:
            max_cov = max(h.get("coverage_pct", 10.0) for h in flood_hazards)
            pts = min(50.0, 25.0 + max_cov * 0.6)
            score += pts
            reasons.append(f"+ Active flood water coverage detected ({max_cov:.1f}% area) (+{pts:.0f} pts)")
            priorities.append("Priority 2: Water rescue & flood-isolated zone inspection.")

        # 3. Structural Building Damage (+25 pts)
        damaged_bldgs = [d for d in damage_incidents if d.get("severity") in ["HIGH", "CRITICAL"]]
        if damaged_bldgs:
            score += 25.0
            reasons.append(f"+ Visual structural damage / roof disruption detected on {len(damaged_bldgs)} building(s) (+25 pts)")
            priorities.append("Priority 3: Structural collapse and building damage verification.")

        # 4. Blocked Access Routes / Debris (+15 pts)
        debris_hazards = [h for h in hazards if h.get("incident_type") == "BLOCKED_ROAD_DEBRIS"]
        if debris_hazards:
            score += 15.0
            reasons.append(f"+ Ground route obstruction / debris blockage detected (+15 pts)")
            priorities.append("Priority 4: Debris clearing for emergency responder access.")

        if not has_active_disaster:
            score = 0.0
            risk_level = "LOW / NO DISASTER EVIDENCE"
            priorities = ["Routine monitoring. No emergency rescue action required."]
        else:
            if score >= 75.0:
                risk_level = "CRITICAL"
            elif score >= 45.0:
                risk_level = "HIGH"
            elif score >= 20.0:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

        return risk_level, min(100.0, round(score, 1)), reasons, priorities

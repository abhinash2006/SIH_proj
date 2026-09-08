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

        # 1. Person Detection (+35 pts per visible person)
        people_count = sum(1 for d in detected_objects if d.get("class_name") == "person")
        if people_count > 0:
            pts = min(45.0, people_count * 30.0)
            score += pts
            reasons.append(f"+ {people_count} Person(s) detected in aerial imagery (+{pts:.0f} pts)")
            priorities.append(f"Priority 1: Immediate life safety inspection for {people_count} detected person(s).")
        else:
            reasons.append("No person detected in analyzed drone imagery.")

        # 2. Flood / Water Hazard (+30 pts)
        flood_hazards = [h for h in hazards if h.get("incident_type") == "FLOOD_ZONE"]
        if flood_hazards:
            max_cov = max(h.get("coverage_pct", 10.0) for h in flood_hazards)
            pts = min(35.0, 20.0 + max_cov * 0.5)
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

        # Determine Risk Level
        if score >= 65.0:
            risk_level = "CRITICAL"
        elif score >= 40.0:
            risk_level = "HIGH"
        elif score >= 20.0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        if not priorities:
            priorities.append("Priority 5: Standard routine aerial survey monitoring.")

        return risk_level, min(100.0, round(score, 1)), reasons, priorities

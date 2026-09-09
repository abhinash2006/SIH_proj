import os
import json
from pathlib import Path
from typing import Dict, Any, List

class DisasterReportGenerator:
    """
    Generates interactive HTML and JSON disaster inspection reports for field command decision support.
    """

    @staticmethod
    def generate_html_report(
        summary_data: Dict[str, Any],
        incidents: List[Dict[str, Any]],
        output_path: str = "outputs/disaster_report.html"
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        risk_level = summary_data.get("risk_level", "MEDIUM")
        risk_color = {
            "CRITICAL": "#dc2626",
            "HIGH": "#ea580c",
            "MEDIUM": "#d97706",
            "LOW": "#16a34a"
        }.get(risk_level, "#2563eb")

        incidents_html = ""
        for inc in incidents:
            if inc.get("x_m") is not None and inc.get("location_status") != "UNLOCALIZED":
                reproj_info = f" (reproj: {inc['reprojection_error_px']}px)" if inc.get("reprojection_error_px") is not None else ""
                loc_str = f"X: {inc['x_m']:.2f}, Y: {inc['y_m']:.2f}, Z: {inc['z_m']:.2f} [VGGT RELATIVE SCALE]{reproj_info}"
            else:
                loc_str = "UNLOCALIZED (Insufficient Depth / Reprojection Fail)"
            
            src_frames = inc.get("source_frames", [inc.get("frame_idx", 0)])
            incidents_html += f"""
            <tr style="border-bottom: 1px solid #e5e7eb;">
                <td style="padding: 10px; font-weight: bold;">{inc.get('incident_id')}</td>
                <td style="padding: 10px;">{inc.get('incident_type')}</td>
                <td style="padding: 10px;"><span style="background-color: #fee2e2; color: #991b1b; padding: 3px 8px; border-radius: 4px;">{inc.get('severity')}</span></td>
                <td style="padding: 10px;">{inc.get('confidence', 0.0):.2f}</td>
                <td style="padding: 10px;">{loc_str}</td>
                <td style="padding: 10px;">Frames {src_frames}</td>
                <td style="padding: 10px;">{inc.get('evidence')}</td>
                <td style="padding: 10px; font-weight: bold; color: #1d4ed8;">{inc.get('status')}</td>
            </tr>
            """

        reasons_html = "".join([f"<li>{r}</li>" for r in summary_data.get("risk_reasons", [])])
        priorities_html = "".join([f"<li><strong>{p}</strong></li>" for p in summary_data.get("rescue_priorities", [])])

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Drone-VGGT AI Disaster Inspection Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8fafc; color: #1e293b; margin: 0; padding: 30px; }}
        .header {{ background-color: #0f172a; color: white; padding: 25px; border-radius: 10px; margin-bottom: 25px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 25px; }}
        .risk-badge {{ display: inline-block; background: {risk_color}; color: white; padding: 6px 16px; border-radius: 20px; font-weight: bold; font-size: 1.1rem; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th {{ background: #f1f5f9; text-align: left; padding: 12px; border-bottom: 2px solid #cbd5e1; }}
        .disclaimer {{ background: #fffbebfb; border-left: 4px solid #f59e0b; padding: 15px; margin-top: 20px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛸 Drone-VGGT: AI Disaster Inspection & 3D Rescue Mapping Report</h1>
        <p>Automated 3D Reconstruction & Spatial Hazard Assessment for Emergency Responders</p>
    </div>

    <div class="card">
        <h2>Executive Scene Assessment</h2>
        <p>Overall Rescue Risk Rating: <span class="risk-badge">{risk_level}</span> (Risk Score: {summary_data.get('risk_score', 0.0)} / 100)</p>
        
        <h3>Key Risk Factors:</h3>
        <ul>{reasons_html}</ul>

        <h3>Actionable Rescue Priorities (Decision Support):</h3>
        <ol>{priorities_html}</ol>
    </div>

    <div class="card">
        <h2>Reconstruction & Detection Statistics</h2>
        <ul>
            <li><strong>Total Drone Input Frames</strong>: {summary_data.get('total_frames', 0)}</li>
            <li><strong>Selected Geometrically Optimal Frames</strong>: {summary_data.get('selected_frames', 0)}</li>
            <li><strong>Reconstructed 3D Points</strong>: {summary_data.get('reconstructed_points', 0):,}</li>
            <li><strong>Detected People</strong>: {summary_data.get('detected_people', 0)}</li>
            <li><strong>Detected Vehicles</strong>: {summary_data.get('detected_vehicles', 0)}</li>
            <li><strong>Identified Hazards / Alerts</strong>: {len(incidents)}</li>
        </ul>
    </div>

    <div class="card">
        <h2>Disaster Incidents & Spatial Hazards Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Incident ID</th>
                    <th>Type</th>
                    <th>Severity</th>
                    <th>Confidence</th>
                    <th>3D Location (Local Metric)</th>
                    <th>Source Frames</th>
                    <th>Evidence Note</th>
                    <th>Verification Status</th>
                </tr>
            </thead>
            <tbody>
                {incidents_html}
            </tbody>
        </table>
    </div>

    <div class="disclaimer">
        <strong>IMPORTANT DISASTER RESPONSE DISCLAIMER:</strong>
        This AI report provides decision support and spatial intelligence compiled from aerial drone imagery. All AI-flagged incidents, structural damage predictions, and person detections require human responder field verification before initiating tactical search-and-rescue operations.
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return str(output_path)

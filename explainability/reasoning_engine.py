"""
explainability/reasoning_engine.py
==================================
Natural Language Reasoner for NeuroDrive-XAI.

Converts structured data (Perception, Planning, Control, SHAP) 
into human-readable English justifications.
"""

import numpy as np

class ReasoningEngine:
    def __init__(self):
        self.feature_map = {
            "speed_mps": "current speed",
            "long_accel": "acceleration",
            "curvature": "road curvature",
            "prev_steering": "previous steering direction",
            "cte": "distance from lane center",
            "heading_err": "heading misalignment",
            "closest_dist": "proximity to obstacle",
            "closest_v_rel": "relative speed of the obstacle",
            "num_agents": "presence of other vehicles"
        }

    def generate_justification(self, scene, decision, control_xai=None):
        """
        Creates a natural language string explaining the current action.
        """
        action = decision.get("action", "cruise")
        reason_meta = decision.get("reason", "normal conditions")
        risk_score = decision.get("risk_score", 0.0)
        
        # 1. Action Summary
        text = f"DECISION: System is choosing to {action.upper()} because {reason_meta}. "
        
        # 2. Risk Context
        if risk_score > 0.5:
            text += f"CRITICAL: Risk level is HIGH ({risk_score:.2f}). "
        elif risk_score > 0.2:
            text += f"NOTE: Moderate risk detected ({risk_score:.2f}). "
            
        # 3. Decision Drivers (SHAP Analysis)
        if control_xai and "contributions" in control_xai:
            # Look at brake/throttle contributions
            relevant_out = "brake" if action == "Brake" else "throttle"
            contribs = next((c for c in control_xai["contributions"] if c["output"] == relevant_out), None)
            
            if contribs:
                top_features = [f for f in contribs["ranked_features"][:2] if abs(f["shap_value"]) > 0.05]
                if top_features:
                    drivers = [f"{self.feature_map.get(f['feature'], f['feature'])}" for f in top_features]
                    text += f"The primary drivers for this choice are {', '.join(drivers)}. "
        
        # 4. Perception Context
        objects = scene.get("objects", [])
        if objects:
            closest = min(objects, key=lambda o: o.get("distance_meters", 999))
            dist = closest.get("distance_meters", 0)
            obj_type = closest.get("type", "object")
            if dist < 20:
                text += f"Close {obj_type} detected at {dist:.1f}m. "

        return text.strip()

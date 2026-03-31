import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class DecisionEngine:
    def __init__(self, history_size=5):
        self.history_size = history_size
        self.history = []
        self.model_path = "weights/decision_rf.pkl"
        
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self._train_synthetic_model()
            
    def _train_synthetic_model(self):
        X, y = [], []
        for _ in range(100):
            X.append([np.random.uniform(30, 100), np.random.uniform(0, 30), np.random.uniform(0, 10)])
            y.append(0) # proceed
        for _ in range(100):
            X.append([np.random.uniform(15, 30), np.random.uniform(20, 50), np.random.uniform(0, 50)])
            y.append(1) # slow
        for _ in range(100):
            X.append([np.random.uniform(0, 15), np.random.uniform(0, 100), np.random.uniform(0, 100)])
            y.append(2) # brake
            
        self.model = RandomForestClassifier(n_estimators=10, max_depth=3)
        self.model.fit(X, y)
        os.makedirs("weights", exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def _update_history(self, action):
        self.history.append(action)
        if len(self.history) > self.history_size:
            self.history.pop(0)
            
    def _get_smoothed_action(self, current_action):
        if not self.history: return current_action
        brakes = self.history.count("Brake")
        if brakes >= 2 and current_action == "Proceed":
            return "Slow"
        return current_action

    def decide(self, scene):
        objects = scene.get("objects", [])
        lane_geometry = scene.get("lane_geometry", {})
        confidences = scene.get("confidence", {"lane": 1.0, "detection": 1.0, "depth": 1.0})
        
        # Phase 2 & 3: Uncertainty Modeling & Fallbacks
        
        # 1. Lane missing check
        if not lane_geometry.get("center_line", []):
            fallback_action = "Slow"
            self._update_history(fallback_action)
            return {
                "action": fallback_action,
                "reason": "Lane bounding unstructured -> graceful speed reduction",
                "risk_score": 0.65,
                "history": self.history.copy(),
                "fallback": "reduce_speed"
            }
            
        # 2. Low generic confidence check (Degrade gracefully)
        if confidences.get("detection", 1.0) < 0.6 or confidences.get("lane", 1.0) < 0.6:
            fallback_action = "Slow"
            self._update_history(fallback_action)
            return {
                "action": fallback_action,
                "reason": f"Low perception confidence (Det: {confidences.get('detection', 1.0):.2f}) -> slowing down",
                "risk_score": 0.70,
                "history": self.history.copy(),
                "fallback": "low_confidence_reduce"
            }
            
        # Parse center lane objects
        center_objs = [obj for obj in objects if obj.get("lane") == "center"]
        
        if not center_objs:
            action = "Proceed"
            smoothed = self._get_smoothed_action(action)
            self._update_history(smoothed)
            return {
                "action": smoothed, 
                "reason": "Path clear",
                "risk_score": 0.1,
                "history": self.history.copy(),
                "fallback": False
            }
            
        closest_obj = min(center_objs, key=lambda x: x.get("distance_meters", 100))
        dist = closest_obj.get("distance_meters", 100)
        
        # 3. Depth Unreliability handler
        if confidences.get("depth", 1.0) < 0.5:
             # Fall back to assuming obstacle is critically close if we can't trust depth
             dist = 10.0 # forces a conservative Slow/Brake
             use_rel = True
        else:
             use_rel = False
             
        velocity = closest_obj.get("velocity", 30.0)
        obj_type = closest_obj.get("type", "unknown")
        
        features = np.array([[dist, velocity, 0.0]])
        pred_class = self.model.predict(features)[0]
        risk_proba = self.model.predict_proba(features)[0]
        
        risk_score = round(float(risk_proba[2] * 1.0 + risk_proba[1] * 0.5), 2)
        
        if pred_class == 2:
            raw_action = "Brake"
            reason = f"High risk (Score: {risk_score}): {obj_type} detected {dist}m ahead"
        elif pred_class == 1:
            raw_action = "Slow"
            reason = f"Moderate risk (Score: {risk_score}): {obj_type} detected {dist}m ahead"
        else:
            raw_action = "Proceed"
            reason = f"Safe distance (Score: {risk_score}): {obj_type} is at {dist}m"
            
        if use_rel:
             reason += " [Depth Unreliable -> Defaulted to Conservative]"
             raw_action = "Slow" if raw_action == "Proceed" else raw_action
            
        smoothed_action = self._get_smoothed_action(raw_action)
        self._update_history(smoothed_action)
        
        return {
            "action": smoothed_action, 
            "reason": reason,
            "risk_score": risk_score,
            "history": self.history.copy(),
            "use_relative_distance": use_rel,
            "fallback": False
        }

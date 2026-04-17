import numpy as np
import torch
import time

class NeuroMetrics:
    @staticmethod
    def calc_precision_recall(y_true, y_pred, threshold=0.5):
        yp = (y_pred >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (yp == 1))
        fp = np.sum((y_true == 0) & (yp == 1))
        fn = np.sum((y_true == 1) & (yp == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return precision, recall

    @staticmethod
    def false_brake_rate(y_true, y_pred, threshold=0.5):
        yp = (y_pred >= threshold).astype(int)
        false_brakes = np.sum((y_true == 0) & (yp == 1))
        return false_brakes / len(y_true)

    @staticmethod
    def intervention_accuracy(distance_feature, y_pred, threshold=0.5):
        """
        Intervention Accuracy Metric: 
        Checks if braking action occurs when distance_to_object is fundamentally dangerous (< 15.0m).
        """
        yp = (y_pred >= threshold).astype(int)
        # Assuming index 0 corresponds to distance_to_object before scaling... 
        # If passed raw features, evaluates physical truth:
        dangerous = distance_feature < 15.0
        prevented = np.sum(dangerous & (yp == 1))
        total_danger = np.sum(dangerous)
        return prevented / total_danger if total_danger > 0 else 1.0


class CounterfactualEngine:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.feature_names = [
            'distance_to_object', 'relative_velocity', 'lane_offset', 
            'lane_curvature', 'num_objects', 'closest_object_type'
        ]

    def binary_search_threshold(self, original_features, feature_idx, bounds, target_decision, steps=10):
        """
        Runs simple, deterministic binary search against a single feature to find delta.
        """
        low, high = bounds
        best_val = None
        
        test_features = np.copy(original_features)
        
        for _ in range(steps):
            mid = (low + high) / 2
            test_features[feature_idx] = mid
            
            # Predict
            x_scaled = self.scaler.transform([test_features])
            x_tensor = torch.FloatTensor(x_scaled)
            self.model.eval()
            with torch.no_grad():
                _, brake_prob = self.model(x_tensor)
                
            current_decision = int(brake_prob.item() > 0.5)
            
            if current_decision == target_decision:
                best_val = mid
                # If target is achieved, try to minimize the delta
                if mid > original_features[feature_idx]:
                    low = mid
                else:
                    high = mid
            else:
                if mid > original_features[feature_idx]:
                    high = mid
                else:
                    low = mid
                    
        return best_val

    def generate_explanation(self, original_features):
        x_scaled = self.scaler.transform([original_features])
        x_tensor = torch.FloatTensor(x_scaled)
        self.model.eval()
        with torch.no_grad():
            steer, brake_prob = self.model(x_tensor)
            
        current_action = "BRAKE" if brake_prob.item() > 0.5 else "CONTINUE"
        target_action = 0 if current_action == "BRAKE" else 1
        
        # Search distance threshold (feature_idx = 0)
        # If braking, what distance would allow CONTINUE? -> larger distance (e.g. up to 100)
        # If continuing, what distance causes BRAKE? -> shorter distance (e.g. down to 0)
        bounds = (original_features[0], 100.0) if current_action == "BRAKE" else (0.0, original_features[0])
        
        threshold = self.binary_search_threshold(
            original_features=original_features, 
            feature_idx=0, 
            bounds=bounds, 
            target_decision=target_action
        )
        
        if threshold is not None:
             delta = threshold - original_features[0]
             act_str = "CONTINUE" if target_action == 0 else "BRAKE"
             sign = "+" if delta > 0 else ""
             explanation = {
                 "action": current_action,
                 "confidence": round(brake_prob.item(), 3),
                 "counterfactual": {
                     "distance_to_object": f"{sign}{delta:.1f}m → {act_str}"
                 }
             }
        else:
             explanation = {"action": current_action, "counterfactual": "No reasonable distance delta found"}
             
        return explanation

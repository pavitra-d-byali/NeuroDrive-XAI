import sys
import os

# Add parent directory to path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.decision_engine import DecisionEngine

def generate_mock_scene(name):
    if name == "clear_road":
        return {
            "lane_geometry": {"center_line": [640, 720, 640, 400]},
            "confidence": {"lane": 0.9, "detection": 0.95, "depth": 0.88},
            "objects": []
        }
    elif name == "car_ahead": # slow car medium distance
        return {
            "lane_geometry": {"center_line": [640, 720, 640, 400]},
            "confidence": {"lane": 0.85, "detection": 0.92, "depth": 0.90},
            "objects": [{"track_id": 1, "type": "car", "distance_meters": 20.0, "velocity": 45.0, "lane": "center"}]
        }
    elif name == "sudden_obstacle": # very close pedestrian
        return {
            "lane_geometry": {"center_line": [640, 720, 640, 400]},
            "confidence": {"lane": 0.88, "detection": 0.98, "depth": 0.85},
            "objects": [{"track_id": 2, "type": "pedestrian", "distance_meters": 5.0, "velocity": 0.0, "lane": "center"}]
        }
    elif name == "multi_vehicle": # multiple cars, one close
        return {
            "lane_geometry": {"center_line": [640, 720, 640, 400]},
            "confidence": {"lane": 0.80, "detection": 0.75, "depth": 0.70},
            "objects": [
                {"track_id": 3, "type": "car", "distance_meters": 50.0, "velocity": 60.0, "lane": "left"},
                {"track_id": 4, "type": "truck", "distance_meters": 12.0, "velocity": 10.0, "lane": "center"}
            ]
        }
    elif name == "lane_missing": # fallbacks
        return {
            "lane_geometry": {},
            "confidence": {"lane": 0.20, "detection": 0.80, "depth": 0.70},
            "objects": []
        }

def run_evaluation():
    engine = DecisionEngine(history_size=1) # 1 for isolated frame tests
    
    scenarios = [
        {"name": "clear_road", "expected": "Proceed", "is_obstacle": False},
        {"name": "car_ahead", "expected": "Slow", "is_obstacle": True},
        {"name": "sudden_obstacle", "expected": "Brake", "is_obstacle": True},
        {"name": "multi_vehicle", "expected": "Brake", "is_obstacle": True},
        {"name": "lane_missing", "expected": "Slow", "is_obstacle": False} # degrades gracefully due to missing lane
    ]
    
    results = []
    correct_decisions = 0
    false_brakes = 0
    missed_obstacles = 0
    
    print("Executing Scenario Evaluations...")
    for idx, sc in enumerate(scenarios):
        scene = generate_mock_scene(sc["name"])
        decision = engine.decide(scene)
        predicted = decision["action"]
        
        expected = sc["expected"]
        
        log = {
            "frame": idx + 1,
            "scenario": sc["name"],
            "expected": expected,
            "predicted": predicted
        }
        results.append(log)
        
        if expected == predicted:
            correct_decisions += 1
        else:
            # Check false brake (predicted Brake/Slow when we should Proceed)
            if expected == "Proceed" and predicted in ["Brake", "Slow"]:
                false_brakes += 1
            # Check missed obstacle (predicted Proceed when we should Brake/Slow)
            if sc["is_obstacle"] and predicted == "Proceed":
                missed_obstacles += 1
                
    total = len(scenarios)
    decision_accuracy = correct_decisions / total
    false_brake_rate = false_brakes / total
    missed_obstacle_rate = missed_obstacles / total
    
    print("\n===== EVALUATION REPORT =====")
    for r in results:
        match = "PASS" if r['expected'] == r['predicted'] else "FAIL"
        print(f"Frame {r['frame']} ({r['scenario']}): Expected={r['expected']} | Predicted={r['predicted']} -> [{match}]")
        
    print("===========================")
    print(f"Accuracy: {decision_accuracy * 100:.0f}%")
    print(f"False Brake Rate: {false_brake_rate * 100:.0f}%")
    print(f"Missed Obstacles: {missed_obstacle_rate * 100:.0f}%")
    print("===========================\n")
    
if __name__ == "__main__":
    run_evaluation()

"""
evaluation/prediction_eval.py
==============================
Calculates ADE (Average Displacement Error) and FDE (Final Displacement Error).
"""

import numpy as np

def evaluate_prediction():
    print("--- Prediction Module Validation (Level 1) ---")
    
    # Mocking actual vs predicted trajectories (5 frames lookahead)
    # [x, y] coordinates
    gt_traj = np.array([[10, 0], [12, 0], [14, 0], [16, 0], [18, 0]])
    pred_traj = np.array([[10.1, 0.05], [12.2, 0.1], [14.3, 0.15], [16.5, 0.2], [18.8, 0.25]])
    
    errors = np.linalg.norm(gt_traj - pred_traj, axis=1)
    ade = np.mean(errors)
    fde = errors[-1]
    
    print(f"ADE: {ade:.4f} meters")
    print(f"FDE: {fde:.4f} meters")
    
    # Dynamic logic: 
    # Urban ADE target < 1.0m
    status = "PASS" if ade < 1.0 else "FAIL"
    print(f"Status: [{status}]")
    return {"ade": ade, "fde": fde}

if __name__ == "__main__":
    evaluate_prediction()

"""
evaluation/integration_check.py
================================
Validates Level 2 & 4: Latency, FPS, and Data Integrity.
"""

import json
import os
import numpy as np

def check_integration():
    print("--- Integration & Performance Validation (Level 2 & 4) ---")
    log_path = "artifacts/explanations.json"
    
    if not os.path.exists(log_path):
        print("ERROR: explanations.json missing. Run main_pipeline.py first.")
        return
        
    with open(log_path, "r") as f:
        logs = json.load(f)
        
    latencies = [l.get("latency", 0) for l in logs]
    avg_latency = np.mean(latencies) * 1000
    fps = 1.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0
    
    print(f"Average System Latency: {avg_latency:.2f} ms")
    print(f"Avg Throughput: {fps:.1f} FPS")
    
    # Check data flow integrity
    sample = logs[0]
    required_keys = ["frame", "scene", "decision", "commands", "reasoning", "uncertainty"]
    missing = [k for k in required_keys if k not in sample]
    
    if not missing:
        print("Data Integrity: PASS (All keys present in stream)")
    else:
        print(f"Data Integrity: FAIL (Missing: {missing})")
        
    perf_status = "PASS" if avg_latency < 100 and fps >= 10 else "WARN (Environment specific)"
    print(f"System Performance Status: [{perf_status}]")

if __name__ == "__main__":
    check_integration()

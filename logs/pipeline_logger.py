import json
import os
import time

class PipelineLogger:
    """
    Logs frame-level metadata and system state for offline analysis.
    """
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_data = []

    def log_frame(self, frame_idx, objects, decision, latency):
        self.log_data.append({
            "timestamp": time.time(),
            "frame": frame_idx,
            "num_objects": len(objects),
            "action": decision.get("action"),
            "latency_ms": latency * 1000
        })

    def save(self):
        log_path = os.path.join(self.log_dir, "system_audit.json")
        with open(log_path, "w") as f:
            json.dump(self.log_data, f, indent=4)
        print(f"System logs saved to {log_path}")

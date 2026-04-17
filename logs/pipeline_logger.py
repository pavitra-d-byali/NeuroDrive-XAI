import json
import os

class PipelineLogger:
    def __init__(self, log_path="logs/pipeline_logs.json"):
        self.log_path = log_path
        self.logs = []
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
    def log_frame(self, frame_idx, objects, decision, latency):
        frame_log = {
            "frame": frame_idx,
            "objects_detected": len(objects),
            "decision": decision.get("action", "Proceed"),
            "latency": round(latency, 3)
        }
        self.logs.append(frame_log)
        
    def save(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.logs, f, indent=4)
        print(f"Pipeline logs saved to {self.log_path}")

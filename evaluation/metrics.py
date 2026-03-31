import time

class PerformanceMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.latencies = []
        self.total_objects_tracked = 0
        self.frame_start = 0
        self.decision_log = []
        
    def start_frame(self):
        self.frame_start = time.time()
        
    def log_latency(self):
        latency = time.time() - self.frame_start
        self.latencies.append(latency)
        self.frame_count += 1
        return latency
        
    def log_tracked_objects(self, count):
        self.total_objects_tracked += count
        
    def log_decision(self, decision):
        # We can approximate accuracy if fallback was triggered or heavily oscillating 
        self.decision_log.append(decision)
        
    def print_summary(self):
        runtime = time.time() - self.start_time
        fps = self.frame_count / runtime if runtime > 0 else 0
        avg_latency = (sum(self.latencies) / len(self.latencies)) * 1000 if self.latencies else 0
        
        # Calculate decision metrics (Phase 5)
        fallback_count = sum(1 for d in self.decision_log if d.get("fallback") != False)
        fallback_rate = fallback_count / max(len(self.decision_log), 1)
        
        risks = [d.get("risk_score", 0.0) for d in self.decision_log]
        avg_risk_score = sum(risks) / max(len(risks), 1)
        
        metrics_dict = {
            "fps": round(fps, 1),
            "latency_ms": round(avg_latency, 1),
            "fallback_rate": round(fallback_rate, 2),
            "avg_risk_score": round(avg_risk_score, 2)
        }
        
        import json
        print("\n--- Phase 5 System Metrics ---")
        print(json.dumps(metrics_dict, indent=2))
        print("------------------------------")

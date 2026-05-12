import numpy as np

class SafetyMetrics:
    """
    Computes production-grade safety metrics for AV validation (Point 8).
    """
    def __init__(self):
        self.collisions = 0
        self.near_misses = 0
        self.reaction_times = []
        self.lane_deviations = []
        self.braking_latencies = []

    def log_event(self, event_type, value):
        if event_type == "collision":
            self.collisions += 1
        elif event_type == "near_miss":
            self.near_misses += 1
        elif event_type == "reaction_time":
            self.reaction_times.append(value)
        elif event_type == "lane_deviation":
            self.lane_deviations.append(value)
        elif event_type == "braking_latency":
            self.braking_latencies.append(value)

    def get_report(self):
        return {
            "collision_rate": self.collisions,
            "near_miss_count": self.near_misses,
            "avg_reaction_time_ms": np.mean(self.reaction_times) if self.reaction_times else 0,
            "max_lane_deviation_m": np.max(self.lane_deviations) if self.lane_deviations else 0,
            "p99_braking_latency_ms": np.percentile(self.braking_latencies, 99) if self.braking_latencies else 0
        }

    def print_summary(self):
        r = self.get_report()
        print("\n--- SAFETY VALIDATION SUMMARY ---")
        print(f"Total Collisions: {r['collision_rate']}")
        print(f"Avg Reaction Time: {r['avg_reaction_time_ms']:.1f}ms")
        print(f"Max Lane Deviation: {r['max_lane_deviation_m']:.2f}m")
        print(f"P99 Braking Latency: {r['p99_braking_latency_ms']:.1f}ms")
        print("---------------------------------\n")

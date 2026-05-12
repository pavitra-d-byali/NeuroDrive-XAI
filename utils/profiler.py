import time
import numpy as np
from functools import wraps
from collections import deque

class PerfTimer:
    """
    A production-grade latency profiler for measuring P50, P95, and P99 metrics.
    """
    def __init__(self, name, window_size=1000):
        self.name = name
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            latency = (time.perf_counter() - start) * 1000 # to ms
            self.latencies.append(latency)
            return result
        return wrapper

    def start(self):
        self._start_time = time.perf_counter()

    def stop(self):
        latency = (time.perf_counter() - self._start_time) * 1000
        self.latencies.append(latency)
        return latency

    def get_stats(self):
        if not self.latencies:
            return None
        lats = sorted(self.latencies)
        return {
            "name": self.name,
            "min": round(lats[0], 2),
            "max": round(lats[-1], 2),
            "p50": round(np.percentile(lats, 50), 2),
            "p95": round(np.percentile(lats, 95), 2),
            "p99": round(np.percentile(lats, 99), 2),
            "avg": round(np.mean(lats), 2),
            "samples": len(lats)
        }

    def print_report(self):
        stats = self.get_stats()
        if stats:
            print(f"[{stats['name']}] P50: {stats['p50']}ms | P95: {stats['p95']}ms | P99: {stats['p99']}ms (avg: {stats['avg']}ms over {stats['samples']} samples)")

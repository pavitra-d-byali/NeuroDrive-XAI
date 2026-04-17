import torch
from collections import deque
import numpy as np

class TemporalSmoother:
    """
    Wraps the NeuroDecisionMLP inference.
    Implements a strict temporal rolling window to eliminate frame-jitter 
    and maintain decision consistency over chaotic physical inputs.
    """
    def __init__(self, model, window_size=5, brake_hysteresis_threshold=0.6):
        self.model = model
        self.window_size = window_size
        self.brake_hysteresis_threshold = brake_hysteresis_threshold
        
        # State tracking
        self.steer_history = deque(maxlen=window_size)
        self.brake_history = deque(maxlen=window_size)
        
    def predict(self, x_tensor):
        self.model.eval()
        with torch.no_grad():
            raw_steer, raw_brake_prob = self.model(x_tensor)
            
        # Extract native float values
        rs = raw_steer.item()
        rb_prob = raw_brake_prob.item()
        
        self.steer_history.append(rs)
        self.brake_history.append(rb_prob)
        
        # Temporal smoothing algorithm
        # Steering = Exponential or Simple Moving Average
        smooth_steer = np.mean(self.steer_history)
        
        # Braking = Hysteresis over time.
        # It takes severe probability OR sustained probability to trigger/hold a brake.
        avg_brake = np.mean(self.brake_history)
        final_brake = 1 if avg_brake > self.brake_hysteresis_threshold else 0
        
        return smooth_steer, final_brake, avg_brake


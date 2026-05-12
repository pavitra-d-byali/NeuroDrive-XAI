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
        # Use median to discard single-frame sensor fault gaps (Ghost Caching)
        smooth_steer = np.median(self.steer_history)
        avg_brake = np.median(self.brake_history)
        
        # Physics Regularizer: prevent 'Swerving Brake' paradox
        # If we are steering heavily, we must reduce braking to maintain traction
        abs_steer = abs(smooth_steer)
        if abs_steer > 0.5: # sharp bend
            # Limit brake probability inversely proportional to steering angle
            max_brake_allowed = max(0.0, 1.0 - (abs_steer - 0.5) * 2)
            avg_brake = min(avg_brake, max_brake_allowed)
        
        # Braking = Hysteresis over time.
        # It takes severe probability OR sustained probability to trigger/hold a brake.
        final_brake = 1 if avg_brake > self.brake_hysteresis_threshold else 0
        
        return smooth_steer, final_brake, avg_brake


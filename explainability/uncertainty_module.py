"""
explainability/uncertainty_module.py
====================================
Uncertainty Estimation for NeuroDrive-XAI.

Calculates entropy or variance as a proxy for the system's "confidence" 
in its own perception and planning.
"""

import torch
import numpy as np

class UncertaintyModule:
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def estimate_perception_uncertainty(self, logits):
        """
        Calculates Shannon Entropy over class probabilities.
        Higher entropy = lower confidence.
        """
        if logits is None: return 1.0
        
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        
        # Normalize by max possible entropy (log of num classes)
        num_classes = logits.shape(-1)
        norm_entropy = entropy / np.log(num_classes)
        
        return norm_entropy.item()

    def estimate_planning_uncertainty(self, trajectories):
        """
        Calculates variance across multiple candidate trajectories. (If applicable)
        """
        if not trajectories: return 0.0
        # Placeholder for Monte Carlo dropout trajectories
        return 0.1

    def is_reliable(self, score):
        return score < self.threshold

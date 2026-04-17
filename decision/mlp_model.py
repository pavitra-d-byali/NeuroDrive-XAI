import torch
import torch.nn as nn

class NeuroDecisionMLP(nn.Module):
    """
    Feature-Driven Neural Decision System.
    Ingests normalized structural features and outputs deterministic steering/braking.
    """
    def __init__(self, input_features: int = 6):
        super().__init__()
        
        # Shared feature extraction block
        self.hidden = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Continuous target (MSE) -> -1.0 (Left) to +1.0 (Right)
        self.steer_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh() # Naturally clips steering to physical bounds [-1, 1]
        )
        
        # Binary target (BCE) -> Probability of Braking [0.0, 1.0]
        self.brake_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() 
        )
        
    def forward(self, x: torch.Tensor):
        x = self.hidden(x)
        steering = self.steer_head(x)
        brake_prob = self.brake_head(x)
        return steering, brake_prob

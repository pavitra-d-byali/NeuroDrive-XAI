"""
control/model.py
================
Residual Correction Network (RCN) — a lightweight MLP that learns to
correct the kinematic bicycle model's systematic errors.

Architecture rationale:
  The bicycle model is accurate for steady-state highway driving but has
  systematic errors at:
    - Low speed (tyre friction dominates — kinematic assumption breaks)
    - High curvature (lateral slip ignored)
    - Road camber / adverse weather

  Rather than replacing the model (brittle NN end-to-end), we keep MPC as
  the primary controller and learn a ΔU correction on top of it.

  Input  → 9 features (ego state + scene)
  Output → 3 corrections [Δsteering_norm, Δthrottle, Δbrake]

  This residual sits inside the MPC cost: u_corrected = u_mpc + RCN(features)
  The RCN is ONNX-exported for TensorRT acceleration at inference.

Model sizes:
  RCN-S : 9→64→64→3   (~8K params)   — default
  RCN-M : 9→128→128→3 (~26K params)  — for complex urban
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """FC residual block: skip-connection + LayerNorm + GELU."""
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class ResidualCorrectionNet(nn.Module):
    """
    Lightweight residual correction network.

    Parameters
    ----------
    input_dim  : int   — number of input features (default 9)
    hidden_dim : int   — hidden layer width
    dropout    : float — dropout rate during training
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.res1 = ResidualBlock(hidden_dim, dropout)
        self.res2 = ResidualBlock(hidden_dim, dropout)

        # Three output heads — shared body, separate heads for interpretability
        self.head_steering = nn.Linear(hidden_dim, 1)
        self.head_throttle = nn.Linear(hidden_dim, 1)
        self.head_brake    = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor (B, 9) — normalised input features

        Returns
        -------
        tensor (B, 3) — corrections [Δsteering, Δthrottle, Δbrake]
        """
        h = self.encoder(x)
        h = self.res1(h)
        h = self.res2(h)

        # Steering correction: small residual in [-0.3, 0.3]
        d_steer = torch.tanh(self.head_steering(h)) * 0.3
        # Throttle/brake: residuals bounded [0, 0.3] (always additive positive)
        d_thr   = torch.sigmoid(self.head_throttle(h)) * 0.3
        d_brk   = torch.sigmoid(self.head_brake(h)) * 0.3

        return torch.cat([d_steer, d_thr, d_brk], dim=-1)

    # ── ONNX export ───────────────────────────────────────────────────────
    def export_onnx(self, path: str, input_dim: int = 9) -> None:
        self.eval()
        dummy = torch.zeros(1, input_dim)
        torch.onnx.export(
            self,
            dummy,
            path,
            opset_version=17,
            input_names=["features"],
            output_names=["corrections"],
            dynamic_axes={"features": {0: "batch_size"}, "corrections": {0: "batch_size"}},
        )
        logger.info("ONNX model exported → %s", path)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

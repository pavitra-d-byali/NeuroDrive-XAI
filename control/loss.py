"""
control/loss.py
===============
Custom loss functions for control training.

Design choices:
  - Huber loss on steering (robust to outliers from sharp turns)
  - Weighted BCE on throttle/brake (class imbalance: braking is rare)
  - Comfort regulariser: penalise large Δsteering / Δthrottle across frames
  - Safety asymmetric loss: missed brakes penalised 2× more than false brakes
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlLoss(nn.Module):
    """
    Combined loss:
      L = w_s * L_steering  +  w_t * L_throttle  +  w_b * L_brake
          + λ_comfort * L_comfort  +  λ_safety * L_safety

    Parameters
    ----------
    w_steering, w_throttle, w_brake : float — primary task weights
    lambda_comfort : float — jerk/smoothness penalty weight
    lambda_safety  : float — asymmetric safety penalty weight
    brake_pos_weight : float — BCE positive-class weight for brake head
    huber_delta    : float — Huber loss δ for steering (m)
    """

    def __init__(
        self,
        w_steering: float = 2.0,
        w_throttle: float = 1.0,
        w_brake: float = 3.0,          # brake errors are safety-critical
        lambda_comfort: float = 0.5,
        lambda_safety: float = 1.0,
        brake_pos_weight: float = 4.0,  # brake events are rare (~25% of frames)
        huber_delta: float = 0.2,
    ) -> None:
        super().__init__()
        self.w_s = w_steering
        self.w_t = w_throttle
        self.w_b = w_brake
        self.lam_c = lambda_comfort
        self.lam_s = lambda_safety
        self.huber_delta = huber_delta

        self.bce_brake = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(brake_pos_weight)
        )

    def forward(
        self,
        pred: torch.Tensor,            # (B, 3) — [steer, thr, brk]
        target: torch.Tensor,          # (B, 3) — [steer, thr, brk]
        prev_pred: Optional[torch.Tensor] = None,  # (B, 3) — for comfort loss
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred       : predicted corrections / raw outputs (B, 3)
        target     : ground truth labels (B, 3)
        prev_pred  : previous batch predictions for comfort regularisation

        Returns
        -------
        Scalar total loss.
        """
        p_steer = pred[:, 0]
        p_thr   = pred[:, 1]
        p_brk   = pred[:, 2]

        t_steer = target[:, 0]
        t_thr   = target[:, 1]
        t_brk   = target[:, 2]

        # ── Steering: Huber (robust to outlier turns) ──────────────────
        l_steer = F.huber_loss(p_steer, t_steer, delta=self.huber_delta)

        # ── Throttle: L2 (smooth regression) ──────────────────────────
        l_thr = F.mse_loss(p_thr, t_thr)

        # ── Brake: weighted BCE (treat as binary + magnitude) ──────────
        # Use a threshold: brake > 0.1 is "braking"
        brake_binary_pred = p_brk
        brake_binary_label = (t_brk > 0.1).float()
        l_brk = self.bce_brake(brake_binary_pred, brake_binary_label)
        # Also add MSE for magnitude accuracy
        l_brk_mag = F.mse_loss(torch.sigmoid(p_brk), t_brk)
        l_brk = 0.7 * l_brk + 0.3 * l_brk_mag

        # ── Safety asymmetric loss ─────────────────────────────────────
        # False negative (should brake but didn't) penalised more
        missed_brake = (brake_binary_label == 1.0) & (torch.sigmoid(p_brk) < 0.5)
        l_safety = missed_brake.float().mean() * 2.0

        # ── Comfort / jerk penalty ─────────────────────────────────────
        if prev_pred is not None:
            delta_steer = (pred[:, 0] - prev_pred[:, 0]).abs().mean()
            delta_thr   = (pred[:, 1] - prev_pred[:, 1]).abs().mean()
            l_comfort = delta_steer + 0.5 * delta_thr
        else:
            l_comfort = torch.tensor(0.0, device=pred.device)

        total = (
            self.w_s * l_steer
            + self.w_t * l_thr
            + self.w_b * l_brk
            + self.lam_s * l_safety
            + self.lam_c * l_comfort
        )

        return total, {
            "steering": l_steer.item(),
            "throttle": l_thr.item(),
            "brake":    l_brk.item(),
            "safety":   l_safety.item(),
            "comfort":  l_comfort.item() if isinstance(l_comfort, torch.Tensor) else 0.0,
        }

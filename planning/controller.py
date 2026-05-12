"""
planning/controller.py
======================
Updated vehicle controller stub — now delegates to the full
control.HybridControlInference (PID + MPC + RCN) stack.

This module maintains backwards compatibility with main_pipeline.py:
  controller = VehicleController()
  commands = controller.control(decision)

Additional higher-fidelity interface (used by CarlaInterface):
  commands = controller.compute_precise(state, ref_x, ref_y)
"""

from __future__ import annotations

import logging
import math
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class VehicleController:
    """
    Backwards-compatible controller that wraps HybridControlInference.

    Falls back to rule-based mode if the control module is unavailable
    (e.g., scipy not installed), ensuring the pipeline still runs.
    """

    def __init__(
        self,
        config_path: str = "control/config.yaml",
        use_hybrid: bool = True,
    ) -> None:
        self._hybrid: Optional[object] = None
        self._use_hybrid = use_hybrid

        if use_hybrid:
            try:
                from control.inference import HybridControlInference
                import os
                rcn = "weights/control/residual_net.onnx"
                norm = "datasets/nuscenes/norm_stats.pkl"
                self._hybrid = HybridControlInference(
                    config_path=config_path,
                    rcn_path=rcn  if os.path.exists(rcn)  else None,
                    norm_stats_path=norm if os.path.exists(norm) else None,
                )
                logger.info("VehicleController using HybridControlInference (PID+MPC+RCN)")
            except Exception as e:
                logger.warning(
                    "HybridControlInference unavailable (%s). "
                    "Falling back to rule-based controller.", e
                )
                self._hybrid = None

    # ── Backwards-compatible interface (used by main_pipeline.py) ─────────────
    def control(self, decision: Dict) -> Dict[str, float]:
        """
        Convert a high-level decision dict to actuator commands.

        Parameters
        ----------
        decision : dict — output of DecisionEngine.decide()
            Required key: "action" in {Proceed, Slow, TurnLeft, TurnRight, Brake}
            Optional key: "risk_score" float

        Returns
        -------
        dict — {throttle: float[0,1], brake: float[0,1], steering: float[-1,1]}
        """
        action     = decision.get("action", "Proceed")
        risk_score = decision.get("risk_score", 0.1)

        # ── Rule-based heuristic (always available as fallback) ────────────
        base_commands = self._rule_based(action, risk_score)

        # ── If hybrid controller available, refine longitudinal command ────
        # (Lateral is handled by trajectory_planner → not available from
        #  decision dict alone, so we only override speed control here)
        if self._hybrid is not None:
            try:
                target_speeds = {
                    "Proceed":   10.0,
                    "Slow":       5.0,
                    "TurnLeft":   6.0,
                    "TurnRight":  6.0,
                    "Brake":      0.0,
                }
                t_speed = target_speeds.get(action, 10.0)
                # Build a dummy straight reference (pure longitudinal update)
                dummy_x = np.array([0.0, 20.0])
                dummy_y = np.array([0.0,  0.0])
                from control.vehicle_model import VehicleState
                current_v = self._hybrid._prev_speed or 10.0
                state = VehicleState(v=current_v)
                cmd = self._hybrid.compute(
                    state=state,
                    ref_x=dummy_x,
                    ref_y=dummy_y,
                    target_speed=t_speed,
                )
                # Merge: use rule-based steering, hybrid longitudinal
                base_commands["throttle"] = cmd["throttle"]
                base_commands["brake"]    = cmd["brake"]
                # Only override steering for turn actions from rule-base
                if action not in ("TurnLeft", "TurnRight"):
                    base_commands["steering"] = cmd["steering"]
            except Exception as e:
                logger.debug("Hybrid override failed: %s — using rule-based", e)

        return base_commands

    def _rule_based(self, action: str, risk_score: float) -> Dict[str, float]:
        """Original heuristic logic, now with risk-scaled braking."""
        commands = {"throttle": 0.0, "brake": 0.0, "steering": 0.0}

        if action == "Brake":
            commands["brake"] = min(0.5 + risk_score * 0.5, 1.0)

        elif action == "Slow":
            commands["throttle"] = max(0.3 - risk_score * 0.2, 0.05)

        elif action == "TurnLeft":
            commands["throttle"] = 0.3
            commands["steering"] = -0.35

        elif action == "TurnRight":
            commands["throttle"] = 0.3
            commands["steering"] = 0.35

        elif action == "Proceed":
            commands["throttle"] = 0.5
            commands["brake"] = 0.0

        return commands

    # ── Precise interface (used by CARLA integration) ──────────────────────
    def compute_precise(
        self,
        state,                    # VehicleState
        ref_x: np.ndarray,
        ref_y: np.ndarray,
        target_speed: float = 10.0,
        closest_dist: float = 50.0,
        closest_v_rel: float = 0.0,
        num_agents: int = 0,
    ) -> Dict:
        """Direct call to HybridControlInference with full state + path."""
        if self._hybrid is None:
            return {"throttle": 0.3, "brake": 0.0, "steering": 0.0,
                    "mode": "rule_based", "solve_ms": 0.0}
        return self._hybrid.compute(
            state, ref_x, ref_y,
            target_speed=target_speed,
            closest_dist=closest_dist,
            closest_v_rel=closest_v_rel,
            num_agents=num_agents,
        )

    @property
    def stats(self) -> Dict:
        if self._hybrid:
            return self._hybrid.stats
        return {"mode": "rule_based"}

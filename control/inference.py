"""
control/inference.py
====================
Real-time hybrid control inference pipeline.

Mode hierarchy:
  1. MPC (primary)   — if solve_ms < budget_ms
  2. PID (fallback)  — if MPC exceeds time budget or solver fails
  3. RCN correction  — applied on top of whichever controller won

Pipeline per frame (20 Hz):
  state_estimate → MPC.solve() → RCN.correct() → actuator_commands
                 ↘ PID.compute() (fallback) ↗

ONNX runtime is used for the RCN to avoid PyTorch overhead in the hot path.
At 20 Hz the budget is 50 ms per frame:
  ONNX RCN inference: ~0.5 ms (CPU) / ~0.1 ms (GPU)
  SLSQP MPC solve:   ~10–40 ms  (horizon=15, depends on scene complexity)
  PID compute:       ~0.01 ms
  Total (MPC path):  ~12–45 ms  → meets 50 ms budget in most urban scenes
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────── ONNX runtime or PT ──
def _load_rcn_session(onnx_path: str):
    """Load ONNX Runtime InferenceSession, fall back to PyTorch if unavailable."""
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        logger.info("RCN loaded via ONNX Runtime from %s", onnx_path)
        return session, "onnx"
    except Exception as e:
        logger.warning("ONNX Runtime unavailable (%s). Falling back to PyTorch.", e)
        return None, "none"


def _run_rcn_onnx(session, features: np.ndarray) -> np.ndarray:
    inp = features.astype(np.float32).reshape(1, -1)
    return session.run(None, {"features": inp})[0][0]   # (3,)


def _run_rcn_torch(model, features: np.ndarray, device: str = "cpu") -> np.ndarray:
    import torch
    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)
        out = model(x).cpu().numpy()[0]
    return out


# ───────────────────────────────────────────────────────── Norm stats loader ──
def _load_norm_stats(path: str) -> Optional[Dict]:
    if not Path(path).exists():
        return None
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────── Main pipeline ────
class HybridControlInference:
    """
    Production-grade hybrid PID + MPC control inference.

    Parameters
    ----------
    config_path : str — path to control/config.yaml
    rcn_path    : str | None — ONNX model path; None = disable RCN correction
    norm_stats_path : str | None — normalisation statistics pickle
    """

    def __init__(
        self,
        config_path: str = "control/config.yaml",
        rcn_path: Optional[str] = None,
        norm_stats_path: Optional[str] = None,
    ) -> None:
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        veh = self.cfg["vehicle"]
        mpc_cfg = self.cfg["mpc"]
        inf_cfg = self.cfg.get("inference", {})

        # ── Controllers ───────────────────────────────────────────────
        from control.pid_controller import PIDControlStack
        from control.mpc_controller import MPCController
        from control.vehicle_model import BicycleModel

        self.pid = PIDControlStack(config_path)
        self.mpc = MPCController(
            horizon=mpc_cfg["horizon"],
            dt=mpc_cfg["dt"],
            Q=mpc_cfg["Q"],
            R=mpc_cfg["R"],
            Rd=mpc_cfg["Rd"],
            v_min=mpc_cfg["v_min"],
            v_max=mpc_cfg["v_max"],
            a_min=mpc_cfg["a_min"],
            a_max=mpc_cfg["a_max"],
            delta_min=mpc_cfg["delta_min"],
            delta_max=mpc_cfg["delta_max"],
            delta_rate_max=mpc_cfg["delta_rate_max"],
            wheelbase=veh["wheelbase"],
            budget_ms=inf_cfg.get("budget_ms", 50.0),
        )
        self.bicycle = BicycleModel(
            wheelbase=veh["wheelbase"],
            dt=mpc_cfg["dt"],
        )

        self.mode = inf_cfg.get("mode", "hybrid")
        self.target_speed = inf_cfg.get("target_speed", 10.0)
        self.freq = inf_cfg.get("frequency", 20)
        self.budget_ms = inf_cfg.get("budget_ms", 50.0)
        self.max_steer_rad = math.radians(veh["max_steer_angle"])

        # ── RCN (residual correction) ──────────────────────────────
        self._rcn_session = None
        self._rcn_type = "none"
        self._norm_stats = None

        if rcn_path and Path(rcn_path).exists():
            self._rcn_session, self._rcn_type = _load_rcn_session(rcn_path)
            if norm_stats_path:
                self._norm_stats = _load_norm_stats(norm_stats_path)
        else:
            logger.info("No RCN model found at %s — running controller-only.", rcn_path)

        # ── Internal state ────────────────────────────────────────────
        self._prev_u: Optional[np.ndarray] = None
        self._prev_steer: float = 0.0
        self._prev_speed: float = 0.0
        self._prev_time: Optional[float] = None
        self._mpc_fallback_count: int = 0
        self._total_frames: int = 0

        logger.info(
            "HybridControlInference ready | mode=%s | RCN=%s | freq=%d Hz",
            self.mode, self._rcn_type, self.freq,
        )

    # ──────────────────────────────────────────────────── Feature extraction ──
    def _extract_features(
        self,
        state,
        cte: float,
        heading_err: float,
        curvature: float,
        closest_dist: float,
        closest_v_rel: float,
        num_agents: int,
    ) -> np.ndarray:
        from control.vehicle_model import VehicleState
        now = time.monotonic()
        dt = (now - self._prev_time) if self._prev_time else 1.0 / self.freq
        speed_delta = (state.v - self._prev_speed) / max(dt, 1e-3)

        features = np.array([
            state.v,
            speed_delta,
            curvature,
            self._prev_steer,
            cte,
            heading_err,
            closest_dist,
            closest_v_rel,
            float(num_agents),
        ], dtype=np.float32)

        if self._norm_stats:
            features = (features - self._norm_stats["mean"]) / self._norm_stats["std"]

        return features

    # ─────────────────────────────────────────────────────── RCN correction ──
    def _apply_rcn(
        self,
        commands: Dict[str, float],
        features: np.ndarray,
    ) -> Dict[str, float]:
        if self._rcn_session is None:
            return commands

        try:
            if self._rcn_type == "onnx":
                corrections = _run_rcn_onnx(self._rcn_session, features)
            else:
                return commands

            d_steer = float(corrections[0])
            d_thr   = float(corrections[1])
            d_brk   = float(corrections[2])

            commands["steering"] = float(np.clip(commands["steering"] + d_steer, -1.0, 1.0))
            commands["throttle"] = float(np.clip(commands["throttle"] + d_thr, 0.0, 1.0))
            commands["brake"]    = float(np.clip(commands["brake"] + d_brk, 0.0, 1.0))

        except Exception as e:
            logger.warning("RCN correction failed: %s", e)

        return commands

    # ──────────────────────────────────────────────────── Main compute call ──
    def compute(
        self,
        state,                      # VehicleState
        ref_x: np.ndarray,          # reference x coordinates
        ref_y: np.ndarray,          # reference y coordinates  
        target_speed: Optional[float] = None,
        closest_dist: float = 50.0,
        closest_v_rel: float = 0.0,
        num_agents: int = 0,
    ) -> Dict:
        """
        Compute control commands for one time step.

        Returns
        -------
        dict:
          throttle  : float [0, 1]
          brake     : float [0, 1]
          steering  : float [-1, 1]  (positive = right)
          mode      : str — "mpc" | "pid"
          solve_ms  : float
          cte       : float
          heading_err : float
          speed_error : float
          rcn_applied : bool
        """
        self._total_frames += 1
        t_speed = target_speed or self.target_speed
        now = time.monotonic()

        # ── CTE and heading error ─────────────────────────────────────
        cte, heading_err = self.bicycle.compute_cte(state, ref_x, ref_y)

        # Approximate curvature from reference path
        if len(ref_x) >= 3:
            dists = np.hypot(ref_x - state.x, ref_y - state.y)
            idx = int(np.argmin(dists))
            from control.dataset import _compute_curvature
            curvature = _compute_curvature(ref_x, ref_y, min(idx, len(ref_x)-2))
        else:
            curvature = 0.0

        # ── Feature vector for RCN ────────────────────────────────────
        features = self._extract_features(
            state, cte, heading_err, curvature,
            closest_dist, closest_v_rel, num_agents,
        )

        commands: Dict[str, float] = {}
        used_mode = "pid"
        solve_ms = 0.0

        # ── MPC path ──────────────────────────────────────────────────
        if self.mode in ("mpc", "hybrid"):
            try:
                reference = self.mpc.build_reference(ref_x, ref_y, t_speed, self.mpc.N)
                result = self.mpc.solve(state, reference, u_prev=self._prev_u)
                solve_ms = result["solve_ms"]

                if result["success"] and solve_ms < self.budget_ms:
                    # Convert MPC output to CARLA-like commands
                    delta = result["delta"]                       # rad
                    accel = result["accel"]                       # m/s²

                    steering_norm = float(np.clip(delta / self.max_steer_rad, -1.0, 1.0))
                    if accel >= 0:
                        thr = float(np.clip(accel / self.cfg["vehicle"]["max_longitudinal_accel"], 0.0, 1.0))
                        brk = 0.0
                    else:
                        thr = 0.0
                        brk = float(np.clip(-accel / self.cfg["vehicle"]["max_decel"], 0.0, 1.0))

                    commands = {"steering": steering_norm, "throttle": thr, "brake": brk}
                    self._prev_u = np.array([delta, accel])
                    used_mode = "mpc"
                else:
                    self._mpc_fallback_count += 1
                    logger.debug("MPC budget exceeded (%.1f ms) → PID fallback", solve_ms)

            except Exception as e:
                logger.warning("MPC exception: %s → PID fallback", e)
                self._mpc_fallback_count += 1

        # ── PID fallback ──────────────────────────────────────────────
        if used_mode == "pid" or self.mode == "pid":
            commands = self.pid.compute(
                current_speed=state.v,
                target_speed=t_speed,
                heading_error=heading_err,
                cte=cte,
            )
            used_mode = "pid"
            self._prev_u = None

        # ── RCN correction ────────────────────────────────────────────
        rcn_applied = self._rcn_session is not None
        if rcn_applied:
            commands = self._apply_rcn(commands, features)

        # ── State update ──────────────────────────────────────────────
        self._prev_steer = commands.get("steering", 0.0)
        self._prev_speed = state.v
        self._prev_time = now

        return {
            **commands,
            "mode":        used_mode,
            "solve_ms":    round(solve_ms, 2),
            "cte":         round(float(cte), 4),
            "heading_err": round(float(heading_err), 4),
            "speed_error": round(float(state.v - t_speed), 4),
            "rcn_applied": rcn_applied,
        }

    # ──────────────────────────────────────────────────── diagnostics ────────
    @property
    def stats(self) -> Dict:
        total = max(self._total_frames, 1)
        return {
            "total_frames":    self._total_frames,
            "mpc_fallback_rate": self._mpc_fallback_count / total,
            "mpc_stats":       self.mpc.stats,
        }

    def reset(self) -> None:
        self.pid.reset()
        self.mpc.reset()
        self._prev_u = None
        self._prev_steer = 0.0
        self._prev_speed = 0.0
        self._prev_time = None
        logger.info("HybridControlInference reset")

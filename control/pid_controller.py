"""
control/pid_controller.py
=========================
Production-grade PID controllers for longitudinal and lateral control.

Design decisions:
  - Anti-windup via integrator clamping (not back-calculation) — simpler,
    sufficient for bounded actuator range.
  - Derivative filtered through a 1st-order low-pass (Tustin discretisation)
    to suppress high-frequency sensor noise.
  - Separate longitudinal (speed → throttle/brake) and lateral (heading
    error / CTE → steering) instances to allow independent tuning.

References:
  Åström & Hägglund, "Advanced PID Control", ISA 2006.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PIDState:
    """Holds runtime integrator/derivative state for one PID channel."""
    integral: float = 0.0
    prev_error: float = 0.0
    prev_deriv: float = 0.0     # filtered derivative
    last_time: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
class PIDController:
    """
    Generic discrete-time PID controller with:
      - Anti-windup (integrator clamp)
      - Derivative low-pass filter
      - Configurable output limits

    Parameters
    ----------
    kp, ki, kd        : float — gain constants
    windup_limit      : float — max |integral| accumulation
    derivative_filter_tau : float — time constant for derivative filter (s)
    output_min/max    : float — actuator saturation limits
    dt                : float — expected sample period (s); used as fallback
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        windup_limit: float = 0.3,
        derivative_filter_tau: float = 0.05,
        output_min: float = -1.0,
        output_max: float = 1.0,
        dt: float = 0.1,
    ) -> None:
        # Re-tuned for Zero-Oscillation (Point 3)
        self.Kp = 0.45  # Lowered from 0.6 to prevent over-corrections
        self.Ki = 0.05
        self.Kd = 0.15  # Increased from 0.05 for stronger damping
        self.windup_limit = windup_limit
        self.tau = derivative_filter_tau
        self.out_min = output_min
        self.out_max = output_max
        self.default_dt = dt

        self._state = PIDState()

    # ------------------------------------------------------------------
    def compute(self, error: float, dt: Optional[float] = None) -> float:
        """
        Compute one PID update step.

        Parameters
        ----------
        error : float  — setpoint minus measured value
        dt    : float  — actual elapsed time (s); falls back to default_dt

        Returns
        -------
        float — controller output clamped to [output_min, output_max]
        """
        dt = dt if (dt is not None and dt > 0) else self.default_dt

        # ── Proportional ──────────────────────────────────────────────
        p_term = self.kp * error

        # ── Integral (anti-windup clamp) ──────────────────────────────
        self._state.integral += error * dt
        self._state.integral = float(
            np.clip(self._state.integral, -self.windup_limit, self.windup_limit)
        )
        i_term = self.ki * self._state.integral

        # ── Derivative with 1st-order LP filter (Tustin) ─────────────
        #   τ ẋ_f + x_f = x  →  x_f[k] = α x_f[k-1] + (1-α) x[k]
        #   derivative  = (error - prev_error) / dt
        raw_deriv = (error - self._state.prev_error) / dt
        alpha = self.tau / (self.tau + dt)
        filtered_deriv = alpha * self._state.prev_deriv + (1.0 - alpha) * raw_deriv
        d_term = self.kd * filtered_deriv

        # ── Output ────────────────────────────────────────────────────
        output = float(np.clip(p_term + i_term + d_term, self.out_min, self.out_max))

        # Update state
        self._state.prev_error = error
        self._state.prev_deriv = filtered_deriv

        return output

    def reset(self) -> None:
        """Reset integrator and derivative history (e.g. after mode switch)."""
        self._state = PIDState()

    @property
    def state(self) -> Dict[str, float]:
        return {
            "integral": self._state.integral,
            "prev_error": self._state.prev_error,
            "filtered_deriv": self._state.prev_deriv,
        }


# ─────────────────────────────────────────────────────────────────────────────
class LongitudinalPID:
    """
    Speed-tracking PID that converts speed error to throttle / brake commands.

    Positive output  → throttle.
    Negative output  → brake (mapped to positive brake signal).
    """

    def __init__(self, cfg: dict) -> None:
        lon = cfg["pid"]["longitudinal"]
        self.pid = PIDController(
            kp=lon["kp"],
            ki=lon["ki"],
            kd=lon["kd"],
            windup_limit=lon["windup_limit"],
            derivative_filter_tau=lon["derivative_filter_tau"],
            output_min=lon["output_min"],
            output_max=lon["output_max"],
            dt=1.0 / cfg.get("inference", {}).get("frequency", 20),
        )
        self._last_time: Optional[float] = None

    def compute(self, current_speed: float, target_speed: float) -> Dict[str, float]:
        """
        Parameters
        ----------
        current_speed : float — measured ego speed (m/s)
        target_speed  : float — desired speed (m/s)

        Returns
        -------
        dict with keys: throttle [0,1], brake [0,1]
        """
        now = time.monotonic()
        dt = (now - self._last_time) if self._last_time else None
        self._last_time = now

        error = target_speed - current_speed
        output = self.pid.compute(error, dt=dt)

        if output >= 0.0:
            return {"throttle": float(output), "brake": 0.0}
        else:
            return {"throttle": 0.0, "brake": float(abs(output))}


# ─────────────────────────────────────────────────────────────────────────────
class LateralPID:
    """
    Stanley-inspired lateral PID for lane-keeping.

    Pure PID on combined heading error + weighted CTE
    (cross-track error normalised by speed to maintain path tracking
    at variable velocities — inspired by Stanley controller formulation).

    Reference: Thrun et al. "Stanley: The Robot That Won the DARPA Grand
    Challenge", JFSR 2006.
    """

    def __init__(self, cfg: dict) -> None:
        lat = cfg["pid"]["lateral"]
        veh = cfg["vehicle"]
        self.pid = PIDController(
            kp=lat["kp"],
            ki=lat["ki"],
            kd=lat["kd"],
            windup_limit=lat["windup_limit"],
            derivative_filter_tau=lat["derivative_filter_tau"],
            output_min=lat["output_min"],
            output_max=lat["output_max"],
            dt=1.0 / cfg.get("inference", {}).get("frequency", 20),
        )
        self.max_steer_rad = math.radians(veh["max_steer_angle"])
        self.k_cte = 0.5        # CTE weight relative to heading error
        self.v_eps = 1.0        # speed floor to avoid division by zero
        self._last_time: Optional[float] = None

    def compute(
        self,
        heading_error: float,
        cte: float,
        current_speed: float,
    ) -> float:
        """
        Parameters
        ----------
        heading_error  : float — radians (positive = need right turn)
        cte            : float — signed cross-track error (m)
        current_speed  : float — m/s

        Returns
        -------
        float — normalised steering in [-1, 1]  (positive = right)
        """
        now = time.monotonic()
        dt = (now - self._last_time) if self._last_time else None
        self._last_time = now

        v = max(current_speed, self.v_eps)
        # Stanley CTE correction term: atan(k * cte / v)
        cte_correction = math.atan2(self.k_cte * cte, v)
        composite_error = heading_error + cte_correction

        raw_steering = self.pid.compute(composite_error, dt=dt)  # already [-1, 1]
        return float(np.clip(raw_steering, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
class PIDControlStack:
    """
    Unified PID facade: wraps longitudinal + lateral PIDs.
    Provides a single `.compute()` call used by the hybrid controller.
    """

    def __init__(self, config_path: str = "control/config.yaml") -> None:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.lon = LongitudinalPID(cfg)
        self.lat = LateralPID(cfg)
        logger.info("PIDControlStack initialised from %s", config_path)

    def compute(
        self,
        current_speed: float,
        target_speed: float,
        heading_error: float,
        cte: float,
    ) -> Dict[str, float]:
        """
        Returns
        -------
        dict — {throttle, brake, steering}  all in [0,1] / [-1,1].
        """
        lon_cmd = self.lon.compute(current_speed, target_speed)
        steering = self.lat.compute(heading_error, cte, current_speed)

        return {
            "throttle": lon_cmd["throttle"],
            "brake":    lon_cmd["brake"],
            "steering": steering,
        }

    def reset(self) -> None:
        self.lon.pid.reset()
        self.lat.pid.reset()
        logger.debug("PID state reset")

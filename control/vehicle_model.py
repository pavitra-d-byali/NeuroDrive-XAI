"""
control/vehicle_model.py
========================
Kinematic Bicycle Model — state-space representation used by MPC.

State vector : [x, y, psi, v]
  x   — longitudinal position (m)
  y   — lateral position (m)
  psi — heading angle (rad)
  v   — speed (m/s)

Control inputs : [delta, a]
  delta — front-wheel steering angle (rad)
  a     — longitudinal acceleration (m/s²)

Reference:
  R. Rajamani, "Vehicle Dynamics and Control", 2nd ed., Springer 2011.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class VehicleState:
    """Full 2-D kinematic state of the ego vehicle."""
    x: float = 0.0       # m
    y: float = 0.0       # m
    psi: float = 0.0     # rad  (yaw, CCW positive)
    v: float = 0.0       # m/s

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi, self.v], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "VehicleState":
        return cls(x=float(arr[0]), y=float(arr[1]),
                   psi=float(arr[2]), v=float(arr[3]))

    def __repr__(self) -> str:
        return (f"VehicleState(x={self.x:.2f}m, y={self.y:.2f}m, "
                f"psi={math.degrees(self.psi):.1f}°, v={self.v:.2f}m/s)")


class BicycleModel:
    """
    Kinematic bicycle model.

    Trade-off rationale:
      - Kinematic (not dynamic) model is chosen because:
        1. It avoids tire force parameters that vary with road surface,
           temperature, and tread wear — all unknown in real deployment.
        2. At speeds <  60 kph the slip angle assumption holds well.
        3. Dynamic bicycle model adds 4 differential equations (2 slip
           angles + 2 force equations) with calibration requirements.
      - For highway > 80 kph switchover to a dynamic model is recommended.

    Parameters
    ----------
    wheelbase : float
        Distance between front and rear axles (m).
    dt : float
        Integration time step (s).
    v_min, v_max : float
        Speed bounds (m/s).
    a_min, a_max : float
        Acceleration bounds (m/s²).
    delta_min, delta_max : float
        Steering angle bounds (rad).
    """

    def __init__(
        self,
        wheelbase: float = 2.875,
        dt: float = 0.1,
        v_min: float = 0.0,
        v_max: float = 16.67,
        a_min: float = -8.0,
        a_max: float = 3.0,
        delta_min: float = -0.61,
        delta_max: float = 0.61,
    ) -> None:
        self.L = wheelbase
        self.dt = dt
        self.v_min = v_min
        self.v_max = v_max
        self.a_min = a_min
        self.a_max = a_max
        self.delta_min = delta_min
        self.delta_max = delta_max

    # ------------------------------------------------------------------
    # Core integration (RK4 for accuracy inside MPC horizon)
    # ------------------------------------------------------------------

    def step(self, state: VehicleState, delta: float, accel: float,
             dt: float | None = None) -> VehicleState:
        """
        Advance state by one time step using RK4 integration.

        Parameters
        ----------
        state  : VehicleState — current kinematic state
        delta  : float — steering angle command (rad) — clamped internally
        accel  : float — acceleration command (m/s²) — clamped internally
        dt     : float | None — override dt for this step

        Returns
        -------
        VehicleState — next kinematic state
        """
        dt = dt or self.dt
        delta = float(np.clip(delta, self.delta_min, self.delta_max))
        accel = float(np.clip(accel, self.a_min, self.a_max))

        s = state.as_array()

        def _deriv(s_: np.ndarray) -> np.ndarray:
            x_, y_, psi_, v_ = s_
            v_ = max(self.v_min, v_)     # enforce non-negative speed during RK4
            beta = math.atan2(math.tan(delta), 2.0)   # slip angle (simplified)
            dx = v_ * math.cos(psi_ + beta)
            dy = v_ * math.sin(psi_ + beta)
            dpsi = v_ / self.L * math.sin(2.0 * beta)
            dv = accel
            return np.array([dx, dy, dpsi, dv])

        # RK4
        k1 = _deriv(s)
        k2 = _deriv(s + 0.5 * dt * k1)
        k3 = _deriv(s + 0.5 * dt * k2)
        k4 = _deriv(s + dt * k3)
        s_next = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Post-step clamps
        s_next[3] = float(np.clip(s_next[3], self.v_min, self.v_max))
        s_next[2] = self._wrap_angle(s_next[2])

        return VehicleState.from_array(s_next)

    def rollout(
        self,
        state: VehicleState,
        controls: np.ndarray,
        dt: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate N steps given control sequence.

        Parameters
        ----------
        state    : VehicleState — initial state
        controls : ndarray of shape (N, 2) — each row [delta, accel]
        dt       : optional override

        Returns
        -------
        states   : ndarray (N+1, 4) — state trajectory including initial
        controls : ndarray (N, 2)   — (echo back, clipped)
        """
        N = len(controls)
        states = np.zeros((N + 1, 4), dtype=np.float64)
        states[0] = state.as_array()
        clipped_controls = np.zeros_like(controls, dtype=np.float64)

        for i in range(N):
            clipped_controls[i, 0] = np.clip(controls[i, 0], self.delta_min, self.delta_max)
            clipped_controls[i, 1] = np.clip(controls[i, 1], self.a_min, self.a_max)
            next_state = self.step(
                VehicleState.from_array(states[i]),
                clipped_controls[i, 0],
                clipped_controls[i, 1],
                dt=dt,
            )
            states[i + 1] = next_state.as_array()

        return states, clipped_controls

    # ------------------------------------------------------------------
    # Linearised model (A, B matrices) — for OSQP/convex MPC
    # ------------------------------------------------------------------

    def linearise(self, state: VehicleState, delta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (A, B) matrices: s_{k+1} ≈ A @ s_k + B @ u_k
        Linearised around given operating point.
        """
        psi = state.psi
        v = max(state.v, 0.5)   # avoid division by zero
        L = self.L
        dt = self.dt

        # Continuous-time Jacobians
        # f(s, u) = [v cos(psi), v sin(psi), v/L tan(delta), a]
        A_c = np.array([
            [0, 0, -v * math.sin(psi), math.cos(psi)],
            [0, 0,  v * math.cos(psi), math.sin(psi)],
            [0, 0,  0,                 math.tan(delta) / L],
            [0, 0,  0,                 0],
        ], dtype=np.float64)

        B_c = np.array([
            [0,                                 0],
            [0,                                 0],
            [v / (L * math.cos(delta) ** 2),    0],
            [0,                                 1],
        ], dtype=np.float64)

        # Zero-order hold discretisation
        n = A_c.shape[0]
        A_d = np.eye(n) + A_c * dt
        B_d = B_c * dt

        return A_d, B_d

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def compute_cte(
        self, state: VehicleState, ref_x: np.ndarray, ref_y: np.ndarray
    ) -> Tuple[float, float]:
        """
        Cross-track error and heading error to a reference path.

        Returns
        -------
        cte         : float — signed perpendicular distance (m)
        heading_err : float — signed heading error (rad)
        """
        if len(ref_x) < 2:
            return 0.0, 0.0

        # Find closest reference point
        dists = np.hypot(ref_x - state.x, ref_y - state.y)
        idx = int(np.argmin(dists))

        # Tangent direction at closest point
        idx2 = min(idx + 1, len(ref_x) - 1)
        dx = ref_x[idx2] - ref_x[idx]
        dy = ref_y[idx2] - ref_y[idx]
        ref_heading = math.atan2(dy, dx)

        # CTE via cross product
        ex = state.x - ref_x[idx]
        ey = state.y - ref_y[idx]
        cte = -ex * math.sin(ref_heading) + ey * math.cos(ref_heading)

        heading_err = self._wrap_angle(state.psi - ref_heading)

        return float(cte), float(heading_err)

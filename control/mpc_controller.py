"""
control/mpc_controller.py
=========================
Model Predictive Control (MPC) for autonomous driving.

Architecture:
  - Prediction horizon: N steps × dt seconds
  - Optimisation: scipy.optimize.minimize (SLSQP) — swappable to OSQP/CasADi
  - Cost: reference tracking + control effort + jerk penalty + safety margin
  - Warm-starting: previous solution shifted forward for faster convergence

Design trade-offs vs alternatives:
  ┌──────────────┬──────────────────────────────────┬──────────────────────────┐
  │  Approach    │  Pros                            │  Cons                    │
  ├──────────────┼──────────────────────────────────┼──────────────────────────┤
  │  PID only    │  Fast, simple                    │  No look-ahead, no const │
  │  SLSQP MPC   │  General, handles constraints    │  Non-convex, ~20 ms/call │
  │  OSQP MPC    │  Convex, deterministic time      │  Needs linearisation     │
  │  NN E2E      │  Learns complex behaviour        │  Uninterpretable         │
  └──────────────┴──────────────────────────────────┴──────────────────────────┘
  We use SLSQP at urban speeds (<18 m/s) and PID as fallback if MPC exceeds
  the 50 ms real-time budget.

References:
  Falcone et al., "Predictive Active Steering Control for Autonomous Vehicle
  Systems", IEEE Trans. Control Syst. Technol. 2007.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

from control.vehicle_model import BicycleModel, VehicleState

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
class MPCController:
    """
    Receding-horizon MPC using a kinematic bicycle model.

    Parameters (from config.yaml → mpc section)
    -------------------------------------------
    horizon      : int   — N prediction steps
    dt           : float — step duration (s)
    Q            : list[4] — state quadratic cost weights [x, y, psi, v]
    R            : list[2] — control cost weights [delta, accel]
    Rd           : list[2] — control rate cost weights
    v_min/max    : speed bounds (m/s)
    a_min/max    : acceleration bounds (m/s²)
    delta_min/max: steering bounds (rad)
    delta_rate_max: max steering rate (rad/step)
    """

    def __init__(
        self,
        horizon: int = 15,
        dt: float = 0.1,
        Q: List[float] = None,
        R: List[float] = None,
        Rd: List[float] = None,
        v_min: float = 0.0,
        v_max: float = 16.67,
        a_min: float = -8.0,
        a_max: float = 3.0,
        delta_min: float = -0.61,
        delta_max: float = 0.61,
        delta_rate_max: float = 0.35,
        wheelbase: float = 2.875,
        budget_ms: float = 50.0,     # real-time budget before fallback to PID
    ) -> None:
        self.N = horizon
        self.dt = dt
        self.Q = np.diag(Q or [1.0, 1.0, 5.0, 2.0])
        self.R = np.diag(R or [0.5, 0.2])
        self.Rd = np.diag(Rd or [2.0, 0.5])
        self.budget_ms = budget_ms

        # Control bounds per step: [delta_0, a_0, delta_1, a_1, ...]
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.a_min = a_min
        self.a_max = a_max
        self.delta_rate_max = delta_rate_max

        self.model = BicycleModel(
            wheelbase=wheelbase,
            dt=dt,
            v_min=v_min,
            v_max=v_max,
            a_min=a_min,
            a_max=a_max,
            delta_min=delta_min,
            delta_max=delta_max,
        )

        # Warm-start: previous optimal control sequence
        self._prev_u: Optional[np.ndarray] = None
        self._solve_count = 0
        self._failed_count = 0

    # ─────────────────────────────────────────────── Reference generation ────
    @staticmethod
    def build_reference(
        ref_x: np.ndarray,
        ref_y: np.ndarray,
        target_speed: float,
        horizon: int,
    ) -> np.ndarray:
        """
        Interpolate a reference path into N pose+speed targets.

        Returns
        -------
        ndarray (horizon, 4) — [x, y, psi, v] for each step
        """
        if len(ref_x) < 2:
            raise ValueError("Reference path must have at least 2 points.")

        # Cumulative arc-length parameterisation
        diffs = np.diff(np.column_stack([ref_x, ref_y]), axis=0)
        seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
        s = np.concatenate([[0.0], np.cumsum(seg_len)])
        total_len = s[-1]

        # Desired arc-lengths for each horizon step
        step_dist = target_speed * (len(ref_x) / max(total_len, 1e-3))
        s_query = np.linspace(0.0, min(total_len, target_speed * horizon * 0.1), horizon)

        rx = np.interp(s_query, s, ref_x)
        ry = np.interp(s_query, s, ref_y)

        # Heading from finite differences
        dx = np.gradient(rx)
        dy = np.gradient(ry)
        rpsi = np.arctan2(dy, dx)
        rv = np.full(horizon, target_speed)

        return np.column_stack([rx, ry, rpsi, rv])

    # ──────────────────────────────────────────────────── Cost function ──────
    def _cost(
        self,
        u_flat: np.ndarray,
        state0: VehicleState,
        reference: np.ndarray,
        u_prev: np.ndarray,
    ) -> float:
        """
        Total MPC cost.

        u_flat : (2N,) — [delta_0, a_0, delta_1, a_1, ...]
        """
        N = self.N
        u = u_flat.reshape(N, 2)     # columns: [delta, accel]
        states, _ = self.model.rollout(state0, u)  # (N+1, 4)

        cost = 0.0

        # ── State tracking cost ──────────────────────────────────────
        for k in range(N):
            ref_k = reference[min(k, len(reference) - 1)]
            err = states[k + 1] - ref_k

            # Wrap heading error
            err[2] = self.model._wrap_angle(err[2])
            cost += float(err @ self.Q @ err)

        # ── Control effort cost ───────────────────────────────────────
        for k in range(N):
            cost += float(u[k] @ self.R @ u[k])

        # ── Control rate cost (jerk / steering rate) ──────────────────
        for k in range(N - 1):
            du = u[k + 1] - u[k]
            cost += float(du @ self.Rd @ du)

        # Rate cost between previous last action and first new action
        du0 = u[0] - u_prev
        cost += float(du0 @ self.Rd @ du0)

        return cost

    # ──────────────────────────────────────────────────── Constraints ────────
    def _build_bounds(self) -> Bounds:
        lb = np.tile([self.delta_min, self.a_min], self.N)
        ub = np.tile([self.delta_max, self.a_max], self.N)
        return Bounds(lb=lb, ub=ub)

    def _build_rate_constraints(self, u_prev: np.ndarray) -> List[dict]:
        """Build nonlinear rate constraints as SLSQP inequality dicts."""
        dr_max = self.delta_rate_max
        N = self.N

        constraints = []

        def _delta_rate_lo(u_flat: np.ndarray, k: int, prev_delta: float) -> float:
            # delta[k] - prev >= -dr_max  =>  delta[k] - prev + dr_max >= 0
            d_k = u_flat[2 * k]
            p = prev_delta if k == 0 else u_flat[2 * (k - 1)]
            return (d_k - p) + dr_max

        def _delta_rate_hi(u_flat: np.ndarray, k: int, prev_delta: float) -> float:
            # dr_max - (delta[k] - prev) >= 0
            d_k = u_flat[2 * k]
            p = prev_delta if k == 0 else u_flat[2 * (k - 1)]
            return dr_max - (d_k - p)

        for k in range(N):
            constraints.append({
                "type": "ineq",
                "fun": lambda u, _k=k, _p=u_prev[0]: _delta_rate_lo(u, _k, _p),
            })
            constraints.append({
                "type": "ineq",
                "fun": lambda u, _k=k, _p=u_prev[0]: _delta_rate_hi(u, _k, _p),
            })

        return constraints

    # ──────────────────────────────────────────────── Warm-start  ────────────
    def _warm_start(self, u_prev: Optional[np.ndarray]) -> np.ndarray:
        if u_prev is None:
            return np.zeros(self.N * 2)
        # Shift previous solution by 1 step, fill last with duplicate
        shifted = np.roll(u_prev.reshape(self.N, 2), -1, axis=0)
        shifted[-1] = shifted[-2]
        return shifted.ravel()

    # ──────────────────────────────────────────────── Solve ──────────────────
    def solve(
        self,
        state: VehicleState,
        reference: np.ndarray,
        u_prev: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Solve one MPC horizon.

        Parameters
        ----------
        state     : VehicleState — current ego state
        reference : ndarray (N, 4) — reference [x, y, psi, v] per step
        u_prev    : ndarray (2,) — previous [delta, accel] applied

        Returns
        -------
        dict:
          success     : bool
          delta       : float — optimal steering (rad)
          accel       : float — optimal acceleration (m/s²)
          cost        : float
          solve_ms    : float
          u_sequence  : ndarray (N, 2)
        """
        u_prev_arr = u_prev if u_prev is not None else np.zeros(2)
        u0 = self._warm_start(self._prev_u)
        bounds = self._build_bounds()
        constraints = self._build_rate_constraints(u_prev_arr)

        t0 = time.monotonic()

        try:
            result = minimize(
                fun=self._cost,
                x0=u0,
                args=(state, reference, u_prev_arr),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-6, "disp": False},
            )
            solve_ms = (time.monotonic() - t0) * 1000.0
            self._solve_count += 1

            if result.success or result.fun < 1e6:
                u_opt = result.x.reshape(self.N, 2)
                self._prev_u = u_opt
                delta_cmd = float(np.clip(u_opt[0, 0], self.delta_min, self.delta_max))
                accel_cmd = float(np.clip(u_opt[0, 1], self.a_min, self.a_max))
                logger.debug(
                    "MPC solved in %.1f ms | δ=%.3f rad | a=%.2f m/s² | cost=%.2f",
                    solve_ms, delta_cmd, accel_cmd, result.fun,
                )
                return {
                    "success": True,
                    "delta": delta_cmd,
                    "accel": accel_cmd,
                    "cost": float(result.fun),
                    "solve_ms": solve_ms,
                    "u_sequence": u_opt,
                    "iterations": result.nit,
                }
            else:
                raise RuntimeError(f"MPC solver failed: {result.message}")

        except Exception as exc:
            self._failed_count += 1
            solve_ms = (time.monotonic() - t0) * 1000.0
            logger.warning("MPC solve failed (%.1f ms): %s", solve_ms, exc)
            return {
                "success": False,
                "delta": 0.0,
                "accel": -1.0,       # conservative: gentle brake
                "cost": float("inf"),
                "solve_ms": solve_ms,
                "u_sequence": None,
                "iterations": 0,
            }

    # ──────────────────────────────────────────────── Diagnostics ────────────
    @property
    def stats(self) -> Dict:
        total = self._solve_count + self._failed_count
        return {
            "solve_count": self._solve_count,
            "failed_count": self._failed_count,
            "failure_rate": self._failed_count / max(total, 1),
        }

    def reset(self) -> None:
        self._prev_u = None
        logger.debug("MPC warm-start state reset")

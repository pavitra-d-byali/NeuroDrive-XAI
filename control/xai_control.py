"""
control/xai_control.py
======================
Explainable AI for the Vehicle Control module using SHAP.

Method: KernelSHAP (model-agnostic) applied to the full hybrid controller.
  - Unlike Grad-CAM (perception), control decisions depend on structured
    tabular features, not spatial activations → SHAP is more interpretable.
  - We explain EACH control output independently: steering, throttle, brake.
  - Background dataset: 100 representative frames from the training set.

Explanation types:
  1. Local (per-frame): Which feature most influenced THIS control decision?
  2. Global (aggregate): Feature importance over an entire episode.
  3. Force plot data: For dashboard visualisation (JSON-serialisable).

Output example (frame with emergency brake):
  Feature contributions to brake=0.92:
    closest_dist    : -0.48  (close object → high brake)
    speed_mps       : -0.21  (high speed → more braking)
    heading_err     : +0.03  (aligned → less panic)
    curvature       : -0.08  (curve → extra caution)
    ...

Integration: The XAI module runs asynchronously every N frames (default 10)
  to avoid blocking the 20 Hz control loop.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "speed_mps",
    "long_accel",
    "curvature",
    "prev_steering",
    "cte",
    "heading_err",
    "closest_dist",
    "closest_v_rel",
    "num_agents",
]

OUTPUT_NAMES = ["steering", "throttle", "brake"]


# ─────────────────────────────────────────────────────────── Controller shim ──
class ControllerPredictor:
    """
    Wraps the hybrid controller as a callable f(X) → Y for SHAP.

    X : ndarray (N, 9)  — feature rows
    Y : ndarray (N, 3)  — [steering_norm, throttle, brake]
    """

    def __init__(self, controller_fn: Callable) -> None:
        """
        Parameters
        ----------
        controller_fn : Callable
            Function that takes a dict of features and returns
            dict {steering, throttle, brake}.
        """
        self.fn = controller_fn

    def __call__(self, X: np.ndarray) -> np.ndarray:
        results = []
        for row in X:
            feat = dict(zip(FEATURE_NAMES, row))
            try:
                out = self.fn(feat)
                results.append([
                    out.get("steering", 0.0),
                    out.get("throttle", 0.0),
                    out.get("brake", 0.0),
                ])
            except Exception as e:
                logger.warning("Predictor error: %s", e)
                results.append([0.0, 0.0, 0.0])
        return np.array(results, dtype=np.float32)


# ─────────────────────────────────────────────────────── SHAP explainer ───────
class ControlXAI:
    """
    SHAP-based explainability for the vehicle control module.

    Parameters
    ----------
    controller_fn  : Callable — maps feature dict → control dict
    background_X   : ndarray (K, 9) — background dataset for KernelSHAP
                     (K=100 is a good trade-off between accuracy and speed)
    """

    def __init__(
        self,
        controller_fn: Callable,
        background_X: Optional[np.ndarray] = None,
    ) -> None:
        try:
            import shap
            self._shap = shap
        except ImportError:
            raise ImportError(
                "SHAP not installed. Run: pip install shap\n"
                "Note: SHAP requires numpy, scipy, and tqdm."
            )

        self.predictor = ControllerPredictor(controller_fn)

        if background_X is None:
            # Use a zero-velocity stationary state as single background
            background_X = np.zeros((1, 9), dtype=np.float32)
            background_X[0, 6] = 50.0   # closest_dist default
            logger.warning("Using default background (no dataset provided). "
                           "Pass background_X from training data for better SHAP accuracy.")

        self.background_X = background_X

        logger.info("Building KernelSHAP explainer with %d background samples …", len(background_X))
        self.explainer = shap.KernelExplainer(
            self.predictor,
            shap.kmeans(background_X, min(20, len(background_X))),  # k-means summary
        )
        logger.info("KernelSHAP explainer ready")

        # History for global importance
        self._history_shap: List[np.ndarray] = []
        self._history_features: List[np.ndarray] = []

    # ──────────────────────────────────────────────────── Local explanation ──
    def explain_frame(
        self,
        features: np.ndarray,     # (9,) normalised
        n_samples: int = 100,
    ) -> Dict:
        """
        Compute SHAP values for a single frame.

        Returns
        -------
        dict:
          shap_values    : list[3] of list[9]  — [steering, throttle, brake] × features
          base_values    : list[3]             — expected output values
          contributions  : list[dict]          — human-readable per-output explanations
          dominant_feature : str              — most influential feature for brake
        """
        x = features.reshape(1, -1)
        shap_vals = self.explainer.shap_values(x, nsamples=n_samples, silent=True)
        # shap_vals: list of 3 arrays, each (1, 9)

        self._history_shap.append(np.array([sv[0] for sv in shap_vals]))   # (3, 9)
        self._history_features.append(features)

        contributions = []
        for i, out_name in enumerate(OUTPUT_NAMES):
            sv = shap_vals[i][0]    # (9,)
            ranked = sorted(
                zip(FEATURE_NAMES, sv.tolist()),
                key=lambda t: abs(t[1]),
                reverse=True,
            )
            contributions.append({
                "output": out_name,
                "base_value": float(self.explainer.expected_value[i]),
                "ranked_features": [
                    {"feature": n, "shap_value": round(v, 5)} for n, v in ranked
                ],
            })

        # Most dominant feature for brake (safety-critical)
        brake_sv = shap_vals[2][0]
        dominant_idx = int(np.argmax(np.abs(brake_sv)))
        dominant_feature = FEATURE_NAMES[dominant_idx]

        return {
            "shap_values":      [[v for v in sv[0]] for sv in shap_vals],
            "base_values":      [float(self.explainer.expected_value[i]) for i in range(3)],
            "contributions":    contributions,
            "dominant_feature": dominant_feature,
        }

    # ──────────────────────────────────────────────────── Global importance ──
    def global_importance(self) -> Dict:
        """
        Aggregate SHAP values across all recorded frames.

        Returns
        -------
        dict:
          per_output : dict — {steering/throttle/brake: ranked feature list}
          combined   : ranked list of overall feature importance
        """
        if not self._history_shap:
            return {"error": "No frames recorded yet"}

        all_shap = np.stack(self._history_shap)    # (T, 3, 9)
        mean_abs = np.mean(np.abs(all_shap), axis=0)  # (3, 9)

        per_output = {}
        for i, name in enumerate(OUTPUT_NAMES):
            ranked = sorted(
                zip(FEATURE_NAMES, mean_abs[i].tolist()),
                key=lambda t: t[1], reverse=True,
            )
            per_output[name] = [{"feature": f, "importance": round(v, 5)} for f, v in ranked]

        combined_imp = mean_abs.mean(axis=0)
        combined_ranked = sorted(
            zip(FEATURE_NAMES, combined_imp.tolist()),
            key=lambda t: t[1], reverse=True,
        )

        return {
            "per_output": per_output,
            "combined":   [{"feature": f, "importance": round(v, 5)} for f, v in combined_ranked],
            "n_frames":   len(self._history_shap),
        }

    # ──────────────────────────────────────────────────── Save / load ────────
    def save_report(self, path: str) -> None:
        """Save global importance report as JSON."""
        report = self.global_importance()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("XAI report saved → %s", path)

    def print_frame_explanation(self, explanation: Dict, frame_idx: int = 0) -> None:
        print(f"\n{'─'*55}")
        print(f"  Control XAI — Frame {frame_idx}")
        print(f"{'─'*55}")
        for contrib in explanation["contributions"]:
            out = contrib["output"]
            print(f"\n  [{out.upper()}] (base: {contrib['base_value']:.3f})")
            for item in contrib["ranked_features"][:5]:   # top-5
                arrow = "↑" if item["shap_value"] > 0 else "↓"
                print(f"    {item['feature']:<20} {arrow}  {item['shap_value']:+.4f}")
        print(f"\n  Dominant safety feature: {explanation['dominant_feature']}")
        print(f"{'─'*55}\n")

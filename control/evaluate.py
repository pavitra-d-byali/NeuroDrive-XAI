"""
control/evaluate.py
===================
Evaluation metrics for the Vehicle Control module.

Metrics:
  Longitudinal:
    - Speed RMSE   (m/s)
    - Speed MAE    (m/s)

  Lateral:
    - Cross-track Error (CTE) RMSE   (m)
    - Heading Error RMSE             (rad)
    - Max lateral deviation          (m)

  Comfort:
    - Jerk RMS                      (m/s³)
    - Steering rate RMS              (rad/s)

  Safety:
    - Collision count  (if ground truth includes collision flags)
    - TTC < 2s rate    (time-to-collision below safety threshold)

  Control quality:
    - Steering MAE     (normalised [-1, 1])
    - Throttle MAE
    - Brake F1 score   (binary: braking vs not braking)

Benchmark ranges (expected for production-grade controllers):
  CTE RMSE      < 0.30 m       (urban), < 0.15 m (highway)
  Heading RMSE  < 3.0°
  Speed RMSE    < 1.5 m/s
  Jerk RMS      < 2.0 m/s³    (ISO 2631 comfort)
  Brake F1      > 0.90
"""

from __future__ import annotations

import math
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────── per-frame buffers ────
class ControlMetrics:
    """
    Accumulates per-frame ground truth and predictions,
    then computes all metrics at once.

    Usage
    -----
    >>> m = ControlMetrics()
    >>> for frame in frames:
    ...     m.update(pred_steer, gt_steer, pred_thr, gt_thr,
    ...               pred_brk, gt_brk, cte, heading_err,
    ...               pred_speed, gt_speed, jerk, steer_rate)
    >>> report = m.compute()
    """

    def __init__(self) -> None:
        self._pred_steer: List[float] = []
        self._gt_steer:   List[float] = []
        self._pred_thr:   List[float] = []
        self._gt_thr:     List[float] = []
        self._pred_brk:   List[float] = []
        self._gt_brk:     List[float] = []
        self._cte:        List[float] = []
        self._head_err:   List[float] = []
        self._pred_spd:   List[float] = []
        self._gt_spd:     List[float] = []
        self._jerks:      List[float] = []
        self._steer_rates: List[float] = []
        self._collision_flags: List[int] = []

    def update(
        self,
        pred_steer: float,
        gt_steer:   float,
        pred_thr:   float,
        gt_thr:     float,
        pred_brk:   float,
        gt_brk:     float,
        cte:        float,
        heading_err: float,
        pred_speed: float,
        gt_speed:   float,
        jerk:       float = 0.0,
        steer_rate: float = 0.0,
        collision:  int = 0,
    ) -> None:
        self._pred_steer.append(pred_steer)
        self._gt_steer.append(gt_steer)
        self._pred_thr.append(pred_thr)
        self._gt_thr.append(gt_thr)
        self._pred_brk.append(pred_brk)
        self._gt_brk.append(gt_brk)
        self._cte.append(cte)
        self._head_err.append(heading_err)
        self._pred_spd.append(pred_speed)
        self._gt_spd.append(gt_speed)
        self._jerks.append(jerk)
        self._steer_rates.append(steer_rate)
        self._collision_flags.append(int(collision))

    def compute(self) -> Dict[str, float]:
        """Compute and return all metrics as a flat dict."""
        if not self._cte:
            logger.warning("No data accumulated. Returning empty metrics.")
            return {}

        cte_arr  = np.array(self._cte)
        head_arr = np.array(self._head_err)
        spd_pred = np.array(self._pred_spd)
        spd_gt   = np.array(self._gt_spd)
        steer_p  = np.array(self._pred_steer)
        steer_g  = np.array(self._gt_steer)
        thr_p    = np.array(self._pred_thr)
        thr_g    = np.array(self._gt_thr)
        brk_p    = np.array(self._pred_brk)
        brk_g    = np.array(self._gt_brk)
        jerks    = np.array(self._jerks)
        s_rates  = np.array(self._steer_rates)

        # Binary brake (threshold = 0.1)
        brk_pred_bin = (brk_p > 0.1).astype(int)
        brk_gt_bin   = (brk_g > 0.1).astype(int)

        results: Dict[str, float] = {}

        # ── Lateral ───────────────────────────────────────────────────
        results["cte_rmse_m"]      = float(np.sqrt(np.mean(cte_arr ** 2)))
        results["cte_mae_m"]       = float(np.mean(np.abs(cte_arr)))
        results["cte_max_m"]       = float(np.max(np.abs(cte_arr)))
        results["heading_rmse_deg"] = float(
            math.degrees(np.sqrt(np.mean(head_arr ** 2)))
        )

        # ── Longitudinal ──────────────────────────────────────────────
        results["speed_rmse_mps"] = float(np.sqrt(np.mean((spd_pred - spd_gt) ** 2)))
        results["speed_mae_mps"]  = float(np.mean(np.abs(spd_pred - spd_gt)))

        # ── Control outputs ───────────────────────────────────────────
        results["steer_mae"]    = float(np.mean(np.abs(steer_p - steer_g)))
        results["throttle_mae"] = float(np.mean(np.abs(thr_p - thr_g)))
        results["brake_mae"]    = float(np.mean(np.abs(brk_p - brk_g)))

        if len(np.unique(brk_gt_bin)) > 1:
            results["brake_f1"]        = float(f1_score(brk_gt_bin, brk_pred_bin, zero_division=0))
            results["brake_precision"]  = float(precision_score(brk_gt_bin, brk_pred_bin, zero_division=0))
            results["brake_recall"]     = float(recall_score(brk_gt_bin, brk_pred_bin, zero_division=0))
        else:
            results["brake_f1"] = results["brake_precision"] = results["brake_recall"] = float("nan")

        # ── Comfort ───────────────────────────────────────────────────
        if np.any(jerks != 0.0):
            results["jerk_rms_mps3"]    = float(np.sqrt(np.mean(jerks ** 2)))
        if np.any(s_rates != 0.0):
            results["steer_rate_rms"]   = float(np.sqrt(np.mean(s_rates ** 2)))

        # ── Safety ───────────────────────────────────────────────────
        results["collision_count"] = int(sum(self._collision_flags))
        results["num_frames"]      = len(self._cte)

        return results

    def print_report(self) -> None:
        r = self.compute()
        print("\n" + "=" * 55)
        print("  NeuroDrive Control Module — Evaluation Report")
        print("=" * 55)
        print(f"  Frames evaluated       : {r.get('num_frames', 0)}")
        print(f"\n  [LATERAL]")
        print(f"  CTE RMSE               : {r.get('cte_rmse_m', 0):.4f} m   (target < 0.30)")
        print(f"  CTE MAE                : {r.get('cte_mae_m', 0):.4f} m")
        print(f"  CTE Max                : {r.get('cte_max_m', 0):.4f} m")
        print(f"  Heading RMSE           : {r.get('heading_rmse_deg', 0):.2f}°  (target < 3.0°)")
        print(f"\n  [LONGITUDINAL]")
        print(f"  Speed RMSE             : {r.get('speed_rmse_mps', 0):.4f} m/s (target < 1.5)")
        print(f"  Speed MAE              : {r.get('speed_mae_mps', 0):.4f} m/s")
        print(f"\n  [CONTROL ACCURACY]")
        print(f"  Steering MAE           : {r.get('steer_mae', 0):.4f}")
        print(f"  Throttle MAE           : {r.get('throttle_mae', 0):.4f}")
        print(f"  Brake F1               : {r.get('brake_f1', float('nan')):.4f} (target > 0.90)")
        print(f"  Brake Precision        : {r.get('brake_precision', float('nan')):.4f}")
        print(f"  Brake Recall           : {r.get('brake_recall', float('nan')):.4f}")
        print(f"\n  [COMFORT]")
        print(f"  Jerk RMS               : {r.get('jerk_rms_mps3', 0):.4f} m/s³ (target < 2.0)")
        print(f"  Steer Rate RMS         : {r.get('steer_rate_rms', 0):.4f} rad/s")
        print(f"\n  [SAFETY]")
        print(f"  Collision count        : {r.get('collision_count', 0)}")
        print("=" * 55 + "\n")

    def reset(self) -> None:
        self.__init__()


# ─────────────────────────────────────── Standalone benchmark (offline) ───────
def offline_benchmark(
    pred_csv: str,
    gt_csv: str,
    separator: str = ",",
) -> Dict[str, float]:
    """
    Compare controller output CSV vs ground-truth CSV.

    Expected columns in both CSVs:
      steering_norm, throttle, brake, speed, cte, heading_err

    Parameters
    ----------
    pred_csv : str — path to controller output CSV
    gt_csv   : str — path to ground truth CSV
    """
    import csv as csv_mod

    def _load(path: str):
        rows = []
        with open(path, newline="") as f:
            reader = csv_mod.DictReader(f, delimiter=separator)
            for row in reader:
                rows.append({k: float(v) for k, v in row.items()})
        return rows

    preds = _load(pred_csv)
    gts   = _load(gt_csv)
    n = min(len(preds), len(gts))

    metrics = ControlMetrics()
    prev_speed = 0.0
    prev_steer = 0.0
    dt = 0.05   # assume 20 Hz

    for i in range(n):
        p, g = preds[i], gts[i]

        jerk = abs(p.get("speed", 0.0) - prev_speed) / dt
        steer_rate = abs(p.get("steering_norm", 0.0) - prev_steer) / dt

        metrics.update(
            pred_steer=p.get("steering_norm", 0.0),
            gt_steer=g.get("steering_norm", 0.0),
            pred_thr=p.get("throttle", 0.0),
            gt_thr=g.get("throttle", 0.0),
            pred_brk=p.get("brake", 0.0),
            gt_brk=g.get("brake", 0.0),
            cte=p.get("cte", 0.0),
            heading_err=p.get("heading_err", 0.0),
            pred_speed=p.get("speed", 0.0),
            gt_speed=g.get("speed", 0.0),
            jerk=jerk,
            steer_rate=steer_rate,
        )
        prev_speed = p.get("speed", 0.0)
        prev_steer = p.get("steering_norm", 0.0)

    metrics.print_report()
    return metrics.compute()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Predicted control CSV")
    parser.add_argument("--gt",   required=True, help="Ground-truth control CSV")
    args = parser.parse_args()
    offline_benchmark(args.pred, args.gt)

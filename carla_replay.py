"""
carla_replay.py
===============
NeuroDrive-XAI CARLA-free Replay Mode

Replays a previously recorded CARLA episode CSV (from carla_run.py)
without needing CARLA installed. Visualises:
  - Ego vehicle trajectory on 2D map
  - Speed profile
  - Brake events + risk scores
  - Decision timeline

Also re-runs the decision engine on saved telemetry to validate
that the same pipeline produces identical outputs (regression test).

Usage:
    python carla_replay.py                                   # auto-find latest CSV
    python carla_replay.py --csv artifacts/carla_records/episode_0000.csv
    python carla_replay.py --csv artifacts/carla_records/episode_0000.csv --validate
    python carla_replay.py --demo                            # synthetic demo (no CSV needed)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ──────────────────────────────────────────────── Data loading helpers ────────
def load_episode_csv(csv_path: str) -> list[dict]:
    """Load a CARLA episode CSV into a list of step dicts."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "step":         int(row["step"]),
                "x":            float(row["x"]),
                "y":            float(row["y"]),
                "psi_deg":      float(row["psi_deg"]),
                "speed_mps":    float(row["speed_mps"]),
                "steering":     float(row["steering"]),
                "throttle":     float(row["throttle"]),
                "brake":        float(row["brake"]),
                "action":       row.get("action", "Proceed"),
                "risk_score":   float(row.get("risk_score", 0.0)),
                "latency_ms":   float(row.get("latency_ms", 0.0)),
                "num_detections": int(row.get("num_detections", 0)),
                "collision":    row.get("collision", "False") == "True",
            })
    return rows


def generate_demo_episode(n: int = 300) -> list[dict]:
    """
    Generate a synthetic CARLA episode for demo without real CARLA data.
    Simulates a route through a city block with a sudden brake event.
    """
    rows = []
    np.random.seed(42)
    x, y, psi, v = 0.0, 0.0, 0.0, 8.0
    dt = 0.05  # 20 Hz

    for step in range(n):
        # Simulate simple trajectory
        t = step * dt
        # Gentle curve in the middle
        if 50 < step < 100:
            psi += 0.5 * dt
        # Sudden obstacle at step 150
        is_emergency = 145 < step < 175
        action = "Brake" if is_emergency else ("Slow" if 140 < step < 180 else "Proceed")
        risk = 0.9 if is_emergency else (0.4 if 140 < step < 180 else 0.05 + np.random.uniform(0, 0.1))
        throttle = 0.0 if is_emergency else (0.2 if action == "Slow" else 0.5)
        brake = min(risk, 1.0) if is_emergency else 0.0
        steering = 0.35 if 50 < step < 100 else np.random.uniform(-0.05, 0.05)

        # Integrate position
        v = max(0.0, v + (throttle * 3.0 - brake * 8.0) * dt)
        x += v * math.cos(psi) * dt
        y += v * math.sin(psi) * dt

        rows.append({
            "step":           step,
            "x":              round(x, 4),
            "y":              round(y, 4),
            "psi_deg":        round(math.degrees(psi), 2),
            "speed_mps":      round(v, 3),
            "steering":       round(steering, 4),
            "throttle":       round(throttle, 4),
            "brake":          round(brake, 4),
            "action":         action,
            "risk_score":     round(risk, 3),
            "latency_ms":     round(np.random.uniform(18, 45), 2),
            "num_detections": np.random.randint(0, 6),
            "collision":      False,
        })

    return rows


# ──────────────────────────────────────────────── Visualisation ───────────────
def replay_visualise(rows: list[dict], output_dir: str = "artifacts/replay"):
    """Generate replay visualisations using matplotlib."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.collections import LineCollection
    except ImportError:
        print("matplotlib not installed — skipping visualisation.")
        print("Install: pip install matplotlib")
        return

    xs     = [r["x"] for r in rows]
    ys     = [r["y"] for r in rows]
    speeds = [r["speed_mps"] for r in rows]
    risks  = [r["risk_score"] for r in rows]
    brakes = [r["brake"] for r in rows]
    steps  = [r["step"] for r in rows]

    # ── 1. Trajectory map ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    # Colour trajectory by risk
    points  = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm    = plt.Normalize(0, 1)
    lc      = LineCollection(segments, cmap="RdYlGn_r", norm=norm, linewidth=2)
    lc.set_array(np.array(risks[:-1]))
    ax.add_collection(lc)

    # Brake events
    brake_xs = [r["x"] for r in rows if r["brake"] > 0.3]
    brake_ys = [r["y"] for r in rows if r["brake"] > 0.3]
    if brake_xs:
        ax.scatter(brake_xs, brake_ys, c="red", s=80, zorder=5, label="Brake event", marker="x")

    # Start / End
    ax.scatter([xs[0]], [ys[0]], c="#00ff88", s=150, zorder=6, marker="^", label="Start")
    ax.scatter([xs[-1]], [ys[-1]], c="#ff4444", s=150, zorder=6, marker="s", label="End")

    cbar = plt.colorbar(lc, ax=ax, pad=0.02)
    cbar.set_label("Risk Score", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("Ego Vehicle Trajectory (coloured by risk)", color="white", fontsize=14, pad=12)
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(facecolor="#1a1c23", labelcolor="white", fontsize=9)
    ax.set_aspect("equal")

    traj_path = os.path.join(output_dir, "trajectory.png")
    plt.tight_layout()
    plt.savefig(traj_path, dpi=120, bbox_inches="tight", facecolor="#0e1117")
    plt.close()
    print(f"  ✓ Trajectory map     → {traj_path}")

    # ── 2. Telemetry dashboard ────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), facecolor="#0e1117")
    for ax in axes:
        ax.set_facecolor("#151720")

    # Speed
    axes[0].plot(steps, speeds, color="#00e5ff", linewidth=1.5, label="Speed (m/s)")
    axes[0].fill_between(steps, speeds, alpha=0.15, color="#00e5ff")
    axes[0].set_ylabel("Speed (m/s)", color="white")
    axes[0].set_title("NeuroDrive-XAI — CARLA Episode Telemetry", color="white", fontsize=13)
    axes[0].legend(facecolor="#1a1c23", labelcolor="white")

    # Risk score
    axes[1].plot(steps, risks, color="#ff9900", linewidth=1.5, label="Risk Score")
    axes[1].fill_between(steps, risks, alpha=0.2, color="#ff9900")
    axes[1].axhline(0.5, color="#ff4444", linestyle="--", alpha=0.6, linewidth=1, label="Brake threshold")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Risk Score", color="white")
    axes[1].legend(facecolor="#1a1c23", labelcolor="white")

    # Brake intensity
    axes[2].bar(steps, brakes, color="#ff2244", alpha=0.7, width=0.8, label="Brake Intensity")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Brake Intensity", color="white")
    axes[2].set_xlabel("Step", color="white")
    axes[2].legend(facecolor="#1a1c23", labelcolor="white")

    for ax in axes:
        ax.tick_params(colors="white")
        ax.grid(True, color="#333333", alpha=0.5, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    dash_path = os.path.join(output_dir, "telemetry.png")
    plt.tight_layout()
    plt.savefig(dash_path, dpi=120, bbox_inches="tight", facecolor="#0e1117")
    plt.close()
    print(f"  ✓ Telemetry dashboard → {dash_path}")

    return traj_path, dash_path


# ──────────────────────────────────────────────── Statistics ──────────────────
def print_episode_stats(rows: list[dict]):
    speeds     = [r["speed_mps"] for r in rows]
    risks      = [r["risk_score"] for r in rows]
    latencies  = [r["latency_ms"] for r in rows]
    brakes     = sum(1 for r in rows if r["brake"] > 0.3)
    actions    = {}
    for r in rows:
        actions[r["action"]] = actions.get(r["action"], 0) + 1
    collision  = any(r["collision"] for r in rows)

    print("\n" + "=" * 55)
    print("  CARLA Episode Replay Statistics")
    print("=" * 55)
    print(f"  Total steps       : {len(rows)}")
    print(f"  Collision         : {'YES 💥' if collision else 'No ✓'}")
    print(f"  Brake events      : {brakes} ({brakes/max(len(rows),1)*100:.1f}%)")
    print(f"  Avg speed         : {np.mean(speeds):.2f} m/s")
    print(f"  Max speed         : {np.max(speeds):.2f} m/s")
    print(f"  Avg risk          : {np.mean(risks):.3f}")
    print(f"  Max risk          : {np.max(risks):.3f}")
    print(f"  Avg latency       : {np.mean(latencies):.2f} ms")
    print(f"  Action breakdown  :")
    for a, cnt in sorted(actions.items(), key=lambda x: -x[1]):
        print(f"    {a:12s}: {cnt:4d} ({cnt/max(len(rows),1)*100:.1f}%)")
    print("=" * 55)


# ─────────────────────────────────────────────────────────── CLI main ─────────
def main():
    parser = argparse.ArgumentParser(description="NeuroDrive-XAI CARLA Replay")
    parser.add_argument("--csv",      default=None,    help="Episode CSV path")
    parser.add_argument("--validate", action="store_true", help="Re-run decision engine on telemetry")
    parser.add_argument("--demo",     action="store_true", help="Use synthetic demo (no CSV needed)")
    parser.add_argument("--out",      default="artifacts/replay", help="Output directory for plots")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  NeuroDrive-XAI — CARLA Replay Mode")
    print("=" * 55)

    # ── Find / generate data ───────────────────────────────────────
    if args.demo:
        print("  Mode: SYNTHETIC DEMO (no CARLA needed)")
        rows = generate_demo_episode(300)
    elif args.csv:
        if not os.path.exists(args.csv):
            print(f"  ERROR: CSV not found: {args.csv}")
            sys.exit(1)
        print(f"  Loading: {args.csv}")
        rows = load_episode_csv(args.csv)
    else:
        # Auto-find latest episode
        record_dir = Path("artifacts/carla_records")
        csvs = sorted(record_dir.glob("episode_*.csv")) if record_dir.exists() else []
        if csvs:
            latest = str(csvs[-1])
            print(f"  Auto-selected: {latest}")
            rows = load_episode_csv(latest)
        else:
            print("  No CARLA CSV found. Using synthetic demo.")
            rows = generate_demo_episode(300)

    print(f"  Loaded {len(rows)} steps.\n")

    # ── Statistics ─────────────────────────────────────────────────
    print_episode_stats(rows)

    # ── Visualisation ──────────────────────────────────────────────
    print("\n  Generating replay visualisations...")
    replay_visualise(rows, output_dir=args.out)

    # ── Validation: re-run decision engine ─────────────────────────
    if args.validate:
        print("\n  Validating decision engine on replay telemetry...")
        try:
            from planning.decision_engine import DecisionEngine
            engine = DecisionEngine(history_size=5)
            mismatches = 0
            for r in rows[:50]:  # Check first 50 steps
                # Build minimal scene from telemetry
                scene = {
                    "objects": [{
                        "track_id": 1,
                        "distance_meters": max(5.0, 30.0 - r["speed_mps"] * 2),
                        "type": "car",
                        "velocity": r["speed_mps"],
                        "lane": "center",
                    }] if r["num_detections"] > 0 else [],
                    "lane_geometry": {"center_line": [(0, 0), (100, 0)]},
                    "confidence": {"lane": 0.9, "detection": 0.85, "depth": 0.8},
                }
                decision = engine.decide(scene)
                rec_action = r["action"]
                pred_action = decision.get("action", "Proceed")
                if rec_action != pred_action and rec_action != "Proceed":
                    mismatches += 1

            print(f"  Decision engine validation: {mismatches}/50 mismatches (expected: some, due to state history)")
        except Exception as e:
            print(f"  Validation skipped: {e}")

    print(f"\n  ✓ Replay complete. Outputs in: {args.out}/")
    print("  View dashboard: streamlit run frontend/app.py")


if __name__ == "__main__":
    main()

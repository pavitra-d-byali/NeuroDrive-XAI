"""
carla_run.py
============
NeuroDrive-XAI — CARLA Simulation Launcher

Full closed-loop autonomous driving pipeline inside CARLA:
  CARLA camera → PerceptionModule → DecisionEngine → VehicleController → CARLA apply_control()

Prerequisites:
  1. CARLA server running:
       Windows: CarlaUE4.exe
       Linux:   ./CarlaUE4.sh -RenderOffScreen
  2. CARLA Python client:
       pip install carla
     Or set: set CARLA_EGG=C:/CARLA/PythonAPI/carla/dist/carla-*.egg
  3. NeuroDrive models trained:
       python decision/train.py
       python setup_models.py

Usage:
    python carla_run.py                          # 3 episodes, Town03
    python carla_run.py --map Town05 --episodes 5 --steps 1000
    python carla_run.py --weather HeavyRain       # Adverse weather test
    python carla_run.py --npc 15 --pedestrians 8  # Dense traffic
    python carla_run.py --record-video            # Save dashcam video
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("NeuroDrive.CARLA")


def _import_carla():
    try:
        import carla
        return carla
    except ImportError:
        eggs = [
            "C:/CARLA/PythonAPI/carla/dist",
            "/opt/carla-simulator/PythonAPI/carla/dist",
            os.environ.get("CARLA_EGG", ""),
        ]
        for base in eggs:
            if not base:
                continue
            from pathlib import Path as P
            for egg in P(base).glob("carla-*.egg"):
                sys.path.insert(0, str(egg))
                try:
                    import carla
                    return carla
                except ImportError:
                    continue
        raise ImportError(
            "\n[ERROR] CARLA Python API not found.\n"
            "Install: pip install carla\n"
            "Or set:  set CARLA_EGG=C:/CARLA/PythonAPI/carla/dist/carla-*.egg\n"
            "Or run:  python carla_replay.py  (no CARLA needed)"
        )


# ──────────────────────────────────────────── Pipeline initialisation ─────────
def init_pipeline(config_path: str = "control/config.yaml"):
    """Load all NeuroDrive-XAI modules."""
    print("\n[Init] Loading NeuroDrive-XAI pipeline modules...")

    from perception.hybridnets_wrapper import PerceptionModule
    from perception.tracker import ObjectTracker
    from perception.depth_estimator import DepthEstimator
    from perception.lane_detector import LaneDetector
    from scene_representation.scene_builder import SceneBuilder
    from planning.decision_engine import DecisionEngine
    from planning.controller import VehicleController
    from planning.trajectory_planner import TrajectoryPlanner
    from prediction.motion_predictor import MotionPredictor
    from explainability.gradcam import PerceptionXAI
    from explainability.reasoning_engine import ReasoningEngine
    from explainability.uncertainty_module import UncertaintyModule
    from visualization.visualizer import Visualizer
    from evaluation.metrics import PerformanceMetrics
    from logs.pipeline_logger import PipelineLogger

    perception       = PerceptionModule(use_cuda=False)
    tracker          = ObjectTracker()
    depth_estimator  = DepthEstimator(use_cuda=False)
    lane_detector    = LaneDetector()
    scene_builder    = SceneBuilder()
    decision_engine  = DecisionEngine(history_size=5)
    controller       = VehicleController(config_path=config_path, use_hybrid=True)
    trajectory       = TrajectoryPlanner()
    motion_predictor = MotionPredictor(fps=20)
    xai              = PerceptionXAI(perception)
    reasoner         = ReasoningEngine()
    uncertainty      = UncertaintyModule()
    visualizer       = Visualizer()
    metrics          = PerformanceMetrics()
    logger_mod       = PipelineLogger()

    print("[Init] ✓ All modules loaded.\n")

    return {
        "perception":       perception,
        "tracker":          tracker,
        "depth_estimator":  depth_estimator,
        "lane_detector":    lane_detector,
        "scene_builder":    scene_builder,
        "decision_engine":  decision_engine,
        "controller":       controller,
        "trajectory":       trajectory,
        "motion_predictor": motion_predictor,
        "xai":              xai,
        "reasoner":         reasoner,
        "uncertainty":      uncertainty,
        "visualizer":       visualizer,
        "metrics":          metrics,
        "logger":           logger_mod,
    }


# ──────────────────────────────────────────────── Per-step inference ──────────
def process_frame(frame: np.ndarray, mods: dict, frame_idx: int, depth_map: Optional[np.ndarray] = None):
    """Run one full perception→decision→control inference step."""
    mods["metrics"].start_frame()

    # 1. Perception
    perc_out      = mods["perception"].run(frame, frame_idx=frame_idx)
    detections    = perc_out["detections"]
    lane_mask     = perc_out["lane_mask"]
    drivable_mask = perc_out["drivable_mask"]
    features      = perc_out["features"]

    # 2. Lane geometry
    lane_out      = mods["lane_detector"].detect_lanes(frame)
    lane_geometry = lane_out["lane_geometry"]

    # 3. Tracking
    valid_dets    = [d for d in detections if d["score"] > 0.2]
    tracked       = mods["tracker"].update(valid_dets, frame)
    mods["metrics"].log_tracked_objects(len(tracked))

    # 4. Depth
    if depth_map is None:
        depth_map = mods["depth_estimator"].estimate(frame)

    # 5. Scene representation
    scene = mods["scene_builder"].build(tracked, depth_map, mods["depth_estimator"], lane_geometry)

    # 6. Motion prediction
    predictions = mods["motion_predictor"].predict(scene.get("objects", []))
    for obj in scene.get("objects", []):
        tid = obj.get("track_id")
        for p in predictions:
            if p["track_id"] == tid:
                obj["velocity"] = p["velocity"]
                obj["predicted_position"] = p["predicted_position"]
                break

    # 7. Trajectory planning
    traj = mods["trajectory"].plan(scene)
    scene["trajectory"] = traj["trajectory"]["points"]

    # 8. Decision
    decision = mods["decision_engine"].decide(scene)
    mods["metrics"].log_decision(decision)

    # 9. Control
    commands = mods["controller"].control(decision)

    # 10. XAI
    input_tensor = features[0] if features else None
    heatmap = mods["xai"].explain_detection(
        frame, input_tensor=input_tensor, detections=valid_dets
    )

    # 11. Uncertainty
    avg_conf = np.mean([d["score"] for d in valid_dets]) if valid_dets else 1.0
    uncertainty = 1.0 - avg_conf

    # 12. NL reasoning
    reasoning = mods["reasoner"].generate_justification(scene, decision)

    # 13. Visualize
    vis_frame = mods["visualizer"].overlay(
        frame, scene, valid_dets, lane_mask, drivable_mask, heatmap, decision, commands
    )

    latency = mods["metrics"].log_latency()
    mods["logger"].log_frame(frame_idx, scene.get("objects", []), decision, latency)

    return {
        "vis_frame":   vis_frame,
        "decision":    decision,
        "commands":    commands,
        "scene":       scene,
        "reasoning":   reasoning,
        "uncertainty": uncertainty,
        "latency":     latency,
        "frame_idx":   frame_idx,
    }


# ──────────────────────────────────────────────────── Episode runner ──────────
def run_carla_episode(
    carla_mod,
    world,
    mods: dict,
    episode_id: int,
    config: dict,
):
    """Run one closed-loop CARLA episode."""
    from control.carla_sensor_bridge import CARLASensorBridge

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # ── Spawn ego ─────────────────────────────────────────────────────
    ego_bp = bp_lib.find(config["ego_vehicle"])
    ego_bp.set_attribute("role_name", "hero")
    spawn_tf = spawn_points[config.get("spawn_idx", 0)]
    ego = world.spawn_actor(ego_bp, spawn_tf)
    logger.info("Ego spawned: %s", spawn_tf.location)

    # ── Sensor bridge ─────────────────────────────────────────────────
    bridge = CARLASensorBridge(world, ego, mods["perception"])
    bridge.attach()

    # ── NPC traffic ───────────────────────────────────────────────────
    npc_actors = bridge.spawn_npc_traffic(
        num_vehicles=config.get("num_npc", 10),
        num_pedestrians=config.get("num_pedestrians", 5),
    )

    # ── Collision sensor ──────────────────────────────────────────────
    collision_flag = [False]
    coll_bp = bp_lib.find("sensor.other.collision")
    coll = world.spawn_actor(coll_bp, carla_mod.Transform(), attach_to=ego)
    coll.listen(lambda _: collision_flag.__setitem__(0, True))

    # ── Output paths ──────────────────────────────────────────────────
    record_dir = Path(config.get("record_path", "artifacts/carla_records"))
    record_dir.mkdir(parents=True, exist_ok=True)
    csv_path = record_dir / f"episode_{episode_id:04d}.csv"
    xai_path = record_dir / f"episode_{episode_id:04d}_xai.json"

    video_writer = None
    if config.get("record_video"):
        video_path = str(record_dir / f"episode_{episode_id:04d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, 20, (1280, 720))

    # ── Episode loop ──────────────────────────────────────────────────
    explanations = []
    total_dist   = 0.0
    prev_loc     = None
    step         = 0
    max_steps    = config.get("max_steps", 500)

    # Let physics settle
    for _ in range(20):
        world.tick()
    time.sleep(0.2)

    logger.info("Episode %d starting (max %d steps)...", episode_id, max_steps)

    import cv2 as cv2_local  # ensure cv2 available in scope

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "step", "x", "y", "psi_deg", "speed_mps",
            "steering", "throttle", "brake",
            "action", "risk_score", "latency_ms",
            "num_detections", "mode", "collision",
        ])

        try:
            while step < max_steps and not collision_flag[0]:
                world.tick()

                transform = ego.get_transform()
                velocity  = ego.get_velocity()
                loc       = transform.location

                if prev_loc is not None:
                    total_dist += math.hypot(loc.x - prev_loc.x, loc.y - prev_loc.y)
                prev_loc = loc

                # Get latest perception result
                result = bridge.get_result(timeout=0.08)

                if result is not None:
                    idx, perc_out, frame = result

                    # Get metric depth from bridge
                    depth_map = bridge.get_latest_depth()

                    # Run decision + control
                    step_result = process_frame(
                        frame, mods, frame_idx=step,
                        depth_map=depth_map,
                    )

                    cmds      = step_result["commands"]
                    decision  = step_result["decision"]
                    vis_frame = step_result["vis_frame"]

                    # Apply to CARLA
                    carla_ctrl = carla_mod.VehicleControl(
                        throttle=float(np.clip(cmds.get("throttle", 0.3), 0.0, 1.0)),
                        steer=float(np.clip(cmds.get("steering", 0.0), -1.0, 1.0)),
                        brake=float(np.clip(cmds.get("brake", 0.0), 0.0, 1.0)),
                        hand_brake=False,
                        reverse=False,
                    )
                    ego.apply_control(carla_ctrl)

                    # Log
                    speed = math.hypot(velocity.x, velocity.y, velocity.z)
                    psi   = transform.rotation.yaw
                    writer.writerow([
                        step,
                        f"{loc.x:.4f}", f"{loc.y:.4f}",
                        f"{psi:.2f}", f"{speed:.4f}",
                        f"{cmds.get('steering', 0):.4f}",
                        f"{cmds.get('throttle', 0):.4f}",
                        f"{cmds.get('brake', 0):.4f}",
                        decision.get("action", "Proceed"),
                        f"{decision.get('risk_score', 0.0):.3f}",
                        f"{step_result['latency'] * 1000:.2f}",
                        len(perc_out.get("detections", [])),
                        cmds.get("mode", "rule"),
                        collision_flag[0],
                    ])

                    explanations.append({
                        "step":       step,
                        "decision":   decision,
                        "commands":   cmds,
                        "reasoning":  step_result["reasoning"],
                        "uncertainty": step_result["uncertainty"],
                        "latency_ms": step_result["latency"] * 1000,
                    })

                    if video_writer is not None:
                        video_writer.write(vis_frame)

                    if step % 50 == 0:
                        logger.info(
                            "Step %d | Action: %-8s | Speed: %.1f m/s | Risk: %.2f",
                            step, decision.get("action"), speed, decision.get("risk_score", 0),
                        )
                else:
                    # No perception result yet — safe default
                    ego.apply_control(carla_mod.VehicleControl(throttle=0.3, steer=0.0, brake=0.0))

                step += 1

        except KeyboardInterrupt:
            logger.info("Episode interrupted.")

    # Save XAI log
    with open(xai_path, "w") as f:
        json.dump(explanations, f, indent=2)

    # Cleanup
    if video_writer:
        video_writer.release()

    coll.stop()
    coll.destroy()
    bridge.destroy()

    for actor in npc_actors:
        try:
            actor.destroy()
        except Exception:
            pass
    if ego.is_alive:
        ego.destroy()

    result_summary = {
        "episode_id":  episode_id,
        "steps":       step,
        "distance_m":  round(total_dist, 2),
        "collision":   collision_flag[0],
        "csv":         str(csv_path),
        "xai_log":     str(xai_path),
    }
    logger.info(
        "Episode %d done | Steps=%d | Dist=%.1fm | Collision=%s",
        episode_id, step, total_dist, collision_flag[0],
    )
    return result_summary


# ─────────────────────────────────────────────────────────── CLI main ─────────
def main():
    parser = argparse.ArgumentParser(description="NeuroDrive-XAI CARLA Runner")
    parser.add_argument("--config",       default="control/config.yaml")
    parser.add_argument("--map",          default="Town03")
    parser.add_argument("--weather",      default="ClearNoon")
    parser.add_argument("--episodes",     type=int, default=3)
    parser.add_argument("--steps",        type=int, default=500)
    parser.add_argument("--npc",          type=int, default=10)
    parser.add_argument("--pedestrians",  type=int, default=5)
    parser.add_argument("--spawn-idx",    type=int, default=0)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--host",         default="localhost")
    parser.add_argument("--port",         type=int, default=2000)
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  NeuroDrive-XAI — CARLA Simulation")
    print("=" * 65)
    print(f"  Map        : {args.map}")
    print(f"  Weather    : {args.weather}")
    print(f"  Episodes   : {args.episodes}")
    print(f"  Steps/ep   : {args.steps}")
    print(f"  NPC cars   : {args.npc}")
    print(f"  Pedestrians: {args.pedestrians}")
    print("=" * 65 + "\n")

    # Import CARLA
    try:
        carla_mod = _import_carla()
    except ImportError as e:
        print(str(e))
        print("\nTip: Run  python carla_replay.py  to replay without CARLA.")
        sys.exit(1)

    # Connect
    client = carla_mod.Client(args.host, args.port)
    client.set_timeout(20.0)
    logger.info("Connecting to CARLA %s:%d ...", args.host, args.port)
    world = client.load_world(args.map)

    # Weather
    preset = getattr(carla_mod.WeatherParameters, args.weather, None)
    if preset:
        world.set_weather(preset)
    logger.info("World loaded: %s | Weather: %s", args.map, args.weather)

    # Init NeuroDrive pipeline
    mods = init_pipeline(args.config)

    config = {
        "ego_vehicle":     "vehicle.lincoln.mkz_2020",
        "spawn_idx":       args.spawn_idx,
        "max_steps":       args.steps,
        "num_npc":         args.npc,
        "num_pedestrians": args.pedestrians,
        "record_path":     "artifacts/carla_records",
        "record_video":    args.record_video,
    }

    # Run episodes
    results = []
    for ep in range(args.episodes):
        print(f"\n{'─'*50}")
        print(f"  Episode {ep+1}/{args.episodes}")
        print(f"{'─'*50}")
        r = run_carla_episode(carla_mod, world, mods, episode_id=ep, config=config)
        results.append(r)

    # Summary
    print("\n" + "=" * 65)
    print("  CARLA Simulation Complete — Summary")
    print("=" * 65)
    total_dist = sum(r["distance_m"] for r in results)
    collisions = sum(1 for r in results if r["collision"])
    print(f"  Episodes run      : {len(results)}")
    print(f"  Total distance    : {total_dist:.1f} m")
    print(f"  Collisions        : {collisions}/{len(results)}")
    print(f"  Records saved in  : artifacts/carla_records/")
    for r in results:
        status = "💥 COLLISION" if r["collision"] else "✓ Clean"
        print(f"    Ep {r['episode_id']:2d}: {r['steps']} steps | {r['distance_m']:.1f}m | {status}")
    print("=" * 65)

    # Print final pipeline stats
    mods["metrics"].print_summary()

    # Save overall summary
    summary_path = "artifacts/carla_records/run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Summary: {summary_path}")

    # Save XAI explanations to standard location for dashboard
    import shutil
    last_xai = results[-1].get("xai_log")
    if last_xai and os.path.exists(last_xai):
        os.makedirs("artifacts", exist_ok=True)
        shutil.copy(last_xai, "artifacts/explanations.json")
        print(f"  XAI log copied → artifacts/explanations.json (for dashboard)")


if __name__ == "__main__":
    import cv2
    main()

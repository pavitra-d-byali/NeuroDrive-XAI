"""
control/carla_sensor_bridge.py
================================
Attaches sensors to a CARLA ego vehicle and streams frames
to the NeuroDrive-XAI perception pipeline in real time.

Sensors:
  - RGB camera  → PerceptionModule (HybridNets object detection + lane seg)
  - Depth camera → DepthEstimator (metric depth map)
  - Semantic seg → Ground truth segmentation (for evaluation)
  - Collision   → Episode termination signal

Architecture:
  CARLA tick → camera callback → frame_queue → PerceptionModule → result_queue

Usage (internal — called by carla_run.py):
    bridge = CARLASensorBridge(world, ego, perception_module)
    bridge.attach()
    frame, perc_out = bridge.get_latest()
    bridge.destroy()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional, Tuple, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────── Camera configuration ────────────
RGB_CAMERA_CONFIG = {
    "image_size_x": "1280",
    "image_size_y":  "720",
    "fov":          "90",
    "sensor_tick":   "0.05",   # 20 Hz
}

DEPTH_CAMERA_CONFIG = {
    "image_size_x": "1280",
    "image_size_y":  "720",
    "fov":          "90",
    "sensor_tick":   "0.05",
}

# Camera mounting position (roof, forward-facing)
CAMERA_TRANSFORM_ARGS = {
    "location":  {"x": 1.5,  "y": 0.0, "z": 2.4},
    "rotation":  {"pitch": -5.0, "yaw": 0.0, "roll": 0.0},
}


# ─────────────────────────────────────────────────────────── Helpers ─────────
def _carla_image_to_bgr(carla_image) -> np.ndarray:
    """Convert CARLA RawImage (BGRA) to OpenCV BGR uint8 array."""
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = array.reshape((carla_image.height, carla_image.width, 4))
    return array[:, :, :3].copy()  # Drop alpha, return BGR


def _carla_depth_to_meters(carla_image) -> np.ndarray:
    """
    Convert CARLA depth image (BGRA encoded 24-bit) to float32 metre map.
    CARLA encoding: depth_m = (R + G*256 + B*65536) / (256^3 - 1) * 1000
    """
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = array.reshape((carla_image.height, carla_image.width, 4)).astype(np.float32)
    depth = (array[:, :, 2] + array[:, :, 1] * 256 + array[:, :, 0] * 65536)
    depth = depth / (256.0 ** 3 - 1) * 1000.0  # metres
    return depth.astype(np.float32)


# ──────────────────────────────────────────────────── Main bridge class ───────
class CARLASensorBridge:
    """
    Manages all sensors for the ego vehicle and bridges data to
    the NeuroDrive-XAI perception pipeline.
    """

    def __init__(
        self,
        world,
        ego_actor,
        perception_module=None,
        queue_size: int = 2,
    ):
        self._world = world
        self._ego   = ego_actor
        self._perception = perception_module

        # Internal queues (backpressure: drop old frames)
        self._frame_queue:  queue.Queue = queue.Queue(maxsize=queue_size)
        self._depth_queue:  queue.Queue = queue.Queue(maxsize=queue_size)
        self._result_queue: queue.Queue = queue.Queue(maxsize=queue_size)

        self._sensors = []
        self._latest_frame:     Optional[np.ndarray] = None
        self._latest_depth:     Optional[np.ndarray] = None
        self._latest_perc_out:  Optional[dict]       = None
        self._frame_count: int = 0
        self._lock = threading.Lock()

        # Perception thread
        self._stop_event = threading.Event()
        self._perc_thread: Optional[threading.Thread] = None

    # ── Sensor attachment ─────────────────────────────────────────────────
    def attach(self) -> None:
        """Spawn and attach all sensors to the ego vehicle."""
        carla = _import_carla()
        bp_lib = self._world.get_blueprint_library()

        # ── RGB Camera ─────────────────────────────────────────────
        rgb_bp = bp_lib.find("sensor.camera.rgb")
        for attr, val in RGB_CAMERA_CONFIG.items():
            rgb_bp.set_attribute(attr, val)

        cam_tf = carla.Transform(
            carla.Location(**CAMERA_TRANSFORM_ARGS["location"]),
            carla.Rotation(**CAMERA_TRANSFORM_ARGS["rotation"]),
        )
        rgb_cam = self._world.spawn_actor(rgb_bp, cam_tf, attach_to=self._ego)
        rgb_cam.listen(self._on_rgb_frame)
        self._sensors.append(rgb_cam)
        logger.info("RGB camera attached.")

        # ── Depth Camera ─────────────────────────────────────────
        dep_bp = bp_lib.find("sensor.camera.depth")
        for attr, val in DEPTH_CAMERA_CONFIG.items():
            dep_bp.set_attribute(attr, val)
        dep_cam = self._world.spawn_actor(dep_bp, cam_tf, attach_to=self._ego)
        dep_cam.listen(self._on_depth_frame)
        self._sensors.append(dep_cam)
        logger.info("Depth camera attached.")

        # Start perception worker thread
        self._perc_thread = threading.Thread(
            target=self._perception_worker, daemon=True
        )
        self._perc_thread.start()
        logger.info("CARLASensorBridge active.")

    # ── Sensor callbacks ─────────────────────────────────────────────────
    def _on_rgb_frame(self, carla_image) -> None:
        """Called by CARLA on each RGB tick. Convert and queue."""
        frame = _carla_image_to_bgr(carla_image)
        with self._lock:
            self._latest_frame = frame
        # Non-blocking put (drop if queue full = frame skip)
        try:
            self._frame_queue.put_nowait((self._frame_count, frame))
            self._frame_count += 1
        except queue.Full:
            pass

    def _on_depth_frame(self, carla_image) -> None:
        """Called by CARLA on each depth tick."""
        depth = _carla_depth_to_meters(carla_image)
        with self._lock:
            self._latest_depth = depth
        try:
            self._depth_queue.put_nowait(depth)
        except queue.Full:
            pass

    # ── Perception worker ─────────────────────────────────────────────────
    def _perception_worker(self) -> None:
        """
        Background thread: dequeues frames, runs perception, stores result.
        Keeps the CARLA tick loop non-blocking.
        """
        logger.info("[PercWorker] Started.")
        while not self._stop_event.is_set():
            try:
                idx, frame = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if self._perception is not None:
                try:
                    perc_out = self._perception.run(
                        frame,
                        conf_thresh=0.25,
                        resolution=640,
                        frame_idx=idx,
                        debug=False,
                    )
                except Exception as e:
                    logger.warning("[PercWorker] Perception error: %s", e)
                    perc_out = {
                        "detections": [], "lane_mask": np.zeros(frame.shape[:2], np.uint8),
                        "drivable_mask": np.zeros(frame.shape[:2], np.uint8), "features": [],
                    }
            else:
                perc_out = {
                    "detections": [], "lane_mask": np.zeros(frame.shape[:2], np.uint8),
                    "drivable_mask": np.zeros(frame.shape[:2], np.uint8), "features": [],
                }

            with self._lock:
                self._latest_perc_out = perc_out

            try:
                self._result_queue.put_nowait((idx, perc_out, frame))
            except queue.Full:
                pass

        logger.info("[PercWorker] Stopped.")

    # ── Data access ───────────────────────────────────────────────────────
    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Non-blocking: returns the most recent (frame, perc_out) pair.
        Returns (None, None) if no data yet.
        """
        with self._lock:
            return self._latest_frame, self._latest_perc_out

    def get_latest_depth(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_depth

    def get_result(self, timeout: float = 0.05) -> Optional[Tuple]:
        """Blocking get with timeout from result queue."""
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── NPC spawning ──────────────────────────────────────────────────────
    def spawn_npc_traffic(self, num_vehicles: int = 10, num_pedestrians: int = 5) -> list:
        """
        Spawn NPC vehicles and pedestrians for realistic multi-agent scenarios.
        Returns list of spawned actor IDs for cleanup.
        """
        carla = _import_carla()
        bp_lib = self._world.get_blueprint_library()
        spawn_points = self._world.get_map().get_spawn_points()
        npc_actors = []

        # Spawn vehicles
        vehicle_blueprints = bp_lib.filter("vehicle.*")
        vehicle_blueprints = [
            bp for bp in vehicle_blueprints
            if int(bp.get_attribute("number_of_wheels")) == 4
        ]

        for i in range(min(num_vehicles, len(spawn_points) - 1)):
            bp = np.random.choice(vehicle_blueprints)
            # Randomise colour
            if bp.has_attribute("color"):
                color = np.random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)

            sp = spawn_points[i + 1]  # Skip idx 0 (used by ego)
            try:
                actor = self._world.try_spawn_actor(bp, sp)
                if actor:
                    actor.set_autopilot(True)
                    npc_actors.append(actor)
            except Exception:
                pass

        # Spawn pedestrians
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        walker_ctrl_bp = bp_lib.find("controller.ai.walker")

        for _ in range(num_pedestrians):
            try:
                walker_bp = np.random.choice(walker_bps)
                spawn_loc = self._world.get_random_location_from_navigation()
                if spawn_loc:
                    walker_tf = carla.Transform(spawn_loc)
                    walker = self._world.try_spawn_actor(walker_bp, walker_tf)
                    if walker:
                        ctrl = self._world.spawn_actor(walker_ctrl_bp, carla.Transform(), attach_to=walker)
                        ctrl.start()
                        ctrl.go_to_location(self._world.get_random_location_from_navigation())
                        npc_actors.extend([walker, ctrl])
            except Exception:
                pass

        logger.info(
            "Spawned %d NPC vehicles + %d pedestrian controllers",
            sum(1 for a in npc_actors if "vehicle" in a.type_id),
            sum(1 for a in npc_actors if "walker" in a.type_id),
        )
        return npc_actors

    # ── Cleanup ───────────────────────────────────────────────────────────
    def destroy(self) -> None:
        self._stop_event.set()
        if self._perc_thread:
            self._perc_thread.join(timeout=3.0)
        for sensor in self._sensors:
            try:
                if sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            except Exception:
                pass
        self._sensors.clear()
        logger.info("CARLASensorBridge destroyed.")


# ─────────────────────────────────────────────────────────── Import helper ────
def _import_carla():
    """Import CARLA Python API, providing a clear error if not found."""
    try:
        import carla
        return carla
    except ImportError:
        import os, sys
        from pathlib import Path
        eggs = [
            "/opt/carla-simulator/PythonAPI/carla/dist",
            "C:/CARLA/PythonAPI/carla/dist",
            os.environ.get("CARLA_EGG", ""),
        ]
        for base in eggs:
            if not base:
                continue
            for egg in Path(base).glob("carla-*.egg"):
                sys.path.insert(0, str(egg))
                try:
                    import carla
                    return carla
                except ImportError:
                    continue
        raise ImportError(
            "CARLA Python API not found.\n"
            "Install via: pip install carla\n"
            "Or set CARLA_EGG=/path/to/carla-x.x.x-py3.x-xxx.egg"
        )

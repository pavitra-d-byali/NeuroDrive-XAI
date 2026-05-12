"""tests/test_carla_interface.py — CARLA interface tests (no CARLA server needed)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import math
import numpy as np


# ── Mock CARLA module ─────────────────────────────────────────────────────────
def make_carla_mock():
    carla = MagicMock()

    # Transform / Location / Rotation
    loc = MagicMock()
    loc.x, loc.y, loc.z = 10.0, -5.0, 0.5
    rot = MagicMock()
    rot.yaw = 45.0
    tf = MagicMock()
    tf.location = loc
    tf.rotation = rot

    carla.Transform.return_value = tf
    carla.Location = MagicMock(return_value=loc)
    carla.Rotation = MagicMock(return_value=rot)

    # Velocity
    vel = MagicMock()
    vel.x, vel.y, vel.z = 5.0, 0.0, 0.0
    carla.Vector3D = MagicMock(return_value=vel)

    # WeatherParameters
    carla.WeatherParameters.ClearNoon = MagicMock()

    # VehicleControl
    carla.VehicleControl = MagicMock(return_value=MagicMock())

    return carla, tf, vel


def test_carla_transform_to_state():
    """Test coordinate system conversion from CARLA → VehicleState."""
    from control.carla_interface import _carla_transform_to_state

    carla, tf, vel = make_carla_mock()
    tf.location.x = 10.0
    tf.location.y = -20.0  # CARLA left-hand Y
    tf.rotation.yaw = 90.0
    vel.x, vel.y = 5.0, 0.0

    state = _carla_transform_to_state(tf, vel)

    assert state.x == pytest.approx(10.0)
    assert state.y == pytest.approx(20.0)  # negated
    assert abs(state.psi - (-math.radians(90.0))) < 0.001
    assert state.v == pytest.approx(5.0)


def test_waypoints_to_arrays():
    """Test CARLA waypoint list → np.ndarray conversion."""
    from control.carla_interface import _waypoints_to_arrays

    wps = []
    for i in range(5):
        wp = MagicMock()
        wp.transform.location.x = float(i * 2)
        wp.transform.location.y = float(-i)  # CARLA Y (will be negated)
        wps.append(wp)

    xs, ys = _waypoints_to_arrays(wps)
    assert len(xs) == 5
    assert len(ys) == 5
    assert xs[0] == pytest.approx(0.0)
    assert ys[0] == pytest.approx(0.0)  # -(-0) = 0
    assert ys[1] == pytest.approx(1.0)  # -(-1)


def test_carla_replay_demo_generation():
    """Test demo episode generation (no CARLA)."""
    from carla_replay import generate_demo_episode

    rows = generate_demo_episode(100)
    assert len(rows) == 100
    assert all("step" in r for r in rows)
    assert all("x" in r for r in rows)
    assert all("action" in r for r in rows)
    assert all(r["action"] in ("Proceed", "Slow", "Brake") for r in rows)


def test_carla_replay_stats():
    """Test episode statistics on demo data."""
    from carla_replay import generate_demo_episode

    rows = generate_demo_episode(200)
    # Emergency braking at step 145-175
    brake_steps = [r for r in rows if r["brake"] > 0.3]
    assert len(brake_steps) > 0, "Expected brake events in demo episode"

    # Speeds should be non-negative
    assert all(r["speed_mps"] >= 0 for r in rows)


def test_episode_csv_roundtrip(tmp_path):
    """Test CSV write → read roundtrip."""
    import csv
    from carla_replay import load_episode_csv

    csv_path = tmp_path / "episode_test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "x", "y", "psi_deg", "speed_mps",
                         "steering", "throttle", "brake", "action",
                         "risk_score", "latency_ms", "num_detections", "collision"])
        writer.writerow([0, 10.0, 5.0, 45.0, 8.5, 0.1, 0.5, 0.0,
                         "Proceed", 0.05, 32.5, 3, False])
        writer.writerow([1, 10.5, 5.2, 46.0, 8.3, 0.15, 0.4, 0.3,
                         "Slow", 0.45, 28.1, 4, False])

    rows = load_episode_csv(str(csv_path))
    assert len(rows) == 2
    assert rows[0]["step"] == 0
    assert rows[0]["x"] == pytest.approx(10.0)
    assert rows[1]["action"] == "Slow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

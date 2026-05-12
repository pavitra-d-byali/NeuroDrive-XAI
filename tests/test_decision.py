"""tests/test_decision.py — Decision engine unit tests"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from planning.decision_engine import DecisionEngine


def make_scene(dist=30.0, lane_center=True, det_conf=0.9, lane_conf=0.9):
    return {
        "objects": [{"track_id": 1, "distance_meters": dist, "type": "car",
                     "velocity": 15.0, "lane": "center"}],
        "lane_geometry": {"center_line": [(0, 0), (100, 0)]} if lane_center else {},
        "confidence": {"lane": lane_conf, "detection": det_conf, "depth": 0.9},
    }


def test_proceed_on_clear_road():
    engine = DecisionEngine(history_size=5)
    scene = make_scene(dist=80.0)
    result = engine.decide(scene)
    assert result["action"] in ("Proceed", "Slow"), f"Unexpected: {result['action']}"


def test_brake_on_close_object():
    engine = DecisionEngine(history_size=5)
    scene = make_scene(dist=5.0)
    result = engine.decide(scene)
    assert result["action"] in ("Brake", "Slow"), f"Expected Brake/Slow, got {result['action']}"


def test_slow_on_missing_lane():
    engine = DecisionEngine(history_size=5)
    scene = make_scene(lane_center=False)
    result = engine.decide(scene)
    assert result["action"] == "Slow", f"Expected Slow on missing lane, got {result['action']}"


def test_slow_on_low_confidence():
    engine = DecisionEngine(history_size=5)
    scene = make_scene(det_conf=0.4, lane_conf=0.4)
    result = engine.decide(scene)
    assert result["action"] == "Slow", f"Expected Slow on low conf, got {result['action']}"


def test_risk_score_range():
    engine = DecisionEngine(history_size=5)
    for dist in [5, 15, 40, 80]:
        result = engine.decide(make_scene(dist=dist))
        risk = result.get("risk_score", 0)
        assert 0.0 <= risk <= 1.0, f"Risk {risk} out of [0,1] for dist={dist}"


def test_history_accumulation():
    engine = DecisionEngine(history_size=5)
    for _ in range(5):
        engine.decide(make_scene(dist=5.0))  # All brakes
    result = engine.decide(make_scene(dist=80.0))
    # Smoothing may delay switching back to Proceed
    assert "history" in result
    assert len(result["history"]) <= 5


def test_latency_failsafe():
    engine = DecisionEngine(history_size=5)
    result = engine.decide(make_scene(dist=30.0), latency_ms=150.0)
    assert result["action"] == "Slow", "Latency failsafe should trigger Slow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

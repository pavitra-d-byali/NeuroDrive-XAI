"""tests/test_perception.py — Perception module unit tests"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import cv2


def make_frame(h=720, w=1280):
    """Create a realistic-looking synthetic BGR driving frame."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:h//2] = [100, 150, 200]   # sky
    img[h//2:] = [80, 80, 80]       # road
    # Lane markings
    cv2.line(img, (w//2, h), (w//2 - 100, h//2), (255, 255, 255), 4)
    cv2.line(img, (w//2, h), (w//2 + 100, h//2), (255, 255, 255), 4)
    # Fake car
    cv2.rectangle(img, (500, 400), (780, 600), (50, 100, 200), -1)
    return img


def test_perception_returns_correct_keys():
    from perception.hybridnets_wrapper import PerceptionModule
    pm = PerceptionModule(use_cuda=False)
    frame = make_frame()
    result = pm.run(frame, resolution=416, frame_idx=0)
    assert "detections" in result
    assert "lane_mask" in result
    assert "drivable_mask" in result
    assert "features" in result


def test_perception_output_shapes():
    from perception.hybridnets_wrapper import PerceptionModule
    pm = PerceptionModule(use_cuda=False)
    frame = make_frame(h=720, w=1280)
    result = pm.run(frame, resolution=416)
    assert result["lane_mask"].shape == (720, 1280)
    assert result["drivable_mask"].shape == (720, 1280)
    assert result["lane_mask"].dtype == np.uint8
    assert result["drivable_mask"].dtype == np.uint8


def test_perception_detections_schema():
    from perception.hybridnets_wrapper import PerceptionModule
    pm = PerceptionModule(use_cuda=False)
    frame = make_frame()
    result = pm.run(frame, resolution=416)
    for det in result["detections"]:
        assert "bbox" in det
        assert "score" in det
        assert "class" in det
        assert len(det["bbox"]) == 4
        assert 0.0 <= det["score"] <= 1.0


def test_perception_bbox_in_bounds():
    from perception.hybridnets_wrapper import PerceptionModule
    pm = PerceptionModule(use_cuda=False)
    frame = make_frame(h=720, w=1280)
    result = pm.run(frame, resolution=416)
    for det in result["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        assert 0 <= x1 < x2 <= 1280, f"X bounds invalid: {x1},{x2}"
        assert 0 <= y1 < y2 <= 720, f"Y bounds invalid: {y1},{y2}"


def test_perception_frame_idx_debug_params():
    """Ensure run() accepts frame_idx and debug without error."""
    from perception.hybridnets_wrapper import PerceptionModule
    pm = PerceptionModule(use_cuda=False)
    frame = make_frame()
    # Should not raise
    result = pm.run(frame, frame_idx=42, debug=True, resolution=416)
    assert isinstance(result, dict)


def test_perception_small_frame():
    """Small frames should not crash."""
    from perception.hybridnets_wrapper import PerceptionModule
    pm = PerceptionModule(use_cuda=False)
    frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    result = pm.run(frame, resolution=320)
    assert "detections" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

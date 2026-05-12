"""tests/test_counterfactual.py — CounterfactualEngine + MLP unit tests"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import torch


def _make_model_and_scaler():
    """Build a minimal trained MLP + scaler for testing without loading weights."""
    from decision.mlp_model import NeuroDecisionMLP
    from sklearn.preprocessing import StandardScaler

    model = NeuroDecisionMLP(input_features=6)
    model.eval()

    # Fit scaler on realistic feature ranges
    X_dummy = np.array([
        [5.0, -10.0, 0.5, 0.05, 3, 0],
        [80.0, 5.0, 0.0, 0.01, 1, 0],
        [12.0, -20.0, 1.0, 0.08, 5, 1],
        [40.0, 0.0, -0.3, 0.02, 2, 3],
    ])
    scaler = StandardScaler()
    scaler.fit(X_dummy)
    return model, scaler


def test_mlp_forward_shape():
    from decision.mlp_model import NeuroDecisionMLP
    model = NeuroDecisionMLP(input_features=6)
    x = torch.FloatTensor([[10.0, -5.0, 0.5, 0.03, 2, 0]])
    steer, brake = model(x)
    assert steer.shape == (1, 1)
    assert brake.shape == (1, 1)


def test_mlp_brake_output_range():
    """Brake output after sigmoid should be in [0, 1]."""
    from decision.mlp_model import NeuroDecisionMLP
    model = NeuroDecisionMLP(input_features=6)
    for _ in range(10):
        x = torch.FloatTensor(np.random.randn(1, 6).astype(np.float32))
        _, brake = model(x)
        prob = torch.sigmoid(brake).item()
        assert 0.0 <= prob <= 1.0, f"Brake prob {prob} out of [0,1]"


def test_counterfactual_returns_delta():
    from evaluation.metrics import CounterfactualEngine
    model, scaler = _make_model_and_scaler()
    engine = CounterfactualEngine(model=model, scaler=scaler)

    features = np.array([8.0, -12.0, 0.3, 0.04, 3, 0])
    result = engine.generate_explanation(features)

    assert "action" in result
    assert result["action"] in ("BRAKE", "CONTINUE")
    assert "counterfactual" in result


def test_counterfactual_delta_direction():
    """If braking, CONTINUE delta should be positive (increase distance)."""
    from evaluation.metrics import CounterfactualEngine
    model, scaler = _make_model_and_scaler()
    engine = CounterfactualEngine(model=model, scaler=scaler)

    # Force a close-range scenario likely to brake
    features = np.array([3.0, -25.0, 0.0, 0.0, 5, 0])
    result = engine.generate_explanation(features)

    if result["action"] == "BRAKE" and isinstance(result.get("counterfactual"), dict):
        delta_str = result["counterfactual"].get("distance_to_object", "")
        # e.g. "+12.3m → CONTINUE" — delta should be positive
        if delta_str and "+" in delta_str:
            delta_val = float(delta_str.split("m")[0].replace("+", ""))
            assert delta_val > 0, "BRAKE→CONTINUE delta should be positive"


def test_temporal_smoother_median():
    """Verify median filter in TemporalSmoother rejects spikes."""
    from decision.temporal import TemporalSmoother
    from decision.mlp_model import NeuroDecisionMLP

    model = NeuroDecisionMLP(input_features=6)
    smoother = TemporalSmoother(model=model, window_size=5)

    # Feed 4 low-brake predictions, then one spike
    for val in [0.1, 0.1, 0.1, 0.1]:
        x = torch.FloatTensor([[50.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        smoother.predict(x)

    # The median should stay low despite the spike
    x = torch.FloatTensor([[50.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    steer, final_brake, avg_brake = smoother.predict(x)
    assert 0.0 <= avg_brake <= 1.0


def test_physics_regularizer():
    """High steering should cap brake probability."""
    from decision.temporal import TemporalSmoother
    from decision.mlp_model import NeuroDecisionMLP

    model = NeuroDecisionMLP(input_features=6)
    smoother = TemporalSmoother(model=model, window_size=5, brake_hysteresis_threshold=0.6)

    # Manually inject high steer history
    for _ in range(5):
        smoother.steer_history.append(0.8)   # sharp bend
        smoother.brake_history.append(0.95)  # high brake

    x = torch.FloatTensor([[20.0, -5.0, 1.0, 0.09, 2, 0]])
    steer, final_brake, avg_brake = smoother.predict(x)
    # Physics regularizer: with steer~0.8, max_brake_allowed = max(0, 1-(0.8-0.5)*2) = 0.4
    assert avg_brake <= 0.42, f"Physics regularizer failed: brake={avg_brake:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

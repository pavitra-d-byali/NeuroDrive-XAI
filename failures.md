# ⚠️ System Architecture Failures

A production ML system is defined by understanding its limits.

### 1. The 'Swerving Brake' Paradox
*   **Input State**: Imminent heavy object collision on a sharp bend.
*   **Feature Vector**: `[dist: 4.2m, rel_v: -8m/s, curv: +0.09]`
*   **Model Output**: `[steer: +0.85, brake: 0.98]`
*   **Why It Fails**: The system acts on MSE/BCE independently. Applying 98% brake pressure during an 85% steering lock breaks traction physics. The models do not mutually share a physics regularizer.

### 2. Temporal Ghost Caching
*   **Input State**: Depth estimation fails for 1 frame due to severe glare, then recovers.
*   **Feature Vector**: `distance_to_object` spikes from `8.0m` -> `100.0m` -> `7.8m`.
*   **Why It Fails**: The smoothing window averages the massive spike, artificially diluting the risk. 
*   **Required Fix**: Instead of a simple mean, the `TemporalSmoother` requires an outlier-rejection filter (e.g. median filter) to discard single-frame sensor fault gaps.

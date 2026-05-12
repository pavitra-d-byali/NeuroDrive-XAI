"""
api/control_routes.py
=====================
FastAPI REST endpoints for the Vehicle Control module.

Endpoints:
  POST /control/compute      — real-time control command
  POST /control/explain      — SHAP explanation for a control decision
  GET  /control/stats        — controller runtime statistics
  POST /control/reset        — reset controller state (new episode)
  GET  /health               — liveness probe

Designed for:
  - Microservice deployment (Docker)
  - Integration with the main NeuroDrive pipeline via HTTP
  - CARLA side-car mode (called per-tick from carla_interface.py)

Run locally:
  uvicorn api.control_routes:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────────────── App init ─────────────
app = FastAPI(
    title="NeuroDrive-XAI Control API",
    description="Vehicle Control (PID + MPC) microservice for NeuroDrive-XAI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────── Lazy singletons ──────
_controller = None
_xai_module = None
CONFIG_PATH = os.environ.get("NEURODRIVE_CONTROL_CONFIG", "control/config.yaml")
RCN_PATH    = os.environ.get("NEURODRIVE_RCN_ONNX", "weights/control/residual_net.onnx")
NORM_PATH   = os.environ.get("NEURODRIVE_NORM_STATS", "datasets/nuscenes/norm_stats.pkl")


def _get_controller():
    global _controller
    if _controller is None:
        from control.inference import HybridControlInference
        _controller = HybridControlInference(
            config_path=CONFIG_PATH,
            rcn_path=RCN_PATH if os.path.exists(RCN_PATH) else None,
            norm_stats_path=NORM_PATH if os.path.exists(NORM_PATH) else None,
        )
        logger.info("Controller initialised (lazy)")
    return _controller


def _get_xai():
    global _xai_module
    if _xai_module is None:
        try:
            from control.xai_control import ControlXAI
            ctrl = _get_controller()

            def _ctrl_fn(feat_dict: Dict) -> Dict:
                from control.vehicle_model import VehicleState
                state = VehicleState(v=feat_dict.get("speed_mps", 0.0))
                ref_x = np.array([state.x, state.x + 10.0])
                ref_y = np.array([state.y, state.y])
                return ctrl.compute(state, ref_x, ref_y)

            _xai_module = ControlXAI(controller_fn=_ctrl_fn)
            logger.info("XAI module initialised (lazy)")
        except ImportError as e:
            logger.warning("XAI not available: %s", e)
            return None
    return _xai_module


# ─────────────────────────────────────────────────────────── Schemas ──────────
class VehicleStateSchema(BaseModel):
    x:   float = Field(0.0, description="Longitudinal position (m)")
    y:   float = Field(0.0, description="Lateral position (m)")
    psi: float = Field(0.0, description="Heading angle (rad)")
    v:   float = Field(0.0, description="Speed (m/s)")


class ControlRequest(BaseModel):
    state:         VehicleStateSchema
    ref_x:         List[float] = Field(..., description="Reference path X coords (m)")
    ref_y:         List[float] = Field(..., description="Reference path Y coords (m)")
    target_speed:  Optional[float] = Field(None, description="Desired speed (m/s)")
    closest_dist:  float = Field(50.0, description="Distance to nearest obstacle (m)")
    closest_v_rel: float = Field(0.0,  description="Relative speed of nearest obstacle (m/s)")
    num_agents:    int   = Field(0,    description="Number of nearby agents")


class ExplainRequest(BaseModel):
    features: List[float] = Field(
        ...,
        description="9-element normalised feature vector "
                    "[speed, accel, curvature, prev_steer, cte, heading_err, "
                    "closest_dist, closest_v_rel, num_agents]",
        min_items=9, max_items=9,
    )
    n_samples: int = Field(100, description="SHAP KernelExplainer samples")


class ControlResponse(BaseModel):
    throttle:    float
    brake:       float
    steering:    float
    mode:        str
    solve_ms:    float
    cte:         float
    heading_err: float
    speed_error: float
    rcn_applied: bool


# ─────────────────────────────────────────────────────────── Endpoints ────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "neurodrive-control"}


@app.post("/control/compute", response_model=ControlResponse)
def compute_control(req: ControlRequest):
    """
    Compute one control command.

    Input:  ego state + reference path + scene context
    Output: throttle, brake, steering + diagnostics
    """
    try:
        from control.vehicle_model import VehicleState
        ctrl = _get_controller()

        state = VehicleState(
            x=req.state.x, y=req.state.y,
            psi=req.state.psi, v=req.state.v,
        )
        ref_x = np.array(req.ref_x, dtype=np.float64)
        ref_y = np.array(req.ref_y, dtype=np.float64)

        if len(ref_x) < 2:
            raise HTTPException(status_code=422, detail="ref_x/ref_y must have >= 2 points")

        result = ctrl.compute(
            state=state,
            ref_x=ref_x,
            ref_y=ref_y,
            target_speed=req.target_speed,
            closest_dist=req.closest_dist,
            closest_v_rel=req.closest_v_rel,
            num_agents=req.num_agents,
        )
        return ControlResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Control compute error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/control/explain")
def explain_control(req: ExplainRequest):
    """
    Generate SHAP explanation for a control decision.

    Returns per-feature contributions for steering, throttle, and brake.
    Note: First call triggers explainer initialisation (~5 s).
    """
    xai = _get_xai()
    if xai is None:
        raise HTTPException(
            status_code=501,
            detail="XAI not available. Install: pip install shap",
        )
    try:
        features = np.array(req.features, dtype=np.float32)
        explanation = xai.explain_frame(features, n_samples=req.n_samples)
        return JSONResponse(content=explanation)
    except Exception as e:
        logger.error("XAI explain error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/control/stats")
def get_stats():
    """Return controller runtime statistics."""
    try:
        ctrl = _get_controller()
        return ctrl.stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/control/reset")
def reset_controller():
    """Reset controller state (call at the start of each new episode)."""
    try:
        ctrl = _get_controller()
        ctrl.reset()
        return {"status": "reset", "message": "Controller state cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────── Error handler ────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ─────────────────────────────────────────────────────────── Dev server ────────
if __name__ == "__main__":
    import uvicorn
    import yaml as _yaml
    with open(CONFIG_PATH) as _f:
        _api_cfg = _yaml.safe_load(_f).get("api", {})
    uvicorn.run(
        "api.control_routes:app",
        host=_api_cfg.get("host", "0.0.0.0"),
        port=_api_cfg.get("port", 8001),
        workers=_api_cfg.get("workers", 1),
        log_level=_api_cfg.get("log_level", "info"),
        reload=False,
    )

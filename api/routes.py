"""
api/routes.py
=============
NeuroDrive-XAI FastAPI inference service.

Real inference pipeline:
  POST /predict-frame  → queues background inference job
  GET  /job/{job_id}   → polls result

Background task runs:
  1. NeuroDecisionMLP forward pass (real weights from weights/neurodrive_mlp.pth)
  2. CounterfactualEngine binary search (real mathematical delta)
  3. Returns real steering angle, brake probability, action, counterfactual
"""

from __future__ import annotations

import os
import sys
import uuid
import time
import logging
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NeuroDrive-XAI Inference API",
    description="Real-time autonomous driving inference with mathematical XAI explanations.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────── Model loading (startup) ──────────
_mlp_model = None
_scaler = None
_cf_engine = None
_model_load_error: Optional[str] = None


def _load_models():
    global _mlp_model, _scaler, _cf_engine, _model_load_error
    try:
        import torch
        import joblib
        from decision.mlp_model import NeuroDecisionMLP
        from evaluation.metrics import CounterfactualEngine

        scaler_path = "weights/feature_scaler.pkl"
        model_path  = "weights/neurodrive_mlp.pth"

        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            _model_load_error = (
                f"Model files not found. Run: python decision/train.py\n"
                f"  Missing: {scaler_path if not os.path.exists(scaler_path) else model_path}"
            )
            logger.warning(_model_load_error)
            return

        _scaler = joblib.load(scaler_path)
        _mlp_model = NeuroDecisionMLP(input_features=6)
        _mlp_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        _mlp_model.eval()

        _cf_engine = CounterfactualEngine(model=_mlp_model, scaler=_scaler)
        logger.info("✓ NeuroDrive MLP + CounterfactualEngine loaded.")
    except Exception as e:
        _model_load_error = str(e)
        logger.error("Model load failed: %s", e)


# Load on startup
_load_models()

# ─────────────────────────────────────────────────── In-memory job store ──────
jobs: Dict[str, dict] = {}


# ─────────────────────────────────────────────────────────── Request schema ───
class FramePayload(BaseModel):
    distance_to_object:  float  # metres
    relative_velocity:   float  # m/s (negative = closing)
    lane_offset:         float  # metres from centre
    lane_curvature:      float  # 1/m
    num_objects:         int
    closest_object_type: int    # 0=car,1=pedestrian,2=bike,3=truck,4=none


# ──────────────────────────────────────────────────── Background inference ────
def _run_real_inference(job_id: str, payload: FramePayload):
    """Real MLP inference + counterfactual explanation."""
    jobs[job_id]["status"] = "processing"
    t0 = time.monotonic()

    type_map = {0: "car", 1: "pedestrian", 2: "bike", 3: "truck", 4: "none"}

    try:
        import torch

        features = np.array([
            payload.distance_to_object,
            payload.relative_velocity,
            payload.lane_offset,
            payload.lane_curvature,
            float(payload.num_objects),
            float(payload.closest_object_type),
        ], dtype=np.float64)

        if _mlp_model is None or _scaler is None:
            # Graceful degradation: rule-based fallback
            brake = 1 if payload.distance_to_object < 15.0 else 0
            steer = float(np.clip(-payload.lane_offset * 0.3, -1.0, 1.0))
            action = "BRAKE" if brake else ("SLOW" if payload.distance_to_object < 30 else "CONTINUE")
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = {
                "steering_angle":   round(steer, 4),
                "brake_probability": float(brake),
                "action":           action,
                "counterfactual":   "Model not loaded — rule-based fallback",
                "inference_ms":     round((time.monotonic() - t0) * 1000, 2),
                "mode":             "rule_based_fallback",
                "warning":          _model_load_error,
            }
            return

        # ── Real MLP inference ────────────────────────────────────────
        x_scaled = _scaler.transform([features])
        x_tensor = torch.FloatTensor(x_scaled)

        with torch.no_grad():
            steer_pred, brake_prob = _mlp_model(x_tensor)

        steer_val  = float(steer_pred.item())
        brake_val  = float(brake_prob.item())
        action     = "BRAKE" if brake_val > 0.5 else ("SLOW" if brake_val > 0.3 else "CONTINUE")

        # ── Real counterfactual explanation ──────────────────────────
        explanation = _cf_engine.generate_explanation(features)
        cf_text     = explanation.get("counterfactual", {})

        inference_ms = round((time.monotonic() - t0) * 1000, 2)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "steering_angle":   round(steer_val, 4),
            "brake_probability": round(brake_val, 4),
            "action":           action,
            "counterfactual":   cf_text,
            "confidence":       explanation.get("confidence", round(brake_val, 3)),
            "features": {
                "distance_to_object":  payload.distance_to_object,
                "relative_velocity":   payload.relative_velocity,
                "lane_offset":         payload.lane_offset,
                "closest_object_type": type_map.get(payload.closest_object_type, "unknown"),
            },
            "inference_ms": inference_ms,
            "mode":         "real_mlp",
        }

    except Exception as e:
        logger.error("Inference error: %s", e)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"]  = str(e)


# ─────────────────────────────────────────────────────────── Endpoints ────────
@app.post("/predict-frame", summary="Queue a frame inference job")
async def predict_frame(payload: FramePayload, background_tasks: BackgroundTasks):
    """
    Submit a feature vector for asynchronous MLP inference.
    Returns a job_id for polling via GET /job/{job_id}.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "result": None, "error": None}
    background_tasks.add_task(_run_real_inference, job_id, payload)
    return {"job_id": job_id, "message": "Inference queued"}


@app.get("/job/{job_id}", summary="Poll inference job result")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "model_loaded": _mlp_model is not None,
        "model_error":  _model_load_error,
    }


@app.post("/predict-sync", summary="Synchronous inference (no polling needed)")
async def predict_sync(payload: FramePayload):
    """
    Direct synchronous inference — blocks until result ready (~1ms).
    Suitable for low-latency clients.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "result": None, "error": None}
    _run_real_inference(job_id, payload)
    return jobs[job_id]["result"] or {"error": jobs[job_id].get("error")}

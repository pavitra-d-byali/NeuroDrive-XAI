import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import io

from perception.hybridnets_wrapper import PerceptionModule
from perception.depth_estimator import DepthEstimator
from perception.lane_detector import LaneDetector
from scene_representation.scene_builder import SceneBuilder
from planning.decision_engine import DecisionEngine
from planning.trajectory_planner import TrajectoryPlanner

app = FastAPI(title="NeuroDrive ML Inference API", version="2.0")

# Load heavy models lazily
class EngineContext:
    def __init__(self):
        self.perception = PerceptionModule(use_cuda=False) # CPU by default for portability
        self.depth_estimator = DepthEstimator(use_cuda=False)
        self.lane_detector = LaneDetector()
        self.scene_builder = SceneBuilder()
        self.decision_engine = DecisionEngine(history_size=5)
        self.trajectory_planner = TrajectoryPlanner()
        
ctx = None

@app.on_event("startup")
def load_models():
    global ctx
    print("Pre-loading ML engines (PyTorch, Random Forest, MiDaS)...")
    ctx = EngineContext()
    print("Inference Engine Ready.")

@app.post("/predict")
async def predict_frame(file: UploadFile = File(...)):
    if not ctx:
         return JSONResponse({"error": "Model engine uninitialized."}, status_code=503)
         
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return JSONResponse({"error": "Invalid image format"}, status_code=400)
        
    # Standard Pipeline
    perc_out = ctx.perception.run(frame)
    lane_output = ctx.lane_detector.detect_lanes(frame)
    lane_geometry = lane_output["lane_geometry"]
    
    valid_detections = [d for d in perc_out["detections"] if d["score"] > 0]
    
    depth_map = ctx.depth_estimator.estimate(frame)
    scene = ctx.scene_builder.build(valid_detections, depth_map, ctx.depth_estimator, lane_geometry)
    
    trajectory_plan = ctx.trajectory_planner.plan(scene)
    scene["trajectory"] = trajectory_plan["trajectory"]["points"]
    cost = trajectory_plan["trajectory"]["cost"]
    
    decision = ctx.decision_engine.decide(scene)
    
    return {
        "status": "success",
        "decision": {
            "action": decision["action"],
            "reason": decision["reason"],
            "risk_score": decision["risk_score"],
            "ml_fallback": decision["fallback"]
        },
        "planning": {
            "trajectory_type": "cubic_spline",
            "cost": cost,
            "waypoints_count": len(scene["trajectory"])
        },
        "perception": {
            "objects_tracked": len(valid_detections),
            "lane_center_geom": lane_geometry.get("center_line", [])
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

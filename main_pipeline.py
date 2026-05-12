import argparse
import cv2
import json
import os
import numpy as np
from tqdm import tqdm

from dataset.bdd_loader import BDD100kLoader
from perception.hybridnets_wrapper import PerceptionModule
from perception.tracker import ObjectTracker
from perception.depth_estimator import DepthEstimator
from scene_representation.scene_builder import SceneBuilder
from planning.decision_engine import DecisionEngine
from planning.controller import VehicleController
from explainability.gradcam import PerceptionXAI
from explainability.reasoning_engine import ReasoningEngine
from explainability.uncertainty_module import UncertaintyModule
from visualization.visualizer import Visualizer

# Extended Modules
from prediction.motion_predictor import MotionPredictor
from planning.trajectory_planner import TrajectoryPlanner
from evaluation.metrics import PerformanceMetrics
from logs.pipeline_logger import PipelineLogger
from perception.lane_detector import LaneDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "--input", "-i", type=str, default="demo/messy_drive.mp4", help="Path to input video")
    parser.add_argument("--output", "-o", type=str, default="artifacts/output_demo.mp4", help="Path to save output video")
    parser.add_argument("--debug", action="store_true", help="Enable debugging mode for perception")
    args = parser.parse_args()
    
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    
    # Initialize modules
    print("Initializing core modules...")
    loader = BDD100kLoader(args.video)
    perception = PerceptionModule(use_cuda=True)
    tracker = ObjectTracker()
    depth_estimator = DepthEstimator(use_cuda=True)
    lane_detector = LaneDetector()
    scene_builder = SceneBuilder()
    decision_engine = DecisionEngine(history_size=5)
    controller = VehicleController()
    explainability = PerceptionXAI(perception)
    reasoner = ReasoningEngine()
    uncertainty_engine = UncertaintyModule()
    visualizer = Visualizer()
    
    # Initialize extension modules
    motion_predictor = MotionPredictor(fps=30)
    trajectory_planner = TrajectoryPlanner()
    metrics = PerformanceMetrics()
    logger = PipelineLogger()
    
    # Setup Video Writer
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    explanations = []
    
    print(f"Starting ML Integrated pipeline on {args.video}...")
    frame_idx = 0
    
    for data in tqdm(loader.get_frames(), desc="Processing frames"):
        metrics.start_frame()
        frame = data["frame"]
        
        # 1. Perception (HybridNets + OpenCV Geometry)
        perc_out = perception.run(frame, debug=args.debug, frame_idx=frame_idx)
        detections = perc_out["detections"]
        lane_mask = perc_out["lane_mask"]
        drivable_mask = perc_out["drivable_mask"]
        features = perc_out["features"]
        
        lane_output = lane_detector.detect_lanes(frame)
        lane_geometry = lane_output["lane_geometry"]
        
        # Ensure tracker only receives filtered valid detections
        valid_detections = [d for d in detections if d["score"] > 0]
        
        # 2. Tracking
        tracked_objects = tracker.update(valid_detections, frame)
        metrics.log_tracked_objects(len(tracked_objects))
        
        # 3. True Depth Estimation (Calibration scaled)
        depth_map = depth_estimator.estimate(frame)
        
        # 4. Scene Representation (Base)
        scene = scene_builder.build(tracked_objects, depth_map, depth_estimator, lane_geometry)
        
        # 5. Temporal Motion Prediction
        predictions = motion_predictor.predict(scene.get("objects", []))
        for obj in scene.get("objects", []):
            tid = obj.get("track_id")
            for p in predictions:
                if p["track_id"] == tid:
                    obj["velocity"] = p["velocity"]
                    obj["predicted_position"] = p["predicted_position"]
                    break
                    
        # 6. Smooth Trajectory Planning (Splines + Cost)
        trajectory_plan = trajectory_planner.plan(scene)
        scene["trajectory"] = trajectory_plan["trajectory"]["points"]
        
        # 7. ML Decision Engine (Risk scoring + history)
        decision = decision_engine.decide(scene)
        metrics.log_decision(decision)
        
        # 8. Vehicle Control
        commands = controller.control(decision)
        
        # 10. Integrated XAI Reasoning (real GradCAM or saliency proxy)
        input_tensor = features[0] if (isinstance(features, list) and len(features) > 0) else None
        heatmap = explainability.explain_detection(
            frame, input_tensor=input_tensor, detections=valid_detections
        )
        
        # Calculate uncertainty based on detection confidences
        avg_conf = np.mean([d["score"] for d in valid_detections]) if valid_detections else 1.0
        uncertainty = 1.0 - avg_conf
        
        # Generate Natural Language Reasoning
        reasoning_text = reasoner.generate_justification(scene, decision)
        
        # 11. Advanced API Visualization
        vis_frame = visualizer.overlay(frame, scene, valid_detections, lane_mask, drivable_mask, 
                                       heatmap, decision, commands)
        
        latency = metrics.log_latency()
        logger.log_frame(frame_idx, scene.get("objects", []), decision, latency)
        out_video.write(vis_frame)
        
        # Record explanation logs (strip non-serializable objects)
        scene_log = {
            "objects": [
                {k: (float(v) if isinstance(v, (np.floating, np.integer)) else
                     v.tolist() if hasattr(v, 'tolist') else v)
                 for k, v in obj.items() if k != 'predicted_position'}
                for obj in scene.get("objects", [])
            ],
            "lane_geometry": {
                k: ([list(p) for p in v] if isinstance(v, list) else v)
                for k, v in scene.get("lane_geometry", {}).items()
            },
        }
        explanations.append({
            "frame": frame_idx,
            "scene": scene_log,
            "decision": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                         for k, v in decision.items() if k != 'history'},
            "commands": {k: float(v) if isinstance(v, (float, np.floating)) else v
                         for k, v in commands.items()},
            "reasoning": reasoning_text,
            "uncertainty": float(uncertainty),
            "latency": float(latency)
        })
        
        frame_idx += 1
        
    out_video.release()
    
    with open("artifacts/explanations.json", "w") as f:
        json.dump(explanations, f, indent=4)
        
    logger.save()
    metrics.print_summary()
        
    print("\nAdvanced Pipeline complete.")
    print("Saved -> artifacts/output_demo.mp4")

if __name__ == "__main__":
    main()

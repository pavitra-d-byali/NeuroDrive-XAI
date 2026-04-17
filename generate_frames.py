import cv2
import numpy as np
import os

from dataset.bdd_loader import BDD100kLoader
from perception.hybridnets_wrapper import PerceptionModule
from perception.tracker import ObjectTracker
from perception.depth_estimator import DepthEstimator
from scene_representation.scene_builder import SceneBuilder
from planning.decision_engine import DecisionEngine
from visualization.visualizer import Visualizer
from perception.lane_detector import LaneDetector

def main():
    loader = BDD100kLoader("demo/sample_drive.mp4")
    frames = loader.get_frames()
    data = next(frames)
    frame = data["frame"]
    
    os.makedirs("artifacts", exist_ok=True)
    cv2.imwrite("artifacts/input_frame.jpg", frame)
    
    print("Running Perception...")
    perception = PerceptionModule(use_cuda=False)
    perc_out = perception.run(frame, debug=True, frame_idx=0)
    
    lane_detector = LaneDetector()
    lane_detector_out = lane_detector.detect_lanes(frame)
    lane_geometry = lane_detector_out["lane_geometry"]
    
    tracker = ObjectTracker()
    valid_detections = [d for d in perc_out["detections"] if d["score"] > 0]
    tracked_objects = tracker.update(valid_detections, frame)
    
    print("Running Depth Estimation...")
    depth_estimator = DepthEstimator(use_cuda=False)
    depth_map = depth_estimator.estimate(frame)
    
    # Normalize depth map for visualization 0..255
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)
    depth_norm = (depth_norm * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    cv2.imwrite("artifacts/depth_estimation.jpg", depth_color)
    
    print("Creating Processing Frame...")
    scene_builder = SceneBuilder()
    scene = scene_builder.build(tracked_objects, depth_map, depth_estimator, lane_geometry)
    
    decision_engine = DecisionEngine()
    decision = decision_engine.decide(scene)
    
    visualizer = Visualizer()
    vis_frame = visualizer.overlay(frame, scene, [], None, None, None, decision, None)
    cv2.imwrite("artifacts/processing_frame.jpg", vis_frame)
    
    print("Done!")

if __name__ == "__main__":
    main()

import torch
import cv2
import numpy as np
import os

class DepthEstimator:
    def __init__(self, model_type="MiDaS", use_cuda=False):
        # model_type can be "MiDaS" (large), "MiDaS_small", etc.
        model_type = "MiDaS_small"
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        print(f"Loading MiDaS model ({model_type})...")
        
        # Load from the local clone if available to bypass GitHub API rate limit 403
        local_dir = os.path.abspath("weights/MiDaS")
        if os.path.exists(local_dir):
            try:
                self.midas = torch.hub.load(local_dir, model_type, source="local")
                midas_transforms = torch.hub.load(local_dir, "transforms", source="local")
                print("Loaded MiDaS from local clone.")
            except Exception as e:
                print(f"Local MiDaS load failed: {e}. Trying remote...")
                self.midas = torch.hub.load("intel-isl/MiDaS", model_type, skip_validation=True)
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", skip_validation=True)
        else:
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type, skip_validation=True)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", skip_validation=True)
            
        self.midas.to(self.device)
        self.midas.eval()
        
        self.transform = midas_transforms.small_transform
        print("MiDaS loaded.")
            
    def estimate(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        return depth_map
        
    def get_object_distance(self, depth_map, bbox):
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        box_area = depth_map[y1:y2, x1:x2]
        if box_area.size == 0:
            return 100.0
            
        mean_disparity = np.median(box_area) 
        if mean_disparity <= 1e-6:
            return 100.0
            
        # Adjusted empirical scaling calibration for MiDaS relative disparity -> Meters (approximate)
        # Assuming standard dashcam F=700px, camera height 1.5m
        scale_calibration = 25000.0 
        distance_meters = scale_calibration / mean_disparity 
        
        # Upper bound distance and smooth it
        if distance_meters > 150.0:
            distance_meters = 150.0
            
        return round(float(distance_meters), 2)

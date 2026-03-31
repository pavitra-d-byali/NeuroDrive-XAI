import numpy as np
from scipy.interpolate import CubicSpline

class TrajectoryPlanner:
    def __init__(self, img_width=1280, img_height=720):
        self.img_width = img_width
        self.img_height = img_height
        
    def _generate_spline(self, waypoints):
        # waypoints format: [(x, y), (x, y), ...]
        waypoints = np.array(waypoints)
        # Sort by y decreasing (since image y starts from 0 top to bottom)
        waypoints = waypoints[waypoints[:, 1].argsort()[::-1]]
        
        # Ensure points are unique in y 
        _, idx = np.unique(waypoints[:, 1], return_index=True)
        waypoints = waypoints[np.sort(idx)]
        
        if len(waypoints) < 3: return waypoints.tolist()
        
        y = waypoints[:, 1]
        x = waypoints[:, 0]
        
        try:
            cs = CubicSpline(y[::-1], x[::-1]) # Must be strictly increasing
            y_smooth = np.linspace(y[-1], y[0], 20)
            x_smooth = cs(y_smooth)
            
            spline_points = np.vstack((x_smooth[::-1], y_smooth[::-1])).T
            return spline_points.astype(int).tolist()
        except:
            return waypoints.tolist()
            
    def _calculate_cost(self, path, obstacles, center_line):
        cost = 0.0
        
        # 1. Lane deviation penalty (how far from center lane line)
        if center_line and len(center_line) == 4:
            cx1, cy1, cx2, cy2 = center_line
            avg_center_x = (cx1 + cx2) // 2
            for px, py in path:
                cost += abs(px - avg_center_x) * 0.01
                
        # 2. Obstacle Proximity Penalty
        for px, py in path:
            for obs in obstacles:
                # Basic 2D projection distance check
                bx, by = obs.get("predicted_position", [obs["bbox"][0], obs["bbox"][3]])
                dist = np.hypot(px - bx, py - by)
                if dist < 100: # dangerous proximity in pixels
                    cost += 500.0 / max(dist, 1.0)
                    
        # 3. Smoothness (gradient changes) and Curvature (Phase 4)
        path = np.array(path)
        if len(path) > 2:
            dx = np.diff(path[:, 0])
            dy = np.diff(path[:, 1])
            # Avoid division by zero
            angles = np.arctan2(dy, dx)
            curvature = np.diff(angles)
            cost += np.sum(np.abs(np.diff(dx))) * 0.5
            
            # Phase 4: Curvature penalty (kappa squared) -> penalizes sharp impossible steering
            cost += np.sum(curvature**2) * 200.0  
            
        return cost
        
    def plan(self, scene):
        ego_x = self.img_width // 2
        ego_y = self.img_height
        
        objects = scene.get("objects", [])
        lane_geom = scene.get("lane_geometry", {})
        center_line = lane_geom.get("center_line", [])
        
        # We generate 3 candidate paths: Center, Shift Left, Shift Right, and Stop
        candidates = {
            "center": [(ego_x, ego_y), (ego_x, ego_y - 150), (ego_x, ego_y - 300)],
            "left": [(ego_x, ego_y), (ego_x - 100, ego_y - 150), (ego_x - 150, ego_y - 300)],
            "right": [(ego_x, ego_y), (ego_x + 100, ego_y - 150), (ego_x + 150, ego_y - 300)],
            "stop": [(ego_x, ego_y), (ego_x, ego_y - 20), (ego_x, ego_y - 40)]
        }
        
        best_path = None
        lowest_cost = float('inf')
        
        # If pedestrians ahead very close, force stop path
        force_stop = any(obj.get("type", "").lower() == "pedestrian" and obj.get("distance_meters", 100) < 15 for obj in objects)
        
        if force_stop:
            best_path_type = "stop"
            spline = self._generate_spline(candidates["stop"])
            lowest_cost = 0.0
        else:
            best_path_type = "center"
            spline = []
            for name, waypoints in candidates.items():
                if name == "stop": continue # handled by force_stop primarily
                
                path_spline = self._generate_spline(waypoints)
                cost = self._calculate_cost(path_spline, objects, center_line)
                
                if cost < lowest_cost:
                    lowest_cost = cost
                    best_path = path_spline
                    best_path_type = name
                    
            spline = best_path
            
        return {
            "trajectory": {
                "type": "spline",
                "action_type": best_path_type,
                "points": spline,
                "cost": round(lowest_cost, 2)
            }
        }

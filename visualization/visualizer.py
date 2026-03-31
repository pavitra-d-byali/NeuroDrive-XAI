import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def overlay(self, frame, scene, detections, mask_lane, mask_drivable, heatmap, decision, controls):
        h, w = frame.shape[:2]
        vis_frame = frame.copy()
        
        # 1. Overlay Lane Detector Geometry (Task 9)
        lane_geometry = scene.get("lane_geometry", {})
        left_l = lane_geometry.get("left_lane", [])
        right_l = lane_geometry.get("right_lane", [])
        center_l = lane_geometry.get("center_line", [])
        
        if left_l and right_l:
            pts = np.array([
                [left_l[0], left_l[1]],
                [left_l[2], left_l[3]],
                [right_l[2], right_l[3]],
                [right_l[0], right_l[1]]
            ], np.int32).reshape((-1, 1, 2))
            
            overlay = vis_frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 70, 255))
            vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)
            
            # Highlight actual line tracking bounds
            cv2.line(vis_frame, (left_l[0], left_l[1]), (left_l[2], left_l[3]), (0, 0, 255), 3)
            cv2.line(vis_frame, (right_l[0], right_l[1]), (right_l[2], right_l[3]), (0, 0, 255), 3)
        
        # 2. Tracking IDs, Bounding Boxes & Distance
        for obj in scene.get("objects", []):
            x1, y1, x2, y2 = obj["bbox"]
            track_id = obj.get("track_id", "?")
            obj_type = obj.get("type", "unknown")
            dist = obj.get("distance_meters", 0.0)
            velocity = obj.get("velocity", 0.0)
            
            # Risk-Based Color (Red if < 15m)
            color = (0, 0, 255) if dist < 15.0 else (0, 255, 255)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"#{track_id} {obj_type} {dist:.1f}m {velocity:.0f}px/s"
            cv2.putText(vis_frame, label, (x1, max(0, y1-10)), self.font, 0.5, color, 1, cv2.LINE_AA)
                
        # 3. Smoothed Cubic Spline Ego Trajectory
        trajectory_plan = scene.get("trajectory", [])
        if len(trajectory_plan) > 2:
            pts = np.array(trajectory_plan, np.int32).reshape((-1, 1, 2))
            # Shift trajectory to center bottom point anchor
            anchor_shift = w // 2 - trajectory_plan[0][0] 
            cv2.polylines(vis_frame, [pts], False, (255, 0, 255), 4)

        # 4. Neural Risk Heatmap & Decision HUD
        hud_bg = vis_frame[40:220, 40:800].copy()
        cv2.rectangle(vis_frame, (40, 40), (800, 220), (0, 0, 0), -1)
        vis_frame = cv2.addWeighted(vis_frame, 0.6, frame, 0.4, 0)
        
        action = decision.get("action", "Proceed")
        reason = decision.get("reason", "...")
        risk = decision.get("risk_score", 0.0)
        fallback = decision.get("fallback", False)
        
        ac_color = (0, 0, 255) if action == "Brake" else ((0, 215, 255) if action == "Slow" else (0, 255, 0))
        cv2.putText(vis_frame, f"Action: {action} (Risk: {risk:.2f})", (50, 70), self.font, 0.8, ac_color, 2)
        cv2.putText(vis_frame, str(reason)[:60] + "...", (50, 105), self.font, 0.6, (200, 200, 200), 1)
        
        if fallback:
             cv2.putText(vis_frame, "SYSTEM FALLBACK ACTIVATED", (50, 135), self.font, 0.6, (0, 0, 255), 2)
             
        # History
        hist = decision.get("history", [])
        hist_str = " -> ".join(hist[-4:]) if hist else "..."
        cv2.putText(vis_frame, f"History: {hist_str}", (50, 165), self.font, 0.5, (150, 150, 150), 1)
        
        return vis_frame

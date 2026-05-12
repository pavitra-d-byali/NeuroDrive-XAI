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

        # 4. HUD & Performance Telemetry (Point 6/8)
        hud_h = 160
        sub_img = vis_frame[0:hud_h, 0:w]
        white_rect = np.full_like(sub_img, 255)
        vis_frame[0:hud_h, 0:w] = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 0)
        
        # Decision & Action
        action = decision.get("action", "Proceed")
        risk = decision.get("risk_score", 0.0)
        ac_color = (0, 0, 200) if action == "Brake" else ((0, 140, 255) if action == "Slow" else (0, 150, 0))
        cv2.putText(vis_frame, f"NEURODRIVE-XAI | STATUS: {action.upper()}", (20, 40), self.font, 0.9, ac_color, 2)
        cv2.putText(vis_frame, f"Risk Probability: {risk * 100:.1f}%", (20, 75), self.font, 0.6, (50, 50, 50), 1)
        
        # Controls Telemetry
        steer = controls.get("steering", 0.0)
        thr = controls.get("throttle", 0.0)
        brk = controls.get("brake", 0.0)
        
        cv2.putText(vis_frame, f"THR: {thr*100:.0f}% | BRK: {brk*100:.0f}%", (400, 75), self.font, 0.6, (50, 50, 50), 1)
        
        # Visual Steering Wheel Icon (Simplified)
        center_x, center_y = 600, 40
        cv2.circle(vis_frame, (center_x, center_y), 25, (100, 100, 100), 2)
        end_x = int(center_x + 25 * np.sin(steer))
        end_y = int(center_y - 25 * np.cos(steer))
        cv2.line(vis_frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3)
        
        # Performance Box
        cv2.rectangle(vis_frame, (w-250, 10), (w-10, 140), (240, 240, 240), -1)
        cv2.putText(vis_frame, "SYSTEM METRICS", (w-240, 35), self.font, 0.6, (0, 0, 0), 2)
        cv2.putText(vis_frame, f"Latency: {decision.get('latency', 0)*1000:.1f}ms", (w-240, 65), self.font, 0.5, (100, 100, 100), 1)
        cv2.putText(vis_frame, f"Uncertainty: {decision.get('uncertainty', 0)*100:.1f}%", (w-240, 95), self.font, 0.5, (100, 100, 100), 1)
        
        # 5. XAI Heatmap Blending (Task 10)
        if heatmap is not None:
            heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            # Re-scale to 1/4 size and place in corner
            xai_viz = cv2.resize(heatmap_color, (200, 120))
            vis_frame[h-140:h-20, w-220:w-20] = cv2.addWeighted(vis_frame[h-140:h-20, w-220:w-20], 0.3, xai_viz, 0.7, 0)
            cv2.putText(vis_frame, "NEURAL FOCUS (XAI)", (w-220, h-145), self.font, 0.5, (255, 255, 255), 1)
        
        return vis_frame

import cv2
import numpy as np

class LaneDetector:
    def __init__(self, use_polyfit=True):
        self.use_polyfit = use_polyfit
        # Precomputed ROI vertices for a standard dashcam (1280x720 assumed). 
        # Fine-tuned for center-focused highway views
        self.roi_vertices = np.array([[(100, 720), (550, 450), (750, 450), (1280, 720)]], dtype=np.int32)
        
    def _region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)
        match_mask_color = 255
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img
        
    def detect_lanes(self, frame):
        try:
            height, width = frame.shape[:2]
            
            # Step 1: Grayscale and Edge Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            canny = cv2.Canny(blur, 50, 150)
            
            # Step 2: Mask out non-ROI sky/dash
            roi_vertices = np.array([[(100, height), (width//2 - 100, height//2 + 50), 
                                      (width//2 + 100, height//2 + 50), (width, height)]], dtype=np.int32)
            cropped_canny = self._region_of_interest(canny, roi_vertices)
            
            # Step 3: Probabilistic Hough Line Transform
            lines = cv2.HoughLinesP(cropped_canny, rho=2, theta=np.pi/180, threshold=50, 
                                    lines=np.array([]), minLineLength=40, maxLineGap=150)
                                    
            left_fit = []
            right_fit = []
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Filter perfectly horizontal to avoid division by zero
                    if x1 == x2: continue
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    
                    if slope < -0.3: # Left lane slope is negative
                        left_fit.append((slope, intercept))
                    elif slope > 0.3: # Right lane slope is positive
                        right_fit.append((slope, intercept))
                        
            # Safely group and construct coordinate bundles for JSON 
            left_line = self._make_coordinates(frame, left_fit)
            right_line = self._make_coordinates(frame, right_fit)
            
            center_line = []
            if left_line and right_line:
                # Calculate center line geometry by averaging left/right bounds
                center_line = [
                    int((left_line[0] + right_line[0]) / 2),
                    left_line[1],
                    int((left_line[2] + right_line[2]) / 2),
                    left_line[3]
                ]
            else:
                # Fallback center line if poor detection
                center_line = [width//2, height, width//2, height//2 + 50]
                
            return {
                "lane_geometry": {
                    "left_lane": left_line if left_line else [],
                    "right_lane": right_line if right_line else [],
                    "center_line": center_line
                }
            }
            
        except Exception as e:
            # Failure case generic fallback
            h, w = frame.shape[:2]
            return {
                "lane_geometry": {
                    "left_lane": [],
                    "right_lane": [],
                    "center_line": [w//2, h, w//2, h//2 + 50]
                }
            }
            
    def _make_coordinates(self, image, line_parameters):
        if not line_parameters:
            return None
        
        # Average the parameters out
        avg_line = np.average(line_parameters, axis=0)
        slope, intercept = avg_line
        
        # Extrapolate to bottom & slightly past halfway point
        y1 = image.shape[0]
        y2 = int(y1 * (3/5))
        
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        
        return [x1, y1, x2, y2]

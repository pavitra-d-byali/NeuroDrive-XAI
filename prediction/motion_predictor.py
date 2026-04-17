import numpy as np

class MotionPredictor:
    def __init__(self, fps=30):
        self.fps = fps
        self.time_delta = 1.0 / fps
        self.history = {}
        
    def predict(self, tracked_objects):
        predictions = []
        current_ids = set()
        
        for obj in tracked_objects:
            track_id = obj["track_id"]
            current_ids.add(track_id)
            
            bbox = obj["bbox"]
            obj_type = obj.get("class", "unknown")
            
            # Calculate bottom-center for a more grounded position
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = float(bbox[3]) # bottom of bounding box
            current_pos = np.array([cx, cy])
            
            if track_id in self.history:
                last_pos = self.history[track_id]
                # velocity vector in pixels per second
                velocity_vector = (current_pos - last_pos) / self.time_delta
                # Speed magnitude
                velocity_mag = np.linalg.norm(velocity_vector)
                
                # Predict position 1 second into the future
                predicted_pos = current_pos + velocity_vector * 1.0
            else:
                velocity_mag = 0.0
                velocity_vector = np.array([0.0, 0.0])
                predicted_pos = current_pos
                
            self.history[track_id] = current_pos
            
            predictions.append({
                "track_id": track_id,
                "type": obj_type,
                "velocity": round(float(velocity_mag), 2),
                "predicted_position": [int(predicted_pos[0]), int(predicted_pos[1])]
            })
            
        # Cleanup old tracks
        tracks_to_remove = [tid for tid in list(self.history.keys()) if tid not in current_ids]
        for tid in tracks_to_remove:
            del self.history[tid]
            
        return predictions

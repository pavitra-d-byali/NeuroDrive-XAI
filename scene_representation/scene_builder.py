class SceneBuilder:
    def __init__(self):
        self.ego_speed = 30 # Mock constant speed for demo purposes
        
    def build(self, tracked_objects, depth_map, depth_estimator, lane_geometry):
        scene = {
            "speed": self.ego_speed,
            "objects": [],
            "lane_geometry": lane_geometry
        }
        
        # Determine center bounds dynamically
        if lane_geometry and lane_geometry.get("left_lane") and lane_geometry.get("right_lane"):
            left_x_bottom = lane_geometry["left_lane"][0]
            right_x_bottom = lane_geometry["right_lane"][0]
        elif lane_geometry and lane_geometry.get("center_line"):
            # Fallback based on center line
            cx = lane_geometry["center_line"][0]
            left_x_bottom = cx - 200
            right_x_bottom = cx + 200
        else:
            left_x_bottom = 1280 // 3
            right_x_bottom = 2 * (1280 // 3)
            
        for obj in tracked_objects:
            bbox = obj["bbox"]
            dist_m = depth_estimator.get_object_distance(depth_map, bbox)
            
            # Predict lane based on dynamic bottom lane geometry
            obj_center_x = (bbox[0] + bbox[2]) // 2
            
            if obj_center_x < left_x_bottom:
                lane = "left"
            elif obj_center_x > right_x_bottom:
                lane = "right"
            else:
                lane = "center"
                
            scene["objects"].append({
                "track_id": obj["track_id"],
                "type": obj["class"],
                "distance_meters": dist_m,  # Updated key name
                "lane": lane,
                "bbox": bbox
            })
            
        return scene

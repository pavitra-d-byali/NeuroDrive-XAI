import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_hybrid_dataset(num_samples=10000, output_path="dataset/hybrid_features.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Feature boundaries
    # distance: 0 to 100m (None = 100)
    # relative_v: -20 m/s (approaching fast) to +20 m/s (pulling away)
    # lane_offset: -2.0m (left) to +2.0m (right)
    # curvature: -0.1 (sharp left) to 0.1 (sharp right)
    
    types = ["car", "pedestrian", "bike", "truck", "none"]
    
    data = []
    
    # 70% "Normal" base driving (Real simulation)
    # 30% "Edge Cases" (Synthetic limit testing)
    
    for i in range(num_samples):
        is_edge_case = (i > num_samples * 0.7)
        
        if is_edge_case:
            # Inject extreme cases (sudden cut-ins, sharp curves)
            dist = np.random.uniform(0.5, 15.0)
            rel_v = np.random.uniform(-15.0, -5.0) # Dangerous approach
            offset = np.random.uniform(-2.0, 2.0)
            curv = np.random.uniform(-0.1, 0.1)
            num_obj = np.random.randint(1, 5)
            obj_type = np.random.choice(["car", "pedestrian", "truck"])
        else:
            # Normal driving logic
            dist = np.random.uniform(20.0, 100.0)
            rel_v = np.random.uniform(-2.0, 5.0)
            offset = np.random.normal(0, 0.3) # Keep near center
            curv = np.random.normal(0, 0.02)
            num_obj = np.random.randint(0, 3)
            obj_type = np.random.choice(types, p=[0.5, 0.05, 0.05, 0.2, 0.2])
            
        if num_obj == 0:
            obj_type = "none"
            dist = 100.0
            
        # Target determination (Logic simulator acting as labeler)
        brake = 0
        steering = -offset * 0.5 # Basic corrective steer back to center
        
        # Adjust steer for curvature
        steering += (curv * 10) 
        
        if obj_type in ["pedestrian", "bike"] and dist < 15.0:
            brake = 1
            
        if dist < 20.0 and rel_v < -2.0:
            brake = 1 # Approaching fast
            
        if dist < 5.0:
            brake = 1 # Emergency
        
        # Clamp steering
        steering = max(min(steering, 1.0), -1.0)
            
        data.append({
            "distance_to_object": dist,
            "relative_velocity": rel_v,
            "lane_offset": offset,
            "lane_curvature": curv,
            "num_objects": num_obj,
            "closest_object_type": obj_type,
            "steering_angle": steering,
            "brake": brake
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Hybrid dataset generated with {len(df)} records at {output_path}")

if __name__ == "__main__":
    generate_hybrid_dataset()

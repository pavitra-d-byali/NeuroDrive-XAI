import time
import pandas as pd
import numpy as np
import torch
from evaluation.metrics import NeuroMetrics
from decision.mlp_model import NeuroDecisionMLP
from decision.temporal import TemporalSmoother

def run_scenario_benchmarks(data_path="dataset/hybrid_features.csv"):
    print("--- NeuroDrive Top 1% System & Robustness Benchmark ---")
    
    df = pd.read_csv(data_path)
    model = NeuroDecisionMLP(input_features=6)
    
    # Baseline comparison models
    window_size = 5
    smoother = TemporalSmoother(model, window_size=window_size)
    
    scenarios = {
        "Highway": df[(df['num_objects'] <= 1) & (df['lane_curvature'].abs() < 0.05)],
        "Traffic": df[(df['num_objects'] > 2) & (df['distance_to_object'] < 40)],
        "Sudden Obstacle": df[(df['distance_to_object'] < 15) & (df['relative_velocity'] < -5.0)]
    }
    
    print("\n[1/4] Performance vs Smoothing Baseline...")
    
    results = []
    
    for name, scen_df in scenarios.items():
        if len(scen_df) == 0:
            continue
            
        y_truths = []
        raw_preds = []
        smooth_preds = []
        
        for _, row in scen_df.iterrows():
            noisy_dist = max(0.5, row['distance_to_object'] + np.random.normal(0, row['distance_to_object'] * 0.1))
            t = torch.FloatTensor([
                noisy_dist, row['relative_velocity'], row['lane_offset'], 
                row['lane_curvature'], row['num_objects'], 0
            ])
            
            y_truths.append(row['brake'])
            
            model.eval()
            with torch.no_grad():
                _, rp = model(t)
                raw_b = 1 if rp.item() > 0.5 else 0
                
            _, fb, _ = smoother.predict(t)
            
            raw_preds.append(raw_b)
            smooth_preds.append(fb)
            
        y_truths = np.array(y_truths)
        raw_brakes = np.array(raw_preds)
        smooth_brakes = np.array(smooth_preds)
        
        raw_fp = np.sum((y_truths == 0) & (raw_brakes == 1))
        smooth_fp = np.sum((y_truths == 0) & (smooth_brakes == 1))
        total_safe = np.sum(y_truths == 0) if np.sum(y_truths == 0) > 0 else 1
        
        r_collision = np.sum((y_truths == 1) & (raw_brakes == 0))
        s_collision = np.sum((y_truths == 1) & (smooth_brakes == 0))
        total_danger = np.sum(y_truths == 1) if np.sum(y_truths == 1) > 0 else 1
        
        results.append({
            "Scenario": name,
            "Frames": len(scen_df),
            "Raw False Brake (%)": (raw_fp/total_safe)*100,
            "Smooth False Brake (%)": (smooth_fp/total_safe)*100,
            "Raw Missed Brake (%)": (r_collision/total_danger)*100,
            "Smooth Missed Brake (%)": (s_collision/total_danger)*100
        })
        
    res_df = pd.DataFrame(results)
    print("\n| Scenario            | Raw False Brake | Smooth False Brake | Raw Missed | Smooth Missed |")
    print("|--------------------|----------------|--------------------|------------|---------------|")
    for _, row in res_df.iterrows():
        print(f"| {row['Scenario']:<18} | {row['Raw False Brake (%)']:5.2f}%         | {row['Smooth False Brake (%)']:5.2f}%             | {row['Raw Missed Brake (%)']:5.2f}%     | {row['Smooth Missed Brake (%)']:5.2f}%        |")
    
    # 2. Reaction Delay Analysis
    print("\n[2/4] Brake Reaction Time Analysis...")
    fps_time_ms = 44.81
    reaction_delay_ms = 3 * fps_time_ms
    print(f"-> Theoretical Hardware Latency : {fps_time_ms:.1f} ms per frame")
    print(f"-> Smoother Induced Delay       : {reaction_delay_ms:.1f} ms (Requires threshold bridging)")
    
    print("\n[3/4] Temporal Variance Analysis...")
    var_raw = np.var(raw_preds)
    var_smooth = np.var(smooth_preds)
    reduction = ((var_raw - var_smooth) / var_raw) * 100 if var_raw > 0 else 0
    print(f"-> Frame jitter variance mathematically reduced by ~{reduction:.1f}%")
    
    print("\n[4/4] Feature Sensitivity & Robustness Analysis...")
    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30] # 0% to 30% sensor variance
    print("| Depth Noise (%) | False Brake Rate | Missed Brake Rate |")
    print("|-----------------|------------------|-------------------|")
    
    # Re-initialize smoother to clear history state for controlled testing
    test_df = df.iloc[-2000:].reset_index(drop=True)
    
    for noise in noise_levels:
        local_smoother = TemporalSmoother(model, window_size=5)
        local_fp = 0
        local_miss = 0
        
        y_t = []
        y_p = []
        
        for _, row in test_df.iterrows():
            noise_val = row['distance_to_object'] * noise
            noisy_dist = max(0.5, row['distance_to_object'] + np.random.normal(0, noise_val))
            t = torch.FloatTensor([
                noisy_dist, row['relative_velocity'], row['lane_offset'], 
                row['lane_curvature'], row['num_objects'], 0
            ])
            
            _, fb, _ = local_smoother.predict(t)
            
            y_t.append(row['brake'])
            y_p.append(fb)
            
        y_t_arr = np.array(y_t)
        y_p_arr = np.array(y_p)
        
        total_s = np.sum(y_t_arr == 0) if np.sum(y_t_arr == 0) > 0 else 1
        total_d = np.sum(y_t_arr == 1) if np.sum(y_t_arr == 1) > 0 else 1
        
        fp_rate = (np.sum((y_t_arr == 0) & (y_p_arr == 1)) / total_s) * 100
        miss_rate = (np.sum((y_t_arr == 1) & (y_p_arr == 0)) / total_d) * 100
        
        print(f"| {noise*100:13.1f}% | {fp_rate:15.2f}% | {miss_rate:16.2f}% |")

    print("\n--- Benchmarks Finalized ---")

if __name__ == "__main__":
    run_scenario_benchmarks()

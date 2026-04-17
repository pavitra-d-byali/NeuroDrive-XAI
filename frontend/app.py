import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="NeuroDrive-XAI System Dashboard")

# Dark Theme CSS enforcing strict system look
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .css-1d391kg { background-color: #1a1c23; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ NeuroDrive-XAI System Dashboard")

METERS_TO_DEG = 1 / 111000.0

@st.cache_data
def load_and_structure_data():
    if os.path.exists("dataset/hybrid_features.csv"):
        df = pd.read_csv("dataset/hybrid_features.csv")
    else:
        # Fallback dataset mapping
        df = pd.DataFrame({
            "distance_to_object": np.random.uniform(5, 50, 100),
            "relative_velocity": np.random.uniform(-10, 10, 100),
            "lane_offset": np.random.uniform(-1.5, 1.5, 100),
            "steering_angle": np.random.uniform(-0.5, 0.5, 100),
            "brake": np.random.choice([0, 1], 100)
        })

    frames = []
    
    for idx, row in df.iterrows():
        dist = row['distance_to_object']
        offset = row['lane_offset']
        steer = row['steering_angle']
        brake = int(row['brake'])
        
        is_critical = (brake == 1)
        
        # Calculate pseudo-trajectory curved by steering angle
        mid_y = (dist / 2) * METERS_TO_DEG
        mid_x = (steer * 20.0) * METERS_TO_DEG 
        end_y = dist * METERS_TO_DEG
        end_x = (steer * 40.0) * METERS_TO_DEG
        
        traj = [[0, 0], [mid_x, mid_y], [end_x, end_y]]
        
        # Calculate Counterfactual
        # Mathematical approximation: how much distance Delta to flip decision
        if brake == 1:
            delta = round((15.0 - dist) + 0.1, 1) if dist < 15.0 else 2.5
        else:
            delta = round(-abs(dist - 14.5), 1)
            
        objs = []
        if dist < 99.0:
            objs.append({
                "lon": offset * METERS_TO_DEG,
                "lat": dist * METERS_TO_DEG,
                "is_critical": is_critical
            })

        frame = {
            "risk": 0.95 if brake == 1 else 0.05,
            "brake_prob": 0.88 if brake == 1 else 0.02,
            "action": "BRAKE" if brake == 1 else "CONTINUE",
            "features": {
                "distance_to_object": round(dist, 1),
                "lane_offset": round(offset, 2),
                "relative_velocity": round(row['relative_velocity'], 1)
            },
            "objects": objs,
            "trajectory": traj,
            "counterfactual": {
                "delta": delta
            }
        }
        frames.append(frame)
    return frames

frames = load_and_structure_data()
num_frames = len(frames)

# ==========================================
# 🎞️ TIMELINE CONTROL (MASTER SYNC)
# ==========================================
frame_idx = st.slider("System Frame Synchronization", 0, num_frames - 1, 50)
frame = frames[frame_idx]

# ==========================================
# 🚗 PYDECK SCENE (EGO + OBJECTS + TRAJECTORY)
# ==========================================

# Ego Vehicle
ego_layer = pdk.Layer(
    "ScatterplotLayer",
    data=[{"lat": 0, "lon": 0}],
    get_position="[lon, lat]",
    get_radius=1.5,
    radius_scale=1,
    get_fill_color=[0, 255, 150, 200], # Neon Green
)

# Objects
obj_layer = pdk.Layer(
    "ScatterplotLayer",
    data=frame["objects"],
    get_position="[lon, lat]",
    get_radius=3,
    get_fill_color="[255, 50, 50, 250] if is_critical else [50, 150, 255, 250]",
)

# Predicted Trajectory Path
traj_layer = pdk.Layer(
    "PathLayer",
    data=[{"path": frame["trajectory"]}],
    get_path="path",
    get_width=2,
    width_min_pixels=4,
    get_color=[255, 165, 0, 200], # Orange projection
)

view_state = pdk.ViewState(
    latitude=0.0001,
    longitude=0,
    zoom=19.5,
    pitch=60, # 3D Angled Pitch
)

deck = pdk.Deck(
    layers=[ego_layer, obj_layer, traj_layer],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/dark-v10",
    tooltip={"text": "Coordinates Locked"}
)

# Render Scene
st.pydeck_chart(deck, use_container_width=True)

# ==========================================
# 🚥 METRICS AND XAI PANELS
# ==========================================

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Telemetry History")
    # Grab rolling subset up to current timeframe
    start_idx = max(0, frame_idx - 50)
    history = pd.DataFrame(frames[start_idx:frame_idx+1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history["risk"],
        mode="lines",
        name="Risk Factor",
        line=dict(color="orange")
    ))

    fig.add_trace(go.Scatter(
        y=history["brake_prob"],
        mode="lines",
        name="Braking Actuation",
        line=dict(color="red"),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.1)'
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=200,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor='#333333', range=[0, 1.1])
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("XAI Explainer")
    
    act_color = "🔴" if frame['action'] == "BRAKE" else "🟢"
    st.markdown(f"**Action Executed:** {act_color} {frame['action']}")

    st.markdown(f"""
    **Sensory Tensors:**
    - Distance to Object: `{frame['features']['distance_to_object']} m`
    - Ego Lane Offset: `{frame['features']['lane_offset']} m`
    - Relative Velocity: `{frame['features']['relative_velocity']} m/s`

    **Mathematical Counterfactual:**
    """)
    
    delta = frame['counterfactual']['delta']
    sign = "+" if delta > 0 else ""
    target = "CONTINUE" if frame['action'] == "BRAKE" else "BRAKE"
    
    st.info(f"If Distance altered by **{sign}{delta}m** → {target}")

# ==========================================
# 📈 GLOBAL DECISION TIMELINE
# ==========================================
st.subheader("Global Action Sequence")

actions = [1 if f["action"] == "BRAKE" else 0 for f in frames]

timeline_fig = go.Figure()
timeline_fig.add_trace(go.Scatter(
    y=actions,
    mode="lines",
    line=dict(shape="hv", color="#00ffcc"), # Step function line
    name="Timeline"
))

# Mark the specific synchronized frame physically on the timeline
timeline_fig.add_vline(x=frame_idx, line_width=2, line_dash="dash", line_color="red")

timeline_fig.update_layout(
    margin=dict(l=0, r=0, t=10, b=20),
    height=120,
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(color="white"),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, tickvals=[0, 1], ticktext=["STR", "BRK"])
)

st.plotly_chart(timeline_fig, use_container_width=True)

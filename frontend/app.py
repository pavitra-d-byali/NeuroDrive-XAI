import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objs as go

st.set_page_config(page_title="NeuroDrive AVS", layout="wide", initial_sidebar_state="collapsed")

# Inject Dark Mode and custom CSS
st.markdown("""
<style>
    /* Darken the background explicitly */
    .stApp {
        background-color: #0e1117;
    }
    .css-1d391kg {
        background-color: #1a1c23;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_log():
    # If the file exists, load it. Otherwise mock bounds.
    if os.path.exists("dataset/hybrid_features.csv"):
        df = pd.read_csv("dataset/hybrid_features.csv")
    else:
        df = pd.DataFrame({
            "distance_to_object": np.random.uniform(5, 50, 100),
            "relative_velocity": np.random.uniform(-10, 10, 100),
            "lane_offset": np.random.uniform(-1.5, 1.5, 100),
            "lane_curvature": np.random.uniform(-0.1, 0.1, 100),
            "brake": np.random.choice([0, 1], 100)
        })
    return df

df = load_sample_log()

st.title("🛰️ NeuroDrive Streetscape.gl")

# Frame Selection
frame_idx = st.slider("Timeline Control", 0, len(df)-1, 50, label_visibility="collapsed")
selected = df.iloc[frame_idx]

# Map physical coordinates into long/lat offset logic to trick PyDeck into rendering local X/Y bounding boxes
# Base center (fake GPS matching an arbitrary intersection)
BASE_LAT = 37.7749
BASE_LON = -122.4194

# 1 degree of lat ~ 111km. So 1 meter map offset = (1 / 111,000)
METERS_TO_DEG = 1 / 111000.0

ego_data = pd.DataFrame([{
    "lat": BASE_LAT, 
    "lon": BASE_LON,
    "color": [0, 255, 150, 200], # Neon Green Ego Car
    "elevation": 1.5
}])

obj_dist = selected['distance_to_object']
obj_offset = selected['lane_offset']

# Calculate 3D Obstacle coordinate
obs_data = pd.DataFrame([{
    "lat": BASE_LAT + (obj_dist * METERS_TO_DEG),
    "lon": BASE_LON + (obj_offset * METERS_TO_DEG),
    "color": [50, 150, 255, 230] if selected['brake'] == 0 else [255, 50, 50, 230], # Blue if safe, Red if colliding
    "elevation": 2.5
}])

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### Telemetry Streams")
    
    # Velocity Plotly Dashboard
    historical_vel = df['relative_velocity'].iloc[max(0, frame_idx-20):frame_idx+1]
    fig_vel = go.Figure()
    fig_vel.add_trace(go.Scatter(
        y=historical_vel.values,
        mode='lines',
        line=dict(color='#888888', width=2),
        fill='tozeroy',
        fillcolor='rgba(150, 150, 150, 0.1)'
    ))
    fig_vel.update_layout(
        title="Relative Velocity",
        margin=dict(l=0, r=0, t=30, b=0),
        height=200,
        paper_bgcolor="#1a1c23",
        plot_bgcolor="#1a1c23",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor='#333333')
    )
    st.plotly_chart(fig_vel, use_container_width=True)

    # Risk Probability
    historical_brake = df['brake'].iloc[max(0, frame_idx-20):frame_idx+1].rolling(3).mean()
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        y=historical_brake.values,
        mode='lines',
        line=dict(color='yellow' if selected['brake'] == 0 else 'red', width=2)
    ))
    fig_acc.update_layout(
        title="Probability of Action",
        margin=dict(l=0, r=0, t=30, b=0),
        height=200,
        paper_bgcolor="#1a1c23",
        plot_bgcolor="#1a1c23",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor='#333333')
    )
    st.plotly_chart(fig_acc, use_container_width=True)
    
    st.metric("Neural Frame Decision", "🟢 CONTINUE" if selected['brake'] == 0 else "🛑 INITIATE BRAKE")


with col2:
    # 3D Pydeck Visualization
    # Render Ego Vehicle
    ego_layer = pdk.Layer(
        "ColumnLayer",
        data=ego_data,
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=1,
        radius=1.5, # 1.5 meters wide
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )
    
    # Render Detected Object Bounding Box
    if obj_dist < 99.0: # Ignore objects that are logged at 100m (none)
        obs_layer = pdk.Layer(
            "ColumnLayer",
            data=obs_data,
            get_position=["lon", "lat"],
            get_elevation="elevation",
            elevation_scale=1,
            radius=1.5,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )
        layers = [ego_layer, obs_layer]
    else:
        layers = [ego_layer]

    view_state = pdk.ViewState(
        latitude=BASE_LAT + (15.0 * METERS_TO_DEG), # Look slightly ahead of ego
        longitude=BASE_LON,
        zoom=19.5,
        pitch=60, # 3D Angled Pitch replicating Streetscape.gl
        bearing=0,
    )
    
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10", # Enforce absolute dark mode maps
        tooltip={"text": "Object Boundary"}
    )
    
    st.pydeck_chart(r, use_container_width=True)

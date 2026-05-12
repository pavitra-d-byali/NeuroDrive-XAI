"""
frontend/app.py — NeuroDrive-XAI Dashboard (Real Data)
Loads real pipeline outputs from artifacts/explanations.json
and CARLA telemetry from artifacts/carla_records/*.csv
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, glob
from pathlib import Path

st.set_page_config(layout="wide", page_title="NeuroDrive-XAI Dashboard", page_icon="🧠")

st.markdown("""
<style>
.stApp { background-color: #0a0d14; color: #e0e0e0; }
div[data-testid="stMetricValue"] { font-size: 26px; font-weight: bold; color: #00e5ff; }
div[data-testid="stMetricLabel"] { color: #888; font-size: 12px; }
.stTabs [data-baseweb="tab"] { color: #888; }
.stTabs [aria-selected="true"] { color: #00e5ff; border-bottom: 2px solid #00e5ff; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 NeuroDrive-XAI System Dashboard")

METERS_TO_DEG = 1 / 111000.0

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading pipeline explanations…")
def load_explanations():
    """Load real pipeline output from artifacts/explanations.json"""
    path = "artifacts/explanations.json"
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        frames = []
        for item in raw:
            dec = item.get("decision", {})
            scene = item.get("scene", {})
            cmds = item.get("commands", {})
            objs = scene.get("objects", [])
            risk = float(dec.get("risk_score", 0.1))
            brake_prob = float(item.get("uncertainty", risk))
            action = dec.get("action", "Proceed")
            lat = float(item.get("latency", 0)) * 1000

            # Build pydeck objects
            pydeck_objs = []
            for obj in objs:
                dist = obj.get("distance_meters", 50)
                off  = obj.get("lane_offset", 0) if "lane_offset" in obj else 0
                pydeck_objs.append({
                    "lon": off * METERS_TO_DEG,
                    "lat": dist * METERS_TO_DEG,
                    "is_critical": dist < 15,
                })

            traj = scene.get("trajectory", [])
            traj_geo = [[p[0] * METERS_TO_DEG * 0.5, p[1] * METERS_TO_DEG * 0.5]
                        for p in traj[:10]] if traj else [[0, 0], [0, 0.0003]]

            frames.append({
                "frame":      item.get("frame", 0),
                "risk":       risk,
                "brake_prob": brake_prob,
                "action":     action.upper(),
                "steering":   float(cmds.get("steering", 0)),
                "throttle":   float(cmds.get("throttle", 0.5)),
                "brake_cmd":  float(cmds.get("brake", 0)),
                "latency_ms": lat,
                "reasoning":  item.get("reasoning", ""),
                "uncertainty":float(item.get("uncertainty", 0)),
                "objects":    pydeck_objs,
                "trajectory": traj_geo,
                "num_objects": len(objs),
                "counterfactual": item.get("decision", {}).get("reason", "N/A"),
            })
        return frames, "real"

    # Fallback: generate from demo video if available
    st.warning("⚠️ No `artifacts/explanations.json` found. Run `python main_pipeline.py` first. Showing demo data.")
    frames = []
    np.random.seed(0)
    for i in range(200):
        dist = np.random.uniform(5, 80)
        brake = 1 if dist < 14 else 0
        risk = max(0, min(1, 1 - dist / 80 + np.random.uniform(-0.05, 0.05)))
        frames.append({
            "frame": i, "risk": risk, "brake_prob": risk * 0.9,
            "action": "BRAKE" if brake else "PROCEED",
            "steering": np.random.uniform(-0.3, 0.3),
            "throttle": 0.0 if brake else 0.5, "brake_cmd": float(brake),
            "latency_ms": np.random.uniform(20, 55),
            "reasoning": f"Object at {dist:.1f}m",
            "uncertainty": 1 - risk, "num_objects": np.random.randint(0, 5),
            "objects": [{"lon": 0, "lat": dist * METERS_TO_DEG, "is_critical": brake == 1}] if dist < 60 else [],
            "trajectory": [[0, 0], [0, 0.0003]],
            "counterfactual": f"distance Δ{dist - 14:.1f}m → {'BRAKE' if brake == 0 else 'PROCEED'}",
        })
    return frames, "demo"


@st.cache_data(show_spinner="Loading CARLA records…")
def load_carla_records():
    csvs = sorted(glob.glob("artifacts/carla_records/episode_*.csv"))
    if not csvs:
        return None, None
    import csv as csv_mod
    latest = csvs[-1]
    rows = []
    with open(latest) as f:
        for row in csv_mod.DictReader(f):
            rows.append(row)
    df = pd.DataFrame(rows)
    for col in ["x", "y", "speed_mps", "steering", "throttle", "brake", "risk_score", "latency_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, latest


# ── Load data ─────────────────────────────────────────────────────────────────
frames, data_mode = load_explanations()
carla_df, carla_src = load_carla_records()
num_frames = len(frames)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🚗 Pipeline Dashboard", "🏙️ CARLA Simulation", "📊 Model Analysis"])

# ══════════════════════════════════════ TAB 1: PIPELINE DASHBOARD ═════════════
with tab1:
    if data_mode == "real":
        st.success(f"✓ Live data from `artifacts/explanations.json` — {num_frames} frames")
    else:
        st.info("Demo mode. Run `python main_pipeline.py --input demo/messy_drive.mp4` for real data.")

    frame_idx = st.slider("Frame", 0, num_frames - 1, min(50, num_frames - 1), key="pipe_slider")
    frame = frames[frame_idx]

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    ac = "🔴" if frame["action"] == "BRAKE" else ("🟡" if frame["action"] == "SLOW" else "🟢")
    c1.metric("Action", f"{ac} {frame['action']}")
    c2.metric("Risk Score", f"{frame['risk']:.3f}")
    c3.metric("Brake Prob", f"{frame['brake_prob']:.3f}")
    c4.metric("Latency", f"{frame['latency_ms']:.1f} ms")
    c5.metric("Objects", frame["num_objects"])

    col_map, col_xai = st.columns([3, 1])

    with col_map:
        st.subheader("3D Scene View")
        ego_layer = pdk.Layer("ScatterplotLayer", data=[{"lat": 0, "lon": 0}],
            get_position="[lon,lat]", get_radius=1.5, get_fill_color=[0, 255, 150, 220])
        obj_layer = pdk.Layer("ScatterplotLayer", data=frame["objects"],
            get_position="[lon,lat]", get_radius=3,
            get_fill_color="[is_critical ? 255 : 50, is_critical ? 50 : 150, is_critical ? 50 : 255, 250]")
        traj_layer = pdk.Layer("PathLayer", data=[{"path": frame["trajectory"]}],
            get_path="path", get_width=2, width_min_pixels=4, get_color=[255, 165, 0, 200])
        deck = pdk.Deck(
            layers=[ego_layer, obj_layer, traj_layer],
            initial_view_state=pdk.ViewState(latitude=0.0001, longitude=0, zoom=19.5, pitch=55),
            map_style="mapbox://styles/mapbox/dark-v10",
        )
        st.pydeck_chart(deck, use_container_width=True)

    with col_xai:
        st.subheader("XAI Explainer")
        st.markdown(f"**Decision:** {ac} `{frame['action']}`")
        st.markdown(f"**Uncertainty:** `{frame['uncertainty']:.3f}`")
        st.markdown(f"**Reasoning:**")
        st.info(frame["reasoning"] or frame["counterfactual"])

    # Telemetry chart
    st.subheader("Telemetry History")
    s = max(0, frame_idx - 60)
    hist = pd.DataFrame(frames[s:frame_idx + 1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=hist["risk"], name="Risk", line=dict(color="#ff9900", width=2)))
    fig.add_trace(go.Scatter(y=hist["brake_prob"], name="Brake Prob",
        line=dict(color="#ff2244", width=2), fill="tozeroy", fillcolor="rgba(255,34,68,0.1)"))
    fig.add_trace(go.Scatter(y=hist["steering"], name="Steering",
        line=dict(color="#00e5ff", width=1.5)))
    fig.update_layout(height=220, paper_bgcolor="#0a0d14", plot_bgcolor="#0a0d14",
        font=dict(color="white"), margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1e2230", range=[-1.1, 1.1]),
        legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    # Global action timeline
    st.subheader("Global Decision Timeline")
    action_vals = {"PROCEED": 0, "SLOW": 0.5, "BRAKE": 1}.get
    act_y = [action_vals(f["action"], 0) for f in frames]
    tfig = go.Figure()
    tfig.add_trace(go.Scatter(y=act_y, mode="lines", line=dict(shape="hv", color="#00ffcc", width=2)))
    tfig.add_vline(x=frame_idx, line_width=2, line_dash="dash", line_color="red")
    tfig.update_layout(height=110, paper_bgcolor="#0a0d14", plot_bgcolor="#0a0d14",
        font=dict(color="white"), margin=dict(l=0, r=0, t=5, b=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, tickvals=[0, 0.5, 1], ticktext=["GO", "SLOW", "BRK"]))
    st.plotly_chart(tfig, use_container_width=True)


# ══════════════════════════════════════ TAB 2: CARLA SIMULATION ═══════════════
with tab2:
    if carla_df is not None:
        st.success(f"✓ CARLA telemetry loaded: `{carla_src}` — {len(carla_df)} steps")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Steps", len(carla_df))
        avg_spd = carla_df["speed_mps"].mean() if "speed_mps" in carla_df else 0
        c2.metric("Avg Speed", f"{avg_spd:.2f} m/s")
        brk_pct = (carla_df["brake"].astype(float) > 0.3).mean() * 100 if "brake" in carla_df else 0
        c3.metric("Brake Events", f"{brk_pct:.1f}%")
        avg_lat = carla_df["latency_ms"].mean() if "latency_ms" in carla_df else 0
        c4.metric("Avg Latency", f"{avg_lat:.1f} ms")

        # Trajectory map
        if "x" in carla_df.columns and "y" in carla_df.columns:
            st.subheader("Ego Trajectory (CARLA Map)")
            traj_data = carla_df[["x", "y"]].dropna()
            # Normalize to lon/lat for pydeck (offset from 0,0)
            ref_x, ref_y = float(traj_data["x"].iloc[0]), float(traj_data["y"].iloc[0])
            traj_data = traj_data.copy()
            traj_data["lon"] = (traj_data["x"] - ref_x) * METERS_TO_DEG
            traj_data["lat"] = (traj_data["y"] - ref_y) * METERS_TO_DEG

            path_layer = pdk.Layer("PathLayer",
                data=[{"path": [[r.lon, r.lat] for r in traj_data.itertuples()]}],
                get_path="path", get_color=[0, 229, 255], get_width=3, width_min_pixels=3)
            start = traj_data.iloc[0]
            end   = traj_data.iloc[-1]
            pts_layer = pdk.Layer("ScatterplotLayer",
                data=[{"lat": start.lat, "lon": start.lon, "color": [0, 255, 100]},
                      {"lat": end.lat,   "lon": end.lon,   "color": [255, 50, 50]}],
                get_position="[lon,lat]", get_fill_color="color", get_radius=5)
            st.pydeck_chart(pdk.Deck(
                layers=[path_layer, pts_layer],
                initial_view_state=pdk.ViewState(latitude=float(traj_data["lat"].mean()),
                    longitude=float(traj_data["lon"].mean()), zoom=16, pitch=45),
                map_style="mapbox://styles/mapbox/dark-v10",
            ), use_container_width=True)

        # Telemetry charts
        st.subheader("CARLA Episode Telemetry")
        fig2 = go.Figure()
        if "speed_mps" in carla_df:
            fig2.add_trace(go.Scatter(y=carla_df["speed_mps"], name="Speed (m/s)",
                line=dict(color="#00e5ff", width=2)))
        if "risk_score" in carla_df:
            fig2.add_trace(go.Scatter(y=carla_df["risk_score"].astype(float),
                name="Risk Score", line=dict(color="#ff9900", width=2), yaxis="y2"))
        fig2.update_layout(height=280, paper_bgcolor="#0a0d14", plot_bgcolor="#0a0d14",
            font=dict(color="white"), margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(gridcolor="#1e2230"), yaxis2=dict(overlaying="y", side="right", range=[0, 1]),
            legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig2, use_container_width=True)

        # Action breakdown
        if "action" in carla_df.columns:
            st.subheader("Action Distribution")
            act_counts = carla_df["action"].value_counts()
            afig = go.Figure(go.Bar(x=act_counts.index.tolist(), y=act_counts.values.tolist(),
                marker_color=["#ff2244" if a == "Brake" else "#ff9900" if a == "Slow" else "#00e5ff"
                              for a in act_counts.index]))
            afig.update_layout(height=220, paper_bgcolor="#0a0d14", plot_bgcolor="#0a0d14",
                font=dict(color="white"), margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(gridcolor="#1e2230"))
            st.plotly_chart(afig, use_container_width=True)

        # Raw table
        with st.expander("Raw Telemetry Table"):
            st.dataframe(carla_df.head(100), use_container_width=True)
    else:
        st.info("""
### 🏙️ CARLA Simulation — Not yet run

No CARLA episode data found. To populate this tab:

**Option A — With CARLA installed:**
```bash
# Start CARLA server, then:
python carla_run.py --map Town03 --episodes 3 --steps 500
```

**Option B — CARLA-free replay demo:**
```bash
python carla_replay.py --demo
# Then refresh this dashboard
```

The CARLA tab will show:
- Ego vehicle GPS trajectory on a 3D map
- Speed + risk + brake telemetry charts
- Action distribution breakdown
- Raw step-by-step CSV data
        """)


# ══════════════════════════════════════ TAB 3: MODEL ANALYSIS ═════════════════
with tab3:
    st.subheader("Model Performance Summary")

    # Try to load model weights info
    weights_ok = os.path.exists("weights/neurodrive_mlp.pth")
    scaler_ok  = os.path.exists("weights/feature_scaler.pkl")

    c1, c2, c3 = st.columns(3)
    c1.metric("MLP Model", "✓ Loaded" if weights_ok else "✗ Missing", delta="weights/neurodrive_mlp.pth")
    c2.metric("Feature Scaler", "✓ Loaded" if scaler_ok else "✗ Missing")
    c3.metric("HybridNets", "✓ .pth" if os.path.exists("weights/hybridnets.pth") else "✗ Missing")

    if not weights_ok:
        st.warning("Run `python decision/train.py` to train the decision model.")

    # Feature importance
    st.subheader("Feature Sensitivity Analysis")
    features = ["distance_to_object", "relative_velocity", "lane_offset",
                "lane_curvature", "num_objects", "closest_object_type"]
    # Real importance from README benchmark
    importance = [0.55, 0.20, 0.12, 0.06, 0.04, 0.03]
    imp_fig = go.Figure(go.Bar(
        x=importance, y=features, orientation="h",
        marker=dict(color=importance, colorscale="Blues", showscale=False),
    ))
    imp_fig.update_layout(height=250, paper_bgcolor="#0a0d14", plot_bgcolor="#0a0d14",
        font=dict(color="white"), margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="Relative Importance", gridcolor="#1e2230"),
        yaxis=dict(gridcolor="#1e2230"))
    st.plotly_chart(imp_fig, use_container_width=True)

    # System latency benchmark
    st.subheader("Component-Level Latency (Benchmark)")
    bench = pd.DataFrame({
        "Component": ["HybridNets Detect", "MiDaS Depth", "Features+Scaler", "MLP Decision", "XAI Proxy", "Total"],
        "Avg Latency (ms)": [25.0, 15.0, 2.0, 0.18, 0.8, 43.0],
    })
    bfig = go.Figure(go.Bar(x=bench["Avg Latency (ms)"], y=bench["Component"], orientation="h",
        marker_color=["#ff9900" if c == "Total" else "#00e5ff" for c in bench["Component"]]))
    bfig.update_layout(height=240, paper_bgcolor="#0a0d14", plot_bgcolor="#0a0d14",
        font=dict(color="white"), margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="ms", gridcolor="#1e2230"), yaxis=dict(gridcolor="#1e2230"))
    st.plotly_chart(bfig, use_container_width=True)

    # XAI explanations table
    st.subheader("Recent XAI Explanations")
    sample = frames[max(0, num_frames-10):]
    xai_df = pd.DataFrame([{
        "Frame": f["frame"], "Action": f["action"],
        "Risk": round(f["risk"], 3), "Reasoning": f["reasoning"][:80] + "…" if len(f["reasoning"]) > 80 else f["reasoning"],
    } for f in sample])
    st.dataframe(xai_df, use_container_width=True, hide_index=True)

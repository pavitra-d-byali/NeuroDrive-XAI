"""
explainability/dash_app.py
==========================
NeuroDrive-XAI Debug Dashboard.

A production-grade Streamlit app to visualize:
  - Perception heatmaps (Grad-CAM)
  - Control feature importance (SHAP)
  - Decision reasoning (Natural Language)
  - Latency and Uncertainty metrics
"""

import streamlit as st
import pandas as pd
import json
import os
from PIL import Image
import numpy as np

def run_dashboard():
    st.set_page_config(page_title="NeuroDrive-XAI Dashboard", layout="wide")
    st.title("🧠 NeuroDrive-XAI | Integrated Reasoning Dashboard")

    # ── Sidebar Configuration ──
    st.sidebar.header("Run Configuration")
    log_file = st.sidebar.file_uploader("Upload XAI Logs (explanations.json)", type=["json"])
    frame_idx = st.sidebar.number_input("Frame Index", min_value=0, value=0)

    if log_file:
        data = json.load(log_file)
        if frame_idx < len(data):
            frame_data = data[frame_idx]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🖼️ Perception & Reasoning")
                # In a real app, we'd load the specific frame image or video
                st.info(f"**AI Reasoning:** {frame_data.get('reasoning', 'No reasoning available.')}")
                
                # Placeholder for frame visualization
                st.image("artifacts/debug_frame_0.jpg", caption=f"Frame {frame_idx} with Heatmap Overlay", use_column_width=True)

            with col2:
                st.subheader("🎮 Control XAI (SHAP)")
                decision = frame_data.get("decision", {})
                st.write(f"**Action:** `{decision.get('action', 'N/A')}`")
                st.write(f"**Risk Score:** `{decision.get('risk_score', 0.0)}`")
                
                # Mock SHAP plot data
                shap_data = pd.DataFrame({
                    "Feature": ["distance", "velocity", "lane_offset", "curvature"],
                    "Contribution": [0.45, -0.2, 0.05, 0.02]
                })
                st.bar_chart(shap_data.set_index("Feature"))

            # ── System Metrics ──
            st.divider()
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Latency", f"{frame_data.get('latency', 0)*1000:.1f} ms")
            m_col2.metric("Uncertainty", f"{frame_data.get('uncertainty', 0):.2f}")
            m_col3.metric("Mode", frame_data.get("commands", {}).get("mode", "PID"))

    else:
        st.warning("Please upload `explanations.json` from the `artifacts/` folder to begin.")
        st.markdown("""
        ### Instructions:
        1. Run the `main_pipeline.py` script.
        2. Locate `artifacts/explanations.json`.
        3. Drag and drop it here to see why the AI made each decision.
        """)

if __name__ == "__main__":
    run_dashboard()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import json

# Local modules
from physical_design import render_floorplan_analyzer, generate_netlist, simulate_strategies
from eda_infra import (
    init_session_state,
    render_license_monitor,
    render_bug_tracker,
    render_tool_registry,
    render_run_tracker,
    render_tapeout_checklist,
    render_database_manager,
    get_license_data,
)
from metrics import calculate_system_health_score

# --- Configuration & Initialization ---
st.set_page_config(page_title="CAD Command Center", layout="wide")

# Initialize Session State (must be called outside a function with st.cache_data)
init_session_state()

# --- Global Navigation & Sidebar ---
with st.sidebar:
    st.title("💻 CAD Command Center")
    st.markdown("---")

    view = st.radio(
        "Select Application",
        ["🔨 Floorplan Analyzer", "📊 EDA Infrastructure"],
        index=0
    )
    st.markdown("---")
    
    # SYSTEM HEALTH SCORE (Unified Metric)
    st.subheader("System Health Status")
    
    # Calculate health score (requires license data for infrastructure status)
    license_df = get_license_data()
    health_score, status_color, status_text = calculate_system_health_score(license_df)
    
    st.metric("Health Score (Lower is better)", f"{health_score:.1f}", delta=status_text)
    
    st.markdown(f"""
    <div style='padding: 10px; border-radius: 5px; background-color: {status_color}; color: white; text-align: center; font-weight: bold;'>
        OVERALL STATUS: {status_text.upper()}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Version Info")
    st.info(f"App Build: {datetime.now().strftime('%Y%m%d')}")
    st.caption("Doing more than expected is sometimes easier—when the goal is greater than the task. ")

# --- Main Page Content ---

if view == "🔨 Floorplan Analyzer":
    # --- Floorplan Analyzer Setup ---
    
    # 1. Inputs (Read from the widget's session state key defined in physical_design.py)
    # The default values are defined in eda_infra.py
    seed = st.session_state.get('fp_seed_display', 42)
    area = st.session_state.get('fp_area_display', 300.0)
    
    # 2. Netlist Generation & Simulation (Cached)
    # FIX: generate_netlist only returns 2 values: blocks_df and G.
    blocks_df, G = generate_netlist(seed)
    
    # blocks_df is updated inside simulate_strategies to include partition columns
    blocks_df, results_df = simulate_strategies(blocks_df, G, area)

    # 3. Render the Floorplan page
    # FIX: render_floorplan_analyzer now accepts the generated data as arguments.
    render_floorplan_analyzer(blocks_df, results_df, G)

elif view == "📊 EDA Infrastructure":
    # Render the EDA Infrastructure page (License Monitor, Bug Tracker, Tool Registry)
    
    # Use a sub-navigation for the infrastructure dashboard
    infra_view = st.radio(
        "Infrastructure View",
        [
            "📊 License Monitor",
            "🐞 Bug Tracker",
            "⚙️ Tool Registry",
            "🚀 Run Tracker",
            "✅ Tapeout Checklist",
            "🗄️ Database Manager",
        ],
        horizontal=True
    )
    
    st.markdown("---")
    
    if infra_view == "📊 License Monitor":
        render_license_monitor()
    elif infra_view == "🐞 Bug Tracker":
        render_bug_tracker()
    elif infra_view == "⚙️ Tool Registry":
        render_tool_registry()
    elif infra_view == "🚀 Run Tracker":
        render_run_tracker()
    elif infra_view == "✅ Tapeout Checklist":
        render_tapeout_checklist()
    elif infra_view == "🗄️ Database Manager":
        render_database_manager()

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import networkx as nx
from math import sqrt
from io import StringIO
import community.community_louvain as community_louvain

try:  # pragma: no cover - guarding for partially deployed environments
    import eda_infra  # type: ignore
except ModuleNotFoundError:
    eda_infra = None


def _fallback_tool_registry() -> pd.DataFrame:
    """Keeps the floorplan dashboard usable when the registry helper is missing."""
    st.warning(
        "Tool registry unavailable — falling back to empty registry. Check the eda_infra module deployment."
    )
    return pd.DataFrame(columns=["Project", "Tool", "Approved Version", "Compiler"])


if eda_infra and hasattr(eda_infra, "load_tool_registry"):
    load_tool_registry = eda_infra.load_tool_registry  # type: ignore[attr-defined]
else:
    load_tool_registry = _fallback_tool_registry

from metrics import calculate_advanced_metrics, generate_export_file, generate_floorplan_coords

# --- Configuration (Moved from main app) ---
NUM_BLOCKS = 60 
NET_COUNT = 180 
FLOORPLAN_PROJECT_NAME = "AI Accelerator" 
PIN_PER_BLOCK_MEAN = 6
NET_SIZE_MEAN = 4.5

# --- Netlist Generation ---
@st.cache_data
def generate_netlist(seed=42):
    np.random.seed(seed)
    
    blocks = pd.DataFrame({
        'block_id': [f'BLOCK_{i:02d}' for i in range(NUM_BLOCKS)],
        'area_mm2': np.abs(np.random.normal(5, 1.8, NUM_BLOCKS)),
        'pin_count': np.clip(np.random.normal(PIN_PER_BLOCK_MEAN, 2, NUM_BLOCKS), 2, 12).astype(int),
        'functional_group': np.random.choice(['Control', 'Datapath', 'Memory', 'IO'], NUM_BLOCKS),
        'max_delay_ps': np.random.uniform(50, 200, NUM_BLOCKS), 
        'switching_activity': np.random.uniform(0.1, 1.0, NUM_BLOCKS)
    })
    
    G = nx.Graph()
    G.add_nodes_from(blocks['block_id'])

    for i in range(NET_COUNT):
        net_size = int(np.clip(np.random.normal(NET_SIZE_MEAN, 1.5), 2, 7))
        connected_blocks = np.random.choice(blocks['block_id'], net_size, replace=False)
        
        for j in range(net_size):
            for k in range(j + 1, net_size):
                wire_delay = 10 * net_size 
                if G.has_edge(connected_blocks[j], connected_blocks[k]):
                    G[connected_blocks[j]][connected_blocks[k]]['weight'] += 1
                    G[connected_blocks[j]][connected_blocks[k]]['wire_delay'] = max(G[connected_blocks[j]][connected_blocks[k]]['wire_delay'], wire_delay)
                else:
                    G.add_edge(connected_blocks[j], connected_blocks[k], weight=1, wire_delay=wire_delay)

    return blocks, G

# --- Strategy Simulation ---
@st.cache_data
def simulate_strategies(blocks_df, _G, area_budget): 
    blocks_df = blocks_df.copy()
    results = []
    
    # Strategy 1: Flat
    blocks_df['P_Min'] = 0
    results.append({
        'Strategy': '1. Flat (All in One)', 'Partition_Col': 'P_Min',
        **calculate_advanced_metrics(blocks_df, _G, 'P_Min', area_budget, blocks_df['max_delay_ps'])
    })
    
    # Strategy 2: Functional
    blocks_df['P_Functional'] = blocks_df['functional_group']
    results.append({
        'Strategy': '2. Functional Grouping', 'Partition_Col': 'P_Functional',
        **calculate_advanced_metrics(blocks_df, _G, 'P_Functional', area_budget, blocks_df['max_delay_ps'])
    })
    
    # Strategy 3: Random
    np.random.seed(1)
    blocks_df['P_Random'] = np.random.randint(0, 6, NUM_BLOCKS)
    results.append({
        'Strategy': '3. Random Grouping (6 Modules)', 'Partition_Col': 'P_Random',
        **calculate_advanced_metrics(blocks_df, _G, 'P_Random', area_budget, blocks_df['max_delay_ps'])
    })
    
    # Strategy 4: Connectivity-Driven
    try:
        partition_map = community_louvain.best_partition(_G)
        blocks_df['P_Louvain'] = blocks_df['block_id'].map(partition_map).fillna(0).astype(int)
    except Exception:
        blocks_df['P_Louvain'] = blocks_df['block_id'].apply(lambda x: int(x.split('_')[1]) % 5).astype(int)
        
    results.append({
        'Strategy': '4. Connectivity-Driven (Louvain)', 'Partition_Col': 'P_Louvain',
        **calculate_advanced_metrics(blocks_df, _G, 'P_Louvain', area_budget, blocks_df['max_delay_ps'])
    })
    
    return blocks_df, pd.DataFrame(results)

# --- Renderer Function (Called by app.py) ---

def render_floorplan_analyzer(blocks_df, results_df, G):
    """
    Renders the Floorplan Optimization Tool UI and results.
    Accepts pre-computed blocks, results, and graph (G) from the main app for caching efficiency.
    """
    
    st.title("🧱 Floorplan Strategy Analyzer — Timing, Power & Routability")
    st.caption("Advanced physical design trade-off exploration for ASIC/FPGA floorplanning.")

    # Fetch Approved Tool Version from Registry for the current project
    registry_df = load_tool_registry()
    
    # Check if registry_df is empty before trying to access iloc[0]
    filtered_registry = registry_df[registry_df['Project'] == FLOORPLAN_PROJECT_NAME]
    
    if not filtered_registry.empty:
        approved_tool = filtered_registry.iloc[0]
        st.subheader(f"Project: {FLOORPLAN_PROJECT_NAME}")
        st.markdown(f"**CAD Compliance:** Approved P&R Tool: **{approved_tool['Tool']} v{approved_tool['Approved Version']}**")
    else:
        st.subheader(f"Project: {FLOORPLAN_PROJECT_NAME}")
        st.warning("CAD Compliance: Approved tool information not found in registry.")

    # --- Sidebar Inputs ---
    # These inputs drive the simulation that happened in streamlit_app.py
    with st.sidebar:
        st.header("🔧 Design Inputs")
        # Display inputs that affect the cached data, using unique keys for the sidebar widgets
        seed = st.number_input("Netlist Random Seed", value=st.session_state.get('fp_seed_display', 42), step=1, key="fp_seed_display")
        floorplan_area_mm2 = st.slider("Total Floorplan Area Budget (mm²)", 120.0, 600.0, st.session_state.get('fp_area_display', 300.0), step=20.0, key="fp_area_display")
        
        st.markdown("---")
        st.subheader("🔍 Compare Strategies")
        compare_strat = st.selectbox("Select Strategy to Inspect", 
                                     ["4. Connectivity-Driven (Louvain)", "2. Functional Grouping", 
                                      "1. Flat (All in One)", "3. Random Grouping (6 Modules)"], key="fp_strat")
        
        st.markdown("---")
        st.subheader("ℹ️ Metrics Guide")
        st.markdown("""
        - **Critical Path**: Intrinsic + wire delay (ps). **Must be minimized.**
        - **Routability**: >1.0 = high congestion risk (due to local block density).
        - **Power Proxy**: Proxy for switching power on cross-module nets.
        """)
        
    # --- Info & Warnings ---
    total_block_area = blocks_df['area_mm2'].sum() 
    st.info(f"Total block area: **{total_block_area:.1f} mm²** | Budget: **{floorplan_area_mm2:.1f} mm²**")
    if total_block_area > floorplan_area_mm2:
        st.warning("⚠️ Area budget insufficient — utilization > 100%")

    louvain_partitions = blocks_df['P_Louvain'].nunique()
    if louvain_partitions <= 1:
        st.warning("💡 Connectivity-Driven used fallback (install `python-louvain` for optimal results).")


    # --- Main Visualization: Multi-Metric Radar Chart ---
    st.header("📊 Multi-Objective Strategy Comparison")
    
    metrics = ['critical_path_proxy_ps', 'switching_power_proxy', 'utilization_pct', 'routability_score']
    normalized = results_df[metrics].copy()
    for col in metrics:
        min_val = normalized[col].min()
        max_val = normalized[col].max()
        normalized[col] = (normalized[col] - min_val) / (max_val - min_val + 1e-5)

    radar_data = []
    for i, row in results_df.iterrows():
        for metric in metrics:
            radar_data.append({
                'Strategy': row['Strategy'],
                'Metric': metric.replace('_proxy_ps', ' (Timing)').replace('_proxy', ' (Power)').replace('utilization_pct', 'Area Utilization').replace('_', ' ').title(),
                'Value': normalized.iloc[i][metric]
            })

    radar_df = pd.DataFrame(radar_data)
    fig_radar = px.line_polar(
        radar_df, r='Value', theta='Metric', color='Strategy', line_close=True,
        title="Normalized Strategy Performance (Closer to Center is Better)", height=550
    )
    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar, use_container_width=True)

    # --- Detailed Table & Export ---
    st.subheader("📋 Full Metrics Dashboard")
    display_df = results_df[[
        'Strategy', 'critical_path_proxy_ps', 'switching_power_proxy',
        'routability_score', 'utilization_pct', 'total_wire_cost'
    ]].round(2)

    display_df.columns = [
        'Strategy', 'Critical Path (ps)', 'Power Proxy', 'Routability Score', 'Utilization (%)', 'Wire Cost'
    ]
    st.dataframe(display_df, hide_index=True)

    col_exp_1, col_exp_2 = st.columns([1, 1])

    with col_exp_1:
        csv = display_df.to_csv(index=False)
        st.download_button(label="📥 Export Metrics to CSV", data=csv, file_name="floorplan_metrics.csv", mime="text/csv")

    # --- Strategy Deep Dive ---
    st.header(f"🔍 Deep Dive: {compare_strat}")
    selected_row = results_df[results_df['Strategy'] == compare_strat].iloc[0]
    partition_col = selected_row['Partition_Col']

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Critical Path", f"{selected_row['critical_path_proxy_ps']:.1f} ps")
        st.metric("Total Wire Cost", f"{selected_row['total_wire_cost']:.0f} nets")
    with col2:
        st.metric("Power Proxy", f"{selected_row['switching_power_proxy']:.2f}")
        routability_status = "High Risk" if selected_row['routability_score'] > 1.0 else "Acceptable"
        st.metric("Routability", routability_status, delta=f"{selected_row['routability_score']:.2f}")


    # --- Physical Visualization ---
    st.markdown("---")
    st.header("🔬 Physical Floorplan Visualization & Connectivity")

    plot_df, module_boundaries = generate_floorplan_coords(blocks_df, partition_col, floorplan_area_mm2)
    floorplan_side = sqrt(floorplan_area_mm2)

    with col_exp_2:
        def_content = generate_export_file(plot_df, floorplan_area_mm2, partition_col, compare_strat)
        st.download_button(label="💾 Export Floorplan (TCL/DEF)", data=def_content, file_name=f"floorplan_{compare_strat.replace(' ', '_').lower()}.def", mime="text/plain")

    col_floorplan, col_graph = st.columns([1, 1])

    with col_floorplan:
        st.subheader("Physical Floorplan View (mm)")
        
        # Ensure plot_df has the correct column for color mapping
        plot_df['Module'] = plot_df[partition_col].apply(lambda x: str(x))

        fig_fp = px.scatter(
            plot_df, x='x_coord', y='y_coord', color='Module', size='area_mm2', 
            hover_data={'block_id': True, 'area_mm2': ':.1f', 'Module': False, 'functional_group': True},
            title=f"Block Placement ({compare_strat})", labels={'x_coord': 'X (mm)', 'y_coord': 'Y (mm)'},
            color_discrete_map={str(module_id): bounds['color'] for module_id, bounds in module_boundaries.items()}
        )
        
        for module_id, bounds in module_boundaries.items():
            fig_fp.add_shape(type="rect", x0=bounds['x_min'], y0=bounds['y_min'], x1=bounds['x_max'], y1=bounds['y_max'],
                line=dict(color=bounds['color'], width=2, dash="dash"), fillcolor="rgba(0,0,0,0)")
            fig_fp.add_annotation(x=(bounds['x_min'] + bounds['x_max']) / 2, y=(bounds['y_min'] + bounds['y_max']) / 2,
                text=f"Module {module_id}", showarrow=False, font=dict(size=12, color=bounds['color']))

        fig_fp.update_layout(xaxis=dict(range=[0, floorplan_side], scaleanchor="y", scaleratio=1), yaxis=dict(range=[0, floorplan_side]), showlegend=False)
        st.plotly_chart(fig_fp, use_container_width=True)

    with col_graph:
        st.subheader("Inter-Module Connectivity")
        current_blocks_df = blocks_df[['block_id', partition_col, 'area_mm2']].rename(columns={partition_col: 'Module'})
        module_G = nx.Graph()
        block_map = current_blocks_df.set_index('block_id')['Module'].to_dict()

        for u, v, data in G.edges(data=True):
            p_u = block_map.get(u); p_v = block_map.get(v)
            # Use string representation for comparison to handle mixed int/string partitions (Functional vs others)
            p_u_str = str(p_u); p_v_str = str(p_v)
            
            if p_u is not None and p_v is not None and p_u_str != p_v_str:
                weight = data.get('weight', 1)
                
                # Use original partition keys for module_G nodes
                if module_G.has_edge(p_u, p_v): module_G[p_u][p_v]['weight'] += weight
                else: module_G.add_edge(p_u, p_v, weight=weight)
                
        pos = nx.spring_layout(module_G, seed=42, k=0.5) 
        edge_x = []; edge_y = []
        for edge in module_G.edges(data=True):
            x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

        trace_edges = dict(type='scatter', x=edge_x, y=edge_y, mode='lines', line=dict(width=1.5, color='#888'), hoverinfo='none')
        node_x = [pos[k][0] for k in module_G.nodes()]; node_y = [pos[k][1] for k in module_G.nodes()]
        
        group_ids = current_blocks_df['Module'].unique()
        group_colors = {gid: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, gid in enumerate(group_ids)}

        node_info = []
        for node in module_G.nodes():
            module_blocks_df = blocks_df[blocks_df[partition_col] == node]
            area = module_blocks_df['area_mm2'].sum(); count = module_blocks_df.shape[0]
            most_common_func = module_blocks_df['functional_group'].mode()
            func_label = most_common_func.iloc[0] if not most_common_func.empty else "Mixed"
            node_info.append(f'Module: {node}<br>Blocks: {count}<br>Area: {area:.2f} mm² ({func_label})')
            
        trace_nodes = dict(type='scatter', x=node_x, y=node_y, mode='markers+text', hoverinfo='text',
            text=[f'M{k}' for k in module_G.nodes()], textposition="top center",
            marker=dict(showscale=False, color=[group_colors.get(node, '#333') for node in module_G.nodes()], size=30, line=dict(width=2, color='DarkSlateGrey')),
            textfont=dict(size=12, color='DarkSlateGrey'), hovertext=node_info)

        layout = dict(showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        
        fig_conn = dict(data=[trace_edges, trace_nodes], layout=layout)
        st.plotly_chart(fig_conn, use_container_width=True)

    st.markdown("---")
    st.subheader("💡 CAD Insights & Trade-off Conclusion")
    st.markdown("""
    This analysis helps the P&R engineer select the optimal **hierarchical grouping** before starting detailed placement.
    The Connectivity-Driven approach (Louvain) is typically the best starting point for performance-critical designs.
    """)

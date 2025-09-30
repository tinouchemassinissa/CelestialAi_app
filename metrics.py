import numpy as np
import pandas as pd
from math import sqrt 
from io import StringIO
import networkx as nx
import plotly.express as px 

# --- Configuration (Needed for DEF Export) ---
FLOORPLAN_PROJECT_NAME = "AI Accelerator" 

# --- Core Metric Calculation ---

def calculate_advanced_metrics(blocks_df, G: nx.Graph, partition_column, floorplan_area_mm2, max_delay_series):
    """Calculates all Timing, Power, Area, and Routability metrics for a given partition."""
    
    total_wire_cost = 0; total_wire_delay = 0; total_switching_power = 0
    block_sw_map = blocks_df.set_index('block_id')['switching_activity'].to_dict()
    
    # 1. Wire Cost, Delay, and Power (Inter-partition communication)
    for u, v, data in G.edges(data=True):
        try:
            part_u = blocks_df.set_index('block_id').loc[u, partition_column]
            part_v = blocks_df.set_index('block_id').loc[v, partition_column]
        except KeyError: continue
        
        weight = data.get('weight', 1); wire_delay = data.get('wire_delay', 10)
        
        # NOTE: Using str() for comparison to handle mixed data types (int for P_Louvain, string for P_Functional)
        if str(part_u) != str(part_v):
            total_wire_cost += weight
            total_wire_delay += weight * wire_delay
            avg_sw = (block_sw_map.get(u, 0.5) + block_sw_map.get(v, 0.5)) / 2
            total_switching_power += weight * avg_sw

    # 2. Area & Utilization
    partition_areas = blocks_df.groupby(partition_column)['area_mm2'].sum()
    total_used_area = partition_areas.sum()
    utilization_pct = (total_used_area / floorplan_area_mm2) * 100
    
    max_partition_area = partition_areas.max() if not partition_areas.empty else 0.0
    max_blocks_in_partition = blocks_df.groupby(partition_column).size().max() if not partition_areas.empty else 0

    # 3. Routability Score (Congestion Proxy)
    ideal_blocks_per_mm2 = 2.0  
    if max_partition_area > 0:
        actual_density = max_blocks_in_partition / max_partition_area
        # Routability is capped at 3.0 for better visualization/scaling
        routability_score = min(actual_density / ideal_blocks_per_mm2, 3.0) 
    else:
        routability_score = 1.0
        
    # 4. Critical Path Proxy (Timing)
    max_intrinsic = max_delay_series.max() # Max intrinsic block delay
    # Average wire delay penalty for crossing a boundary
    avg_cross_wire_delay = (total_wire_delay / max(1, total_wire_cost)) if total_wire_cost > 0 else 0
    critical_path_proxy = max_intrinsic + avg_cross_wire_delay

    return {
        'total_wire_cost': total_wire_cost,
        'critical_path_proxy_ps': critical_path_proxy,
        'switching_power_proxy': total_switching_power,
        'utilization_pct': utilization_pct,
        'max_partition_area': max_partition_area,
        'routability_score': routability_score 
    }

def calculate_system_health_score(license_df: pd.DataFrame):
    """Calculates a unified system health score based on EDA infrastructure status."""
    
    # 1. License Shortage Penalty
    shortage_count = sum(license_df['Available'] == 0)
    license_penalty = shortage_count * 5.0 # High penalty for critical shortages

    # 2. Utilization Penalty
    # Penalty for tools with very high utilization (risk of future shortages)
    high_util_count = sum(license_df['Utilization (%)'] > 90)
    utilization_penalty = high_util_count * 2.0 
    
    # 3. Bug Penalty (Assuming this data is used elsewhere, but adding a proxy here)
    # The number of 'Open' bugs in the registry would typically be used, but we don't have that data in this function.
    # We will use a fixed small penalty for demonstration.
    bug_penalty_proxy = 1.0 # Proxy for general bug risk

    health_score = license_penalty + utilization_penalty + bug_penalty_proxy

    # Determine status text and color
    if health_score >= 10:
        status_text = "Critical Risk"
        status_color = "#e74c3c" # Red
    elif health_score >= 5:
        status_text = "High Alert"
        status_color = "#f39c12" # Orange
    else:
        status_text = "Optimal"
        status_color = "#2ecc71" # Green
        
    return health_score, status_color, status_text


# --- Physical Coordinates for Visualization ---

def generate_floorplan_coords(blocks_df, partition_col, total_area):
    """Generates 2D coordinates for visual block placement within calculated module boundaries."""
    
    floorplan_side = sqrt(total_area)
    module_areas = blocks_df.groupby(partition_col)['area_mm2'].sum().reset_index()
    module_areas['area_ratio'] = module_areas['area_mm2'] / module_areas['area_mm2'].sum()
    current_x = 0; module_boundaries = {}
    
    # Module boundaries (simple sequential packing)
    for i, row in module_areas.iterrows():
        module_id = row[partition_col]
        module_width = floorplan_side * row['area_ratio'] 
        module_boundaries[module_id] = {
            'x_min': current_x, 'x_max': current_x + module_width,
            'y_min': 0, 'y_max': floorplan_side,
            'color': px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        }
        current_x += module_width

    # Block placement within modules (random scatter for visualization)
    plot_df = blocks_df.copy()
    
    for module_id, bounds in module_boundaries.items():
        # Ensure we filter correctly, handling potential mixed types if partition_col is 'functional_group'
        module_blocks = plot_df[plot_df[partition_col] == module_id].index
        count = len(module_blocks)
        
        if count > 0:
            # Use random placement within bounds for visualization purposes
            plot_df.loc[module_blocks, 'x_coord'] = np.random.uniform(bounds['x_min'] + 0.05*floorplan_side, bounds['x_max'] - 0.05*floorplan_side, count)
            plot_df.loc[module_blocks, 'y_coord'] = np.random.uniform(bounds['y_min'] + 0.05*floorplan_side, bounds['y_max'] - 0.05*floorplan_side, count)
            plot_df.loc[module_blocks, 'Module'] = module_id # Use 'Module' for plotting color
            
    return plot_df.dropna(subset=['x_coord', 'y_coord']), module_boundaries

# --- Export File Generation ---

def generate_export_file(plot_df, total_area, partition_col, compare_strat):
    """Generates a simplified DEF-like (Design Exchange Format) content for tool handoff."""
    
    output = StringIO()
    floorplan_side = int(sqrt(total_area) * 1000) # Dimensions in microns
    
    output.write(f"# Design Exchange Format (DEF) - Floorplan Export\n")
    output.write(f"# Strategy: {compare_strat}\n")
    output.write(f"VERSION 5.8 ;\n")
    output.write(f"DIVIDERCHAR \"/\" ;\n")
    output.write("UNITS DISTANCE MICRONS 1000 ;\n")
    output.write(f"DESIGN {FLOORPLAN_PROJECT_NAME} ;\n")
    output.write("DIEAREA ( 0 0 ) ( {floorplan_side} {floorplan_side} ) ;\n\n".format(floorplan_side=floorplan_side))
    output.write("COMPONENTS {} ;\n".format(len(plot_df)))
    
    for _, row in plot_df.iterrows():
        x_center_um = int(row['x_coord'] * 1000)
        y_center_um = int(row['y_coord'] * 1000)
        
        # Fixed placement command is common in DEF for floorplanned macros
        output.write(f"- {row['block_id']} + FIXED ( {x_center_um} {y_center_um} ) N\n")
        output.write(f"  + PROPERTY MODULE_ID {row[partition_col]}\n")
        
    output.write("END COMPONENTS\n")
    return output.getvalue()

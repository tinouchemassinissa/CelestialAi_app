import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os

# --- Configuration & Data Loading ---
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def load_data(filename):
    """Loads CSV files from the data directory."""
    path = os.path.join(DATA_DIR, filename)
    try:
        # Attempt to load the user's local CSV
        df = pd.read_csv(path)
        return df if not df.empty else pd.DataFrame()
    except FileNotFoundError:
        # Provide sensible default if data dir not accessible (e.g., in canvas)
        if 'license_costs.csv' in filename:
            return pd.DataFrame({'Tool': ['Synopsys DC', 'Cadence Innovus', 'Siemens Calibre'], 
                                 'Cost_Per_Seat_USD': [25000, 35000, 15000]})
        if 'known_bugs.csv' in filename:
            return pd.DataFrame([
                {"ID": "BUG-001", "Tool": "Cadence Innovus", "Version": "22.1", "Issue": "Crashes during MMMC analysis", "Workaround": "Use -legacy_mode", "Reported By": "Alice", "Status": "Open"},
                {"ID": "BUG-002", "Tool": "Synopsys ICC2", "Version": "2023.03", "Issue": "Incorrect congestion map", "Workaround": "Run with -fix_congestion_patch", "Reported By": "Bob", "Status": "Fixed in 2023.06"}
            ])
        return pd.DataFrame()

def init_session_state():
    """Initializes session state with default data on first run."""
    if 'license_data' not in st.session_state:
        # Define a robust fallback in case the CSV load fails/is malformed
        FALLBACK_LICENSE_COSTS = pd.DataFrame({
            'Tool': ['Synopsys DC', 'Cadence Innovus', 'Siemens Calibre'], 
            'Cost_Per_Seat_USD': [25000, 35000, 15000]
        })
        
        # Load license costs from file
        license_costs_csv = load_data('license_costs.csv')
        
        # **ROBUSTNESS FIX**: Check for integrity. If 'Tool' column is missing, use the fallback.
        if 'Tool' not in license_costs_csv.columns:
            # If the CSV exists but is malformed, use the safe, hardcoded list
            license_costs_df = FALLBACK_LICENSE_COSTS
            st.warning("License cost data loaded is missing the 'Tool' column. Using synthetic fallback data for initialization.")
        else:
            license_costs_df = license_costs_csv

        # license_costs_df is now guaranteed to have the 'Tool' column
        tools = license_costs_df['Tool'].str.lower().str.replace(' ', '_').tolist()
        total_licenses = {t.replace('_', ' ').title(): 10 for t in tools}
        
        np.random.seed(42)
        data = []
        for tool in tools:
            tool_title = tool.replace("_", " ").title()
            total = total_licenses.get(tool_title, 10)
            used = np.random.randint(0, total + 1)
            
            data.append({
                "Tool": tool_title,
                "Vendor": "Synopsys" if "synopsys" in tool else ("Cadence" if "cadence" in tool else "Siemens"),
                "Total Licenses": total,
                "Used Licenses": used,
                "Available": total - used,
                "Utilization (%)": round(used / total * 100, 1)
            })
        
        st.session_state.license_data = pd.DataFrame(data)
        st.session_state.bug_registry = load_data('known_bugs.csv')
        st.session_state.tool_registry = pd.DataFrame([
            {"Project": "AI Accelerator", "Tool": "Cadence Innovus", "Approved Version": "23.1", "Compiler": "GCC 9.4"},
            {"Project": "IoT Sensor", "Tool": "Synopsys DC", "Approved Version": "2023.06", "Compiler": "GCC 8.5"},
            {"Project": "CPU Core", "Tool": "Synopsys PT", "Approved Version": "2022.12-SP3", "Compiler": "Clang 12"}
        ])
        
        # Floorplan defaults for cross-page persistence
        if 'fp_seed_display' not in st.session_state:
            st.session_state.fp_seed_display = 42
            st.session_state.fp_area_display = 300.0

def load_tool_registry():
    """Retrieves the current tool registry DataFrame from session state for use in physical_design."""
    if 'tool_registry' not in st.session_state:
        init_session_state()
    return st.session_state.tool_registry


def get_license_data():
    """Retrieves the current, dynamic license DataFrame from session state for use in metrics calculation."""
    if 'license_data' not in st.session_state:
        init_session_state()
    return st.session_state.license_data

# --- Rendering Functions ---

def render_license_monitor():
    st.header("Real-Time License Utilization")
    st.markdown("""
    > EDA licenses are high-value assets. Monitor usage, cost, and availability to ensure smooth design flow.
    """)
    
    df = get_license_data()
    if df.empty:
        st.warning("No license data available.")
        return

    # Load costs for cost analysis
    # Use try/except to handle the case where 'Tool' is missing from the loaded cost_df.
    try:
        cost_df = load_data('license_costs.csv').set_index('Tool')
    except KeyError:
        cost_df = pd.DataFrame() # Fallback to an empty DataFrame if 'Tool' is missing from the file.

    df = df.join(cost_df, on='Tool', how='left', rsuffix='_cost_join') # Join license_data with loaded cost data
    
    # If the join was successful (i.e., 'Cost_Per_Seat_USD' exists from the loaded CSV or fallback)
    if 'Cost_Per_Seat_USD' not in df.columns:
        # Fallback for if the join didn't work (e.g. if the original CSV was missing data)
        df['Cost_Per_Seat_USD'] = df['Tool'].apply(
            lambda x: 25000 if 'Synopsys' in x else (35000 if 'Cadence' in x else 15000)
        )

    df['Total Cost'] = df['Total Licenses'] * df['Cost_Per_Seat_USD'].fillna(0) 
    df['Unused Cost'] = df['Available'] * df['Cost_Per_Seat_USD'].fillna(0)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tools", len(df))
    col2.metric("Avg Utilization", f"{df['Utilization (%)'].mean():.1f}%")
    
    critical_shortages = sum(df['Available'] == 0)
    col3.metric("Critical Shortages", critical_shortages, delta_color="inverse")
    
    unused_cost = df['Unused Cost'].sum()
    col4.metric("Unused License Cost (Annual)", f"${unused_cost:,.0f}", help="Estimated annual cost of currently available (unused) seats.")

    # Chart
    fig = px.bar(
        df, 
        x="Tool", 
        y=["Used Licenses", "Available"],
        title="License Allocation by Tool",
        color_discrete_sequence=["#e74c3c", "#2ecc71"]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Table with conditional formatting
    st.subheader("License Details")
    
    # Select subset of columns to display
    display_cols = [col for col in ['Tool', 'Vendor', 'Total Licenses', 'Used Licenses', 'Available', 'Utilization (%)', 'Cost_Per_Seat_USD', 'Total Cost', 'Unused Cost'] if col in df.columns]

    # FIX: Apply styling and formatting to the sliced DataFrame and save the Styler object.
    # The Styler object itself is not subscriptable.
    styled_df = df[display_cols].style.applymap(
        # Use a high-contrast color for utilization warning (dark red on dark theme)
        lambda x: 'background-color: #a33333; color: white' if x >= 90 else '', 
        subset=['Utilization (%)']
    ).applymap(
        # Use a light gold/yellow color to highlight the largest license pool
        lambda x: 'background-color: #ffdb58; color: black' if x == df['Total Licenses'].max() and x > 0 else '', 
        subset=['Total Licenses']
    ).format({
        'Cost_Per_Seat_USD': '${:,.0f}',
        'Total Cost': '${:,.0f}',
        'Unused Cost': '${:,.0f}'
    })
    
    # Pass the resulting Styler object directly to st.dataframe
    st.dataframe(styled_df, hide_index=True)
    
    # --- Add New Tool Form (Dynamic User Input) ---
    with st.expander("➕ Add New EDA Tool"):
        with st.form("new_tool_form"):
            st.subheader("Input New Tool Specifications")
            
            tool_name = st.text_input("Tool Name (e.g., Cadence Genus)", key='new_tool_name')
            vendor = st.selectbox("Vendor", ["Synopsys", "Cadence", "Siemens", "Other"], key='new_tool_vendor')
            total_licenses = st.number_input("Total Licenses Owned", min_value=1, step=1, key='new_tool_total', value=5)
            used_licenses = st.number_input("Currently Used Licenses", min_value=0, max_value=total_licenses, step=1, key='new_tool_used', value=1)
            cost_per_seat = st.number_input("Annual Cost Per Seat (USD)", min_value=0, step=1000, key='new_tool_cost', value=10000)

            submitted = st.form_submit_button("Add Tool to Dashboard")
            
            if submitted and tool_name:
                new_data = {
                    "Tool": tool_name.title(),
                    "Vendor": vendor,
                    "Total Licenses": total_licenses,
                    "Used Licenses": used_licenses,
                    "Available": total_licenses - used_licenses,
                    "Utilization (%)": round(used_licenses / total_licenses * 100, 1),
                    "Cost_Per_Seat_USD": cost_per_seat,
                    "Total Cost": total_licenses * cost_per_seat,
                    "Unused Cost": (total_licenses - used_licenses) * cost_per_seat
                }
                
                # Append to session state
                current_df = st.session_state.license_data
                if tool_name.title() not in current_df['Tool'].tolist():
                    updated_df = pd.concat([current_df, pd.DataFrame([new_data])], ignore_index=True)
                    st.session_state.license_data = updated_df
                    st.success(f"Tool '{tool_name.title()}' added and dashboard updated!")
                    st.rerun()
                else:
                    st.warning(f"Tool '{tool_name.title()}' already exists.")


def render_bug_tracker():
    st.header("🐞 Known EDA Tool Issues & Workarounds")
    st.markdown("""
    > Central repository for tool bugs, workarounds, and status—reducing redundant debugging across teams. (Data persists within this session.)
    """)
    
    bugs_df = st.session_state.bug_registry
    
    # FIX: Check if the 'Status' column exists before trying to access it for filtering.
    if "Status" in bugs_df.columns and not bugs_df.empty:
        status_options = ["All"] + list(bugs_df["Status"].unique())
    else:
        # Fallback to standard status options if the column is missing or DataFrame is empty
        status_options = ["All", "Open", "Fixed", "Under Review"]
        
    status_filter = st.selectbox("Filter by Status", status_options)
    
    if status_filter != "All":
        # Check again if the column exists before filtering
        if "Status" in bugs_df.columns:
            bugs_df = bugs_df[bugs_df["Status"] == status_filter]
        else:
            st.warning(f"Cannot filter by status '{status_filter}' because the 'Status' column is missing from the bug registry data.")

    # Display bugs
    for _, bug in bugs_df.iterrows():
        # Check for 'Status' key before using it
        status = bug.get('Status', 'Unknown')
        status_emoji = "🔴" if status == "Open" else ("🟠" if status == "Under Review" else "🟢")
        
        with st.expander(f"{status_emoji} {bug['ID']} - {bug['Tool']} v{bug['Version']}"):
            st.markdown(f"**Issue**: {bug['Issue']}")
            st.markdown(f"**Workaround**: `{bug.get('Workaround', 'None yet.')}`")
            st.markdown(f"**Reported By**: {bug.get('Reported By', 'N/A')} | **Status**: `{status}`")
    
    # Add new bug (now uses session state for persistence)
    with st.form("new_bug"):
        st.subheader("➕ Report New Issue")
        tool = st.text_input("Tool (e.g., Cadence Innovus)")
        version = st.text_input("Version")
        issue = st.text_area("Issue Description")
        workaround = st.text_input("Workaround (if known)")
        reported_by = st.text_input("Reported By (Your Name)", value="User")
        
        submitted = st.form_submit_button("Submit Bug Report")
        if submitted and tool and issue:
            new_id = f"BUG-{len(st.session_state.bug_registry) + 1:03d}"
            new_bug = {
                "ID": new_id,
                "Tool": tool.title(),
                "Version": version,
                "Issue": issue,
                "Workaround": workaround if workaround else "None yet.",
                "Reported By": reported_by,
                "Status": "Open"
            }
            
            # Append to session state
            new_bugs_df = pd.concat([st.session_state.bug_registry, pd.DataFrame([new_bug])], ignore_index=True)
            st.session_state.bug_registry = new_bugs_df
            st.success(f"Bug {new_id} reported successfully!")
            st.rerun()


def render_tool_registry():
    st.header("⚙️ Tool Version Registry")
    st.markdown("""
    > Ensures design reproducibility by tracking approved, stable tool versions per project.
    """)
    
    registry = st.session_state.tool_registry
    st.dataframe(registry, hide_index=True)
    
    st.markdown("### Why This Matters for Design Integrity")
    st.markdown("""
    - **Prevents Version Drift:** Ensures all teams are using the exact same, verified executables.
    - **Reproducible Builds:** Essential for achieving consistent results between design iterations, especially for timing closure.
    - **CI/CD Integration:** The registry acts as the source of truth for automated build pipelines.
    """)

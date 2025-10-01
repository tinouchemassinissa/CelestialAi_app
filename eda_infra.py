def render_license_monitor():
    st.header("Real-Time License Utilization")
    st.markdown("""
    > EDA licenses are high-value assets. Monitor usage, cost, and availability to ensure smooth design flow.
    """)

    df = get_license_data()
    if df.empty:
        st.warning("No license data available.")
        return

    # Load costs for cost analysis (robust to malformed/missing CSV)
    try:
        cost_df = load_data('license_costs.csv')
    except KeyError:
        cost_df = pd.DataFrame()

    if not cost_df.empty and 'Tool' in cost_df.columns:
        cost_df = cost_df.set_index('Tool')
        df = df.join(cost_df, on='Tool', how='left', rsuffix='_cost_join')
    else:
        df = df.copy()

    # Harmonize columns introduced by the join, preferring live dashboard values
    for col in ['Vendor', 'Cost_Per_Seat_USD', 'Total Licenses']:
        join_col = f"{col}_cost_join"
        if join_col in df.columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[join_col])
            else:
                df[col] = df[join_col]
            df.drop(columns=[join_col], inplace=True)

    # Ensure costs exist even if CSV lacked them
    if 'Cost_Per_Seat_USD' not in df.columns:
        df['Cost_Per_Seat_USD'] = df['Tool'].apply(
            lambda x: 25000 if 'Synopsys' in x else (35000 if 'Cadence' in x else 15000)
        )

    df['Total Cost'] = df['Total Licenses'] * df['Cost_Per_Seat_USD'].fillna(0)
    df['Unused Cost'] = df['Available'] * df['Cost_Per_Seat_USD'].fillna(0)

    # --- Interactive Filters ---
    with st.expander("🔍 Focus the Dashboard", expanded=True):
        vendor_values = df['Vendor'].fillna('Unknown').unique().tolist()
        vendor_values.sort()
        selected_vendors = st.multiselect(
            "Filter by Vendor", vendor_values, default=vendor_values
        )
        max_util = float(df['Utilization (%)'].max()) if not df['Utilization (%)'].empty else 100.0
        slider_upper = max(5.0, max_util)
        min_utilization = st.slider("Minimum Utilization (%)", 0.0, slider_upper, 0.0, step=5.0)

    filtered_df = df[df['Vendor'].fillna('Unknown').isin(selected_vendors)].copy()
    filtered_df = filtered_df[filtered_df['Utilization (%)'] >= min_utilization]

    # Record a snapshot only when the visible numbers actually change
    snapshot_signature = None
    if not filtered_df.empty:
        snapshot_signature = (
            len(filtered_df),
            int(filtered_df['Total Licenses'].sum()),
            int(filtered_df['Used Licenses'].sum()),
            round(float(filtered_df['Utilization (%)'].mean()), 2),
            round(float(filtered_df['Total Cost'].sum()), 2) if 'Total Cost' in filtered_df.columns else 0.0,
            round(float(filtered_df['Unused Cost'].sum()), 2) if 'Unused Cost' in filtered_df.columns else 0.0,
        )
        if st.session_state.get('last_snapshot_signature') != snapshot_signature:
            record_license_snapshot(filtered_df)
            st.session_state['last_snapshot_signature'] = snapshot_signature

    if filtered_df.empty:
        st.warning("No tools match the current filters. Adjust the vendor selection or utilization threshold.")
        return

    # Spend callouts
    total_cost_series = filtered_df['Total Cost'].fillna(0)
    unused_cost_series = filtered_df['Unused Cost'].fillna(0)
    total_unused_cost = float(unused_cost_series.sum())
    total_used_cost = float(total_cost_series.sum() - total_unused_cost)
    st.info(
        "\n".join(
            [
                f"**Active Spend:** ${total_used_cost:,.0f}",
                f"**Idle Spend Exposure:** ${total_unused_cost:,.0f}",
                "Balance procurement plans against these figures to keep quarterly CAD budgets on track.",
            ]
        )
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tools Shown", len(filtered_df))
    col2.metric("Avg Utilization", f"{filtered_df['Utilization (%)'].mean():.1f}%")
    critical_shortages = int((filtered_df['Available'] == 0).sum())
    col3.metric("Critical Shortages", critical_shortages, delta_color="inverse")

    top_idle_tool = filtered_df.sort_values('Unused Cost', ascending=False).iloc[0]
    col4.metric(
        "Highest Idle Spend",
        f"${top_idle_tool['Unused Cost']:,.0f}",
        help=f"{top_idle_tool['Tool']} is carrying the largest unused license cost."
    )

    # Vendor summary + pie
    vendor_summary = (
        filtered_df.groupby('Vendor', dropna=False)
        .agg({
            'Tool': 'count',
            'Total Licenses': 'sum',
            'Used Licenses': 'sum',
            'Available': 'sum',
            'Total Cost': 'sum',
            'Unused Cost': 'sum'
        })
        .rename(columns={'Tool': 'Tool Count'})
        .reset_index()
    )
    if not vendor_summary.empty:
        vendor_summary['Total Cost'] = vendor_summary['Total Cost'].fillna(0)
        vendor_summary['Unused Cost'] = vendor_summary['Unused Cost'].fillna(0)
        vendor_summary['Utilization (%)'] = (
            vendor_summary['Used Licenses'] / vendor_summary['Total Licenses'] * 100
        ).round(1).fillna(0)
        vendor_summary['Buffer Seats'] = vendor_summary['Available']

        st.subheader("Vendor Health Overview")
        st.dataframe(
            vendor_summary[
                ['Vendor', 'Tool Count', 'Total Licenses', 'Used Licenses', 'Buffer Seats',
                 'Utilization (%)', 'Total Cost', 'Unused Cost']
            ].style.format({'Total Cost': '${:,.0f}', 'Unused Cost': '${:,.0f}'}),
            hide_index=True
        )

        spend_metric = 'Total Cost' if vendor_summary['Total Cost'].sum() > 0 else 'Tool Count'
        spend_fig = px.pie(
            vendor_summary,
            names='Vendor',
            values=spend_metric,
            title='Spend Distribution by Vendor' if spend_metric == 'Total Cost' else 'Tool Footprint by Vendor',
            hole=0.4
        )
        st.plotly_chart(spend_fig, use_container_width=True)

    # Download filtered snapshot
    st.download_button(
        label="📥 Download Filtered Snapshot (CSV)",
        data=filtered_df.to_csv(index=False),
        file_name="license_dashboard_snapshot.csv",
        mime="text/csv"
    )

    # Bar chart
    fig = px.bar(
        filtered_df,
        x="Tool",
        y=["Used Licenses", "Available"],
        title="License Allocation by Tool",
        color_discrete_sequence=["#e74c3c", "#2ecc71"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table with conditional formatting (use Styler.map to avoid deprecation warnings)
    st.subheader("License Details")
    display_cols = [c for c in ['Tool', 'Vendor', 'Total Licenses', 'Used Licenses', 'Available',
                                'Utilization (%)', 'Cost_Per_Seat_USD', 'Total Cost', 'Unused Cost']
                    if c in filtered_df.columns]

    styled_df = (
        filtered_df[display_cols]
        .style
        .map(lambda x: 'background-color: #a33333; color: white' if x >= 90 else '',
             subset=['Utilization (%)'])
        .map(lambda x: 'background-color: #ffdb58; color: black'
             if x == filtered_df['Total Licenses'].max() and x > 0 else '',
             subset=['Total Licenses'])
        .format({'Cost_Per_Seat_USD': '${:,.0f}', 'Total Cost': '${:,.0f}', 'Unused Cost': '${:,.0f}'})
    )
    st.dataframe(styled_df, hide_index=True)

    # History (from license_snapshots)
    history_df = fetch_license_snapshot_history()
    if not history_df.empty:
        history_chart = history_df.sort_values('snapshot_id').copy()
        history_chart['created_at'] = pd.to_datetime(history_chart['created_at'])
        chart_fig = px.line(
            history_chart,
            x='created_at',
            y=['avg_utilization', 'used_licenses'],
            labels={'created_at': 'Captured', 'value': 'Value', 'variable': 'Metric'},
            title='Utilization & Usage Trend',
        )
        chart_fig.update_yaxes(title='Average Utilization (%) / Used Licenses')
        st.plotly_chart(chart_fig, use_container_width=True)

        history_display = history_df.rename(columns={
            'snapshot_id': 'Snapshot #', 'created_at': 'Captured', 'total_tools': 'Tools',
            'total_licenses': 'Licenses', 'used_licenses': 'In Use', 'avg_utilization': 'Avg Util (%)',
            'total_cost': 'Total Cost (USD)', 'unused_cost': 'Idle Cost (USD)',
        })
        history_display['Captured'] = pd.to_datetime(history_display['Captured']).dt.strftime('%Y-%m-%d %H:%M')
        with st.expander("📚 Historical Dashboard Snapshots", expanded=False):
            st.dataframe(history_display, hide_index=True, use_container_width=True)

    # Risk spotlight + capacity alerts
    st.markdown("### 🔎 Risk Spotlight")
    risk_cols = [c for c in display_cols if c in
                 ['Tool', 'Vendor', 'Total Licenses', 'Used Licenses', 'Available', 'Utilization (%)', 'Unused Cost']]

    shortage_tools = filtered_df[filtered_df['Available'] == 0].sort_values('Utilization (%)', ascending=False)
    idle_tools = filtered_df.sort_values('Unused Cost', ascending=False)

    col_short, col_idle = st.columns(2)
    with col_short:
        st.caption("Critical Shortage (0 seats free)")
        if shortage_tools.empty:
            st.success("No tools are completely allocated. 🎉")
        else:
            st.dataframe(shortage_tools[risk_cols].head(3), hide_index=True)

    with col_idle:
        st.caption("High Idle Spend (Top 3)")
        st.dataframe(idle_tools[risk_cols].head(3), hide_index=True)

    high_pressure = filtered_df[
        (filtered_df['Available'] <= 2)
        & (filtered_df['Available'] >= 0)
        & (filtered_df['Utilization (%)'] >= 85)
    ].sort_values('Utilization (%)', ascending=False)

    if not high_pressure.empty:
        st.markdown("### 🚨 Capacity Alerts")
        st.write("These tools are nearly saturated—initiate procurement or re-allocation discussions before the next tapeout build.")
        st.dataframe(
            high_pressure[risk_cols].assign(**{'Buffer Seats': high_pressure['Available']})[
                ['Tool', 'Vendor', 'Utilization (%)', 'Available', 'Buffer Seats', 'Unused Cost']
            ],
            hide_index=True,
        )

    st.markdown("### 🧮 Capacity Planning Sandbox")
    with st.form("capacity_planner"):
        plan_tool = st.selectbox("Tool to Evaluate", sorted(filtered_df['Tool'].unique()))
        additional_seats = st.number_input("Projected New Seats Needed", min_value=1, max_value=200, value=5, step=1)
        target_util = st.slider("Target Utilization After Procurement (%)", 60, 100, 85, step=5)
        submitted_plan = st.form_submit_button("Run Projection")

    if submitted_plan:
        baseline_df = df[df['Tool'] == plan_tool]
        if baseline_df.empty:
            st.warning("Unable to locate the selected tool in the active dataset.")
            return
        baseline = baseline_df.iloc[0]
        current_total = baseline['Total Licenses']
        current_used = baseline['Used Licenses']
        projected_total = current_total + additional_seats
        projected_util = round((current_used / projected_total) * 100, 1) if projected_total else 0
        cost_per_seat = baseline.get('Cost_Per_Seat_USD', 0) or 0
        incremental_cost = additional_seats * cost_per_seat
        seats_needed_for_target = max(int(np.ceil(current_used / (target_util / 100))) - current_total, 0)

        st.info(
            f"Adding **{additional_seats}** seats for **{plan_tool}** lowers utilization from "
            f"{baseline['Utilization (%)']:.1f}% to approximately **{projected_util:.1f}%**."
        )
        if incremental_cost:
            st.markdown(f"- **Incremental Annual Cost:** `${incremental_cost:,.0f}`")
        st.markdown("- **Buffer After Purchase:** ``{}`` seats".format(projected_total - current_used))
        if seats_needed_for_target > 0:
            st.markdown(f"- To hit the target utilization of {target_util}%, plan for at least ``{seats_needed_for_target}`` additional seats.")
        else:
            st.markdown("- Current capacity already meets the utilization goal.")

    # --- Add New Tool Form ---
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
                current_df = st.session_state.license_data
                if tool_name.title() not in current_df['Tool'].tolist():
                    upsert_license_record(new_data)
                    refresh_license_session_state()
                    st.success(f"Tool '{tool_name.title()}' added, persisted to the database, and dashboard updated!")
                    st.rerun()
                else:
                    st.warning(f"Tool '{tool_name.title()}' already exists.")

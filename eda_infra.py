import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import os
import re
import sqlite3
import uuid
from io import StringIO

# --- Configuration & Data Loading ---
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DATABASE_PATH = os.path.join(DATA_DIR, 'celestial_ai.db')

FALLBACK_LICENSE_DATA = pd.DataFrame([
    {
        'Tool': 'Synopsys DC',
        'Vendor': 'Synopsys',
        'Total Licenses': 10,
        'Used Licenses': 6,
        'Cost_Per_Seat_USD': 25000,
    },
    {
        'Tool': 'Cadence Innovus',
        'Vendor': 'Cadence',
        'Total Licenses': 12,
        'Used Licenses': 9,
        'Cost_Per_Seat_USD': 35000,
    },
    {
        'Tool': 'Siemens Calibre',
        'Vendor': 'Siemens',
        'Total Licenses': 8,
        'Used Licenses': 4,
        'Cost_Per_Seat_USD': 15000,
    },
])

FALLBACK_BUGS = pd.DataFrame([
    {"ID": "BUG-001", "Tool": "Cadence Innovus", "Version": "22.1", "Issue": "Crashes during MMMC analysis", "Workaround": "Use -legacy_mode", "Reported By": "Alice", "Status": "Open"},
    {"ID": "BUG-002", "Tool": "Synopsys ICC2", "Version": "2023.03", "Issue": "Incorrect congestion map", "Workaround": "Run with -fix_congestion_patch", "Reported By": "Bob", "Status": "Fixed"},
])

FALLBACK_TOOL_REGISTRY = pd.DataFrame([
    {"Project": "AI Accelerator", "Tool": "Cadence Innovus", "Approved Version": "23.1", "Compiler": "GCC 9.4"},
    {"Project": "IoT Sensor", "Tool": "Synopsys DC", "Approved Version": "2023.06", "Compiler": "GCC 8.5"},
    {"Project": "CPU Core", "Tool": "Synopsys PT", "Approved Version": "2022.12-SP3", "Compiler": "Clang 12"},
])

FALLBACK_RUN_HISTORY = pd.DataFrame([
    {
        "run_id": "RUN-001",
        "Project": "AI Accelerator",
        "Tool": "Cadence Innovus",
        "Flow Stage": "Place & Route",
        "Owner": "Alice",
        "Status": "Passed",
        "Start": (datetime.now() - timedelta(days=2, hours=6)).isoformat(timespec="minutes"),
        "Duration (hrs)": 5.5,
        "Iteration": 12,
        "Blockers": "",
    },
    {
        "run_id": "RUN-002",
        "Project": "AI Accelerator",
        "Tool": "Synopsys PrimeTime",
        "Flow Stage": "Timing Sign-off",
        "Owner": "Wei",
        "Status": "Failed",
        "Start": (datetime.now() - timedelta(days=1, hours=9)).isoformat(timespec="minutes"),
        "Duration (hrs)": 3.2,
        "Iteration": 4,
        "Blockers": "Clock uncertainty margin too tight",
    },
    {
        "run_id": "RUN-003",
        "Project": "Photon Switch",
        "Tool": "Siemens Calibre",
        "Flow Stage": "DRC",
        "Owner": "Priya",
        "Status": "Running",
        "Start": (datetime.now() - timedelta(hours=2)).isoformat(timespec="minutes"),
        "Duration (hrs)": 6.0,
        "Iteration": 2,
        "Blockers": "",
    },
])

FALLBACK_TAPEOUT_CHECKLIST = pd.DataFrame([
    {
        "task_id": "TASK-001",
        "Project": "AI Accelerator",
        "Milestone": "Floorplanning",
        "Task": "Complete block placement convergence study",
        "Owner": "Alice",
        "Status": "Complete",
        "Due": (datetime.now() - timedelta(days=7)).date().isoformat(),
        "Notes": "Documented in Confluence",
    },
    {
        "task_id": "TASK-002",
        "Project": "AI Accelerator",
        "Milestone": "Timing Closure",
        "Task": "PrimeTime sign-off with SI corners",
        "Owner": "Wei",
        "Status": "In Progress",
        "Due": (datetime.now() + timedelta(days=3)).date().isoformat(),
        "Notes": "Need updated parasitics",
    },
    {
        "task_id": "TASK-003",
        "Project": "Photon Switch",
        "Milestone": "Physical Verification",
        "Task": "Tapeout checklist review with foundry",
        "Owner": "Priya",
        "Status": "Not Started",
        "Due": (datetime.now() + timedelta(days=10)).date().isoformat(),
        "Notes": "Schedule call with GF",
    },
])

RUN_STATUS_OPTIONS = ["Queued", "Running", "Passed", "Failed", "Needs Review", "Aborted"]
CHECKLIST_STATUS_OPTIONS = ["Not Started", "In Progress", "Blocked", "Complete"]


def get_db_connection():
    """Returns a sqlite3 connection to the application database."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Creates the local SQLite database and seeds baseline records if needed."""
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS license_usage (
                Tool TEXT PRIMARY KEY,
                Vendor TEXT,
                TotalLicenses INTEGER,
                UsedLicenses INTEGER,
                Available INTEGER,
                Utilization REAL,
                CostPerSeat REAL,
                TotalCost REAL,
                UnusedCost REAL,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bug_registry (
                ID TEXT PRIMARY KEY,
                Tool TEXT,
                Version TEXT,
                Issue TEXT,
                Workaround TEXT,
                ReportedBy TEXT,
                Status TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_registry (
                Project TEXT,
                Tool TEXT,
                ApprovedVersion TEXT,
                Compiler TEXT,
                PRIMARY KEY (Project, Tool)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_history (
                run_id TEXT PRIMARY KEY,
                project TEXT,
                tool TEXT,
                flow_stage TEXT,
                owner TEXT,
                status TEXT,
                start_time TEXT,
                duration_hours REAL,
                iteration INTEGER,
                blockers TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS design_checklist (
                task_id TEXT PRIMARY KEY,
                project TEXT,
                milestone TEXT,
                task TEXT,
                owner TEXT,
                status TEXT,
                due_date TEXT,
                notes TEXT,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS license_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                total_tools INTEGER,
                total_licenses INTEGER,
                used_licenses INTEGER,
                avg_utilization REAL,
                total_cost REAL,
                unused_cost REAL
            )
            """
        )

        _seed_database(conn)


def _seed_database(conn):
    """Seeds the database with baseline values if the tables are empty."""
    license_count = conn.execute("SELECT COUNT(*) FROM license_usage").fetchone()[0]
    if license_count == 0:
        for _, row in FALLBACK_LICENSE_DATA.iterrows():
            total = int(row['Total Licenses'])
            used = min(int(row['Used Licenses']), total)
            available = total - used
            cost_per = float(row.get('Cost_Per_Seat_USD') or 0)
            total_cost = cost_per * total if cost_per else None
            unused_cost = cost_per * available if cost_per else None
            utilization = round(used / total * 100, 1) if total else 0.0
            conn.execute(
                """
                INSERT INTO license_usage (Tool, Vendor, TotalLicenses, UsedLicenses, Available, Utilization, CostPerSeat, TotalCost, UnusedCost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row['Tool'],
                    row['Vendor'],
                    total,
                    used,
                    available,
                    utilization,
                    cost_per if cost_per else None,
                    total_cost,
                    unused_cost,
                ),
            )

    bug_count = conn.execute("SELECT COUNT(*) FROM bug_registry").fetchone()[0]
    if bug_count == 0:
        for _, row in FALLBACK_BUGS.iterrows():
            conn.execute(
                """
                INSERT INTO bug_registry (ID, Tool, Version, Issue, Workaround, ReportedBy, Status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row['ID'],
                    row['Tool'],
                    row['Version'],
                    row['Issue'],
                    row['Workaround'],
                    row.get('Reported By', row.get('ReportedBy', 'Unknown')),
                    row['Status'],
                ),
            )

    tool_count = conn.execute("SELECT COUNT(*) FROM tool_registry").fetchone()[0]
    if tool_count == 0:
        for _, row in FALLBACK_TOOL_REGISTRY.iterrows():
            conn.execute(
                """
                INSERT INTO tool_registry (Project, Tool, ApprovedVersion, Compiler)
                VALUES (?, ?, ?, ?)
                """,
                (
                    row['Project'],
                    row['Tool'],
                    row['Approved Version'],
                    row['Compiler'],
                ),
            )

    run_count = conn.execute("SELECT COUNT(*) FROM run_history").fetchone()[0]
    if run_count == 0:
        for _, row in FALLBACK_RUN_HISTORY.iterrows():
            conn.execute(
                """
                INSERT INTO run_history (
                    run_id, project, tool, flow_stage, owner, status, start_time, duration_hours, iteration, blockers
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row['run_id'],
                    row['Project'],
                    row['Tool'],
                    row['Flow Stage'],
                    row['Owner'],
                    row['Status'],
                    row['Start'],
                    row['Duration (hrs)'],
                    row['Iteration'],
                    row['Blockers'],
                ),
            )

    checklist_count = conn.execute("SELECT COUNT(*) FROM design_checklist").fetchone()[0]
    if checklist_count == 0:
        for _, row in FALLBACK_TAPEOUT_CHECKLIST.iterrows():
            conn.execute(
                """
                INSERT INTO design_checklist (
                    task_id, project, milestone, task, owner, status, due_date, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row['task_id'],
                    row['Project'],
                    row['Milestone'],
                    row['Task'],
                    row['Owner'],
                    row['Status'],
                    row['Due'],
                    row['Notes'],
                ),
            )

    conn.commit()


def list_database_tables():
    """Returns the application table names in deterministic order."""
    if not os.path.exists(DATABASE_PATH):
        return []

    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()

    return [row["name"] for row in rows]


def get_database_overview():
    """Collects row counts, schema summaries, and last-updated hints for each table."""
    tables = list_database_tables()
    overview_rows = []

    if not tables:
        return tables, pd.DataFrame(columns=["Table", "Rows", "Columns", "Last Updated"])

    timestamp_candidates = {"last_updated", "created_at", "updated_at", "modified_at"}

    with get_db_connection() as conn:
        for table in tables:
            column_rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            columns = [row["name"] for row in column_rows]
            column_summary = ", ".join(columns)

            count = conn.execute(f'SELECT COUNT(*) AS row_count FROM "{table}"').fetchone()["row_count"]

            timestamp_column = next((col for col in columns if col in timestamp_candidates), None)
            last_updated = None
            if timestamp_column:
                last_updated = conn.execute(
                    f'SELECT MAX("{timestamp_column}") AS ts FROM "{table}"'
                ).fetchone()["ts"]

            overview_rows.append(
                {
                    "Table": table,
                    "Rows": int(count or 0),
                    "Columns": column_summary,
                    "Last Updated": last_updated or "N/A",
                }
            )

    overview_df = pd.DataFrame(overview_rows)
    return tables, overview_df


def fetch_table_preview(table_name: str, limit: int = 100):
    """Returns a limited preview of the requested table ordered by most recent entries."""
    if not table_name:
        return pd.DataFrame()

    with get_db_connection() as conn:
        query = f'SELECT * FROM "{table_name}" ORDER BY rowid DESC LIMIT ?'
        df = pd.read_sql_query(query, conn, params=(limit,))

    return df


def perform_database_maintenance():
    """Runs lightweight VACUUM/ANALYZE maintenance on the SQLite file."""
    if not os.path.exists(DATABASE_PATH):
        return

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.isolation_level = None  # required for VACUUM
        conn.execute("VACUUM")
        conn.execute("ANALYZE")


def _fetch_license_usage(conn):
    query = (
        "SELECT Tool, Vendor, TotalLicenses, UsedLicenses, Available, Utilization, CostPerSeat, TotalCost, UnusedCost "
        "FROM license_usage ORDER BY Tool"
    )
    return pd.read_sql_query(query, conn)


def fetch_license_data_from_db():
    with get_db_connection() as conn:
        df = _fetch_license_usage(conn)
    if df.empty:
        return pd.DataFrame()
    df = df.rename(
        columns={
            'TotalLicenses': 'Total Licenses',
            'UsedLicenses': 'Used Licenses',
            'Utilization': 'Utilization (%)',
            'CostPerSeat': 'Cost_Per_Seat_USD',
            'TotalCost': 'Total Cost',
            'UnusedCost': 'Unused Cost',
        }
    )
    return df


def fetch_bug_registry_from_db():
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            "SELECT ID, Tool, Version, Issue, Workaround, ReportedBy AS 'Reported By', Status FROM bug_registry ORDER BY created_at DESC",
            conn,
        )
    return df


def fetch_tool_registry_from_db():
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            "SELECT Project, Tool, ApprovedVersion AS 'Approved Version', Compiler FROM tool_registry ORDER BY Project",
            conn,
        )
    return df


def fetch_run_history_from_db(limit: int = 250):
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT run_id, project AS Project, tool AS Tool, flow_stage AS 'Flow Stage', owner AS Owner,
                   status AS Status, start_time AS Start, duration_hours AS 'Duration (hrs)',
                   iteration AS Iteration, blockers AS Blockers, updated_at
            FROM run_history
            ORDER BY datetime(start_time) DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
    if not df.empty:
        df['Start'] = pd.to_datetime(df['Start'])
        df['Duration (hrs)'] = pd.to_numeric(df['Duration (hrs)'], errors='coerce')
    return df


def fetch_design_checklist_from_db():
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT task_id, project AS Project, milestone AS Milestone, task AS Task, owner AS Owner,
                   status AS Status, due_date AS Due, notes AS Notes, last_updated
            FROM design_checklist
            ORDER BY project, milestone
            """,
            conn,
        )
    return df


def upsert_license_record(record):
    total = int(record.get('Total Licenses', 0) or 0)
    used = min(int(record.get('Used Licenses', 0) or 0), total) if total else 0
    available = total - used
    cost_per = record.get('Cost_Per_Seat_USD')
    cost_per = float(cost_per) if cost_per not in (None, "", np.nan) else None
    total_cost = cost_per * total if cost_per is not None else None
    unused_cost = cost_per * available if cost_per is not None else None
    utilization = round(used / total * 100, 1) if total else 0.0

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO license_usage (
                Tool, Vendor, TotalLicenses, UsedLicenses, Available, Utilization, CostPerSeat, TotalCost, UnusedCost, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(Tool) DO UPDATE SET
                Vendor=excluded.Vendor,
                TotalLicenses=excluded.TotalLicenses,
                UsedLicenses=excluded.UsedLicenses,
                Available=excluded.Available,
                Utilization=excluded.Utilization,
                CostPerSeat=excluded.CostPerSeat,
                TotalCost=excluded.TotalCost,
                UnusedCost=excluded.UnusedCost,
                last_updated=CURRENT_TIMESTAMP
            """,
            (
                record.get('Tool'),
                record.get('Vendor'),
                total,
                used,
                available,
                utilization,
                cost_per,
                total_cost,
                unused_cost,
            ),
        )
        conn.commit()


def bulk_upsert_license_costs(cost_df: pd.DataFrame):
    if cost_df is None or cost_df.empty:
        return

    normalized = cost_df.rename(columns={
        'Total Licenses': 'Total_Licenses',
        'Total_License': 'Total_Licenses',
        'TotalLicenses': 'Total_Licenses',
        'Cost Per Seat': 'Cost_Per_Seat_USD',
        'CostPerSeat': 'Cost_Per_Seat_USD',
    })

    with get_db_connection() as conn:
        existing = {
            row['Tool']: {
                'Vendor': row['Vendor'],
                'UsedLicenses': row['UsedLicenses'],
                'CostPerSeat': row['CostPerSeat'],
            }
            for row in conn.execute(
                "SELECT Tool, Vendor, UsedLicenses, CostPerSeat FROM license_usage"
            ).fetchall()
        }

    for _, row in normalized.iterrows():
        tool = str(row.get('Tool', '') or '').strip()
        if not tool:
            continue

        total = int(row.get('Total_Licenses') or 0)
        vendor = row.get('Vendor') or existing.get(tool, {}).get('Vendor')
        if not vendor:
            lower_tool = tool.lower()
            if 'synopsys' in lower_tool:
                vendor = 'Synopsys'
            elif 'cadence' in lower_tool:
                vendor = 'Cadence'
            elif 'siemens' in lower_tool:
                vendor = 'Siemens'
            else:
                vendor = 'Other'
        cost_per = row.get('Cost_Per_Seat_USD')
        if pd.isna(cost_per):
            cost_per = existing.get(tool, {}).get('CostPerSeat')

        base_used = existing.get(tool, {}).get('UsedLicenses', 0) if existing.get(tool) else 0
        used = row.get('Used Licenses')
        used = int(used) if not pd.isna(used) and used is not None else int(base_used or max(total // 2, 0))
        upsert_license_record({
            'Tool': tool,
            'Vendor': vendor,
            'Total Licenses': total,
            'Used Licenses': min(used, total),
            'Cost_Per_Seat_USD': cost_per,
        })


def record_license_snapshot(df: pd.DataFrame):
    if df is None or df.empty:
        return

    snapshot = {
        'total_tools': int(len(df)),
        'total_licenses': int(df['Total Licenses'].sum()),
        'used_licenses': int(df['Used Licenses'].sum()),
        'avg_utilization': float(df['Utilization (%)'].mean()) if not df.empty else 0.0,
        'total_cost': float(df['Total Cost'].sum()) if 'Total Cost' in df.columns else 0.0,
        'unused_cost': float(df['Unused Cost'].sum()) if 'Unused Cost' in df.columns else 0.0,
    }

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO license_snapshots (total_tools, total_licenses, used_licenses, avg_utilization, total_cost, unused_cost)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot['total_tools'],
                snapshot['total_licenses'],
                snapshot['used_licenses'],
                snapshot['avg_utilization'],
                snapshot['total_cost'],
                snapshot['unused_cost'],
            ),
        )
        conn.commit()


def log_run_event(run_record):
    """Insert or update a run tracking record."""
    start_time = run_record.get('Start')
    if isinstance(start_time, datetime):
        start_time = start_time.isoformat(timespec="minutes")

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO run_history (
                run_id, project, tool, flow_stage, owner, status, start_time, duration_hours, iteration, blockers, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id) DO UPDATE SET
                project=excluded.project,
                tool=excluded.tool,
                flow_stage=excluded.flow_stage,
                owner=excluded.owner,
                status=excluded.status,
                start_time=excluded.start_time,
                duration_hours=excluded.duration_hours,
                iteration=excluded.iteration,
                blockers=excluded.blockers,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                run_record.get('run_id'),
                run_record.get('Project'),
                run_record.get('Tool'),
                run_record.get('Flow Stage'),
                run_record.get('Owner'),
                run_record.get('Status'),
                start_time,
                run_record.get('Duration (hrs)'),
                run_record.get('Iteration'),
                run_record.get('Blockers'),
            ),
        )
        conn.commit()


def update_run_status(run_id: str, status: str, blockers: str | None = None, duration_hours: float | None = None):
    """Update status, blockers, or duration for an existing run."""
    if not run_id:
        return

    updates = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
    params = [status]

    if blockers is not None:
        updates.append("blockers = ?")
        params.append(blockers)

    if duration_hours is not None:
        updates.append("duration_hours = ?")
        params.append(duration_hours)

    params.append(run_id)

    with get_db_connection() as conn:
        conn.execute(
            f"UPDATE run_history SET {', '.join(updates)} WHERE run_id = ?",
            params,
        )
        conn.commit()


def upsert_checklist_task(task_record):
    """Insert or update a tapeout checklist item."""
    due_date = task_record.get('Due')
    if isinstance(due_date, datetime):
        due_date = due_date.date().isoformat()

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO design_checklist (
                task_id, project, milestone, task, owner, status, due_date, notes, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(task_id) DO UPDATE SET
                project=excluded.project,
                milestone=excluded.milestone,
                task=excluded.task,
                owner=excluded.owner,
                status=excluded.status,
                due_date=excluded.due_date,
                notes=excluded.notes,
                last_updated=CURRENT_TIMESTAMP
            """,
            (
                task_record.get('task_id'),
                task_record.get('Project'),
                task_record.get('Milestone'),
                task_record.get('Task'),
                task_record.get('Owner'),
                task_record.get('Status'),
                due_date,
                task_record.get('Notes'),
            ),
        )
        conn.commit()


def fetch_license_snapshot_history(limit: int = 15):
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT snapshot_id, created_at, total_tools, total_licenses, used_licenses, avg_utilization, total_cost, unused_cost
            FROM license_snapshots
            ORDER BY snapshot_id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
    return df


def insert_bug_record(bug):
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO bug_registry (ID, Tool, Version, Issue, Workaround, ReportedBy, Status, created_at)
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM bug_registry WHERE ID = ?), CURRENT_TIMESTAMP)
            )
            """,
            (
                bug['ID'],
                bug['Tool'],
                bug.get('Version'),
                bug.get('Issue'),
                bug.get('Workaround'),
                bug.get('Reported By'),
                bug.get('Status'),
                bug['ID'],
            ),
        )
        conn.commit()


def generate_bug_id():
    with get_db_connection() as conn:
        ids = [row[0] for row in conn.execute("SELECT ID FROM bug_registry").fetchall()]
    numbers = [int(re.findall(r"\d+", bug_id)[0]) for bug_id in ids if re.findall(r"\d+", bug_id)]
    next_number = max(numbers) + 1 if numbers else 1
    return f"BUG-{next_number:03d}"


def generate_run_id():
    return f"RUN-{uuid.uuid4().hex[:8].upper()}"


def generate_task_id():
    return f"TASK-{uuid.uuid4().hex[:8].upper()}"


def refresh_bug_registry_session_state():
    st.session_state.bug_registry = fetch_bug_registry_from_db()


def refresh_license_session_state():
    st.session_state.license_data = fetch_license_data_from_db()


def refresh_tool_registry_session_state():
    st.session_state.tool_registry = fetch_tool_registry_from_db()


def refresh_run_history_session_state():
    st.session_state.run_history = fetch_run_history_from_db()


def refresh_checklist_session_state():
    st.session_state.design_checklist = fetch_design_checklist_from_db()

def _clean_license_costs(path, df):
    """Attempts to repair malformed license cost CSVs shipped with the app."""
    if not df.empty and 'Tool' in df.columns:
        return df

    try:
        with open(path, encoding='utf-8') as handle:
            raw = handle.read().strip()
    except FileNotFoundError:
        return pd.DataFrame()

    if not raw:
        return pd.DataFrame()

    # Remove leading/trailing quotes that collapse the CSV into a single cell
    raw = raw.strip('"').replace('",,,', '')
    raw = raw.replace('\r\n', '\n')

    # Normalize column naming before parsing
    raw = raw.replace('Cost_Per_Year', 'Cost_Per_Seat_USD')
    raw = raw.replace('Seats ', 'Seats\n')
    # Insert newlines between consecutive records (after seat counts)
    raw = re.sub(r'(?<=\d)\s+(?=[A-Z])', '\n', raw)

    repaired = pd.read_csv(StringIO(raw))
    repaired = repaired.dropna(how='all')
    repaired = repaired.rename(columns={'Seats': 'Total Licenses'})
    numeric_cols = ['Cost_Per_Seat_USD', 'Total Licenses']
    for col in numeric_cols:
        if col in repaired.columns:
            repaired[col] = pd.to_numeric(repaired[col], errors='coerce')
    if not repaired.empty and 'Tool' in repaired.columns:
        st.session_state['license_costs_repaired'] = True
    return repaired


def load_data(filename):
    """Loads CSV files from the data directory."""
    path = os.path.join(DATA_DIR, filename)
    try:
        # Attempt to load the user's local CSV
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        df = pd.DataFrame()
    except FileNotFoundError:
        # Provide sensible default if data dir not accessible (e.g., in canvas)
        if 'license_costs.csv' in filename:
            return pd.DataFrame({
                'Tool': ['Synopsys DC', 'Cadence Innovus', 'Siemens Calibre'],
                'Vendor': ['Synopsys', 'Cadence', 'Siemens'],
                'Cost_Per_Seat_USD': [25000, 35000, 15000],
                'Total Licenses': [10, 12, 8]
            })
        if 'known_bugs.csv' in filename:
            return pd.DataFrame([
                {"ID": "BUG-001", "Tool": "Cadence Innovus", "Version": "22.1", "Issue": "Crashes during MMMC analysis", "Workaround": "Use -legacy_mode", "Reported By": "Alice", "Status": "Open"},
                {"ID": "BUG-002", "Tool": "Synopsys ICC2", "Version": "2023.03", "Issue": "Incorrect congestion map", "Workaround": "Run with -fix_congestion_patch", "Reported By": "Bob", "Status": "Fixed in 2023.06"}
            ])
        return pd.DataFrame()

    if 'license_costs.csv' in filename:
        df = _clean_license_costs(path, df)
    return df if not df.empty else pd.DataFrame()

def init_session_state():
    """Initializes session state, database, and baseline records on first run."""
    if not st.session_state.get('_db_ready', False):
        init_database()
        st.session_state['_db_ready'] = True

    # Load license costs from CSV (if provided) to refresh the database
    license_costs_csv = load_data('license_costs.csv')

    if 'Tool' not in license_costs_csv.columns or license_costs_csv.empty:
        # CSV missing Tool column or empty -> fall back to baked-in defaults
        st.warning("License cost data loaded is missing the 'Tool' column. Using synthetic fallback data for initialization.")
        bulk_upsert_license_costs(FALLBACK_LICENSE_DATA)
        st.session_state['license_costs_repaired'] = False
    else:
        bulk_upsert_license_costs(license_costs_csv)
        if st.session_state.get('license_costs_repaired'):
            st.info("Recovered license cost data from malformed CSV. Please fix the source file when convenient.")
            st.session_state['license_costs_repaired'] = False

    refresh_license_session_state()
    refresh_bug_registry_session_state()
    refresh_tool_registry_session_state()
    refresh_run_history_session_state()
    refresh_checklist_session_state()

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

    # Load costs for cost analysis (robust to malformed/missing CSV)
    try:
        cost_df = load_data('license_costs.csv')
    except KeyError:
        cost_df = pd.DataFrame()

    merge_columns = ('Vendor', 'Cost_Per_Seat_USD', 'Total Licenses')

    if not cost_df.empty and 'Tool' in cost_df.columns:
        cost_df = cost_df.set_index('Tool')
        df = df.join(cost_df, on='Tool', how='left', rsuffix='_cost_join')
    else:
        df = df.copy()

    # Harmonize columns introduced by the join, preferring live dashboard values
    for col in merge_columns:
        join_col = f"{col}_cost_join"
        if join_col in df.columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[join_col])
            else:
                df[col] = df[join_col]
            df.drop(columns=[join_col], inplace=True)

    # If the join was successful (i.e., 'Cost_Per_Seat_USD' exists from the loaded CSV or fallback)
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
        min_utilization = st.slider(
            "Minimum Utilization (%)", 0.0, slider_upper, 0.0, step=5.0
        )

    filtered_df = df[df['Vendor'].fillna('Unknown').isin(selected_vendors)].copy()
    filtered_df = filtered_df[filtered_df['Utilization (%)'] >= min_utilization]

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

    unused_cost = filtered_df['Unused Cost'].sum()
    top_idle_tool = filtered_df.sort_values('Unused Cost', ascending=False).iloc[0]
    col4.metric(
        "Highest Idle Spend",
        f"${top_idle_tool['Unused Cost']:,.0f}",
        help=f"{top_idle_tool['Tool']} is carrying the largest unused license cost."
    )

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
                ['Vendor', 'Tool Count', 'Total Licenses', 'Used Licenses', 'Buffer Seats', 'Utilization (%)', 'Total Cost', 'Unused Cost']
            ].style.format({
                'Total Cost': '${:,.0f}',
                'Unused Cost': '${:,.0f}'
            }),
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

    # Provide download of filtered snapshot
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
    
    # Select subset of columns to display
    display_cols = [col for col in ['Tool', 'Vendor', 'Total Licenses', 'Used Licenses', 'Available', 'Utilization (%)', 'Cost_Per_Seat_USD', 'Total Cost', 'Unused Cost'] if col in df.columns]

    # FIX: Apply styling and formatting to the sliced DataFrame and save the Styler object.
    # The Styler object itself is not subscriptable.
    styled_df = filtered_df[display_cols].style.map( # CHANGED .applymap TO .map
        # Use a high-contrast color for utilization warning (dark red on dark theme)
        lambda x: 'background-color: #a33333; color: white' if x >= 90 else '',
        subset=['Utilization (%)']
    ).map( # CHANGED .applymap TO .map
        # Use a light gold/yellow color to highlight the largest license pool
        lambda x: 'background-color: #ffdb58; color: black' if x == filtered_df['Total Licenses'].max() and x > 0 else '',
        subset=['Total Licenses']
    ).format({
        'Cost_Per_Seat_USD': '${:,.0f}',
        'Total Cost': '${:,.0f}',
        'Unused Cost': '${:,.0f}'
    })

    # Pass the resulting Styler object directly to st.dataframe
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
            labels={
                'created_at': 'Captured',
                'value': 'Value',
                'variable': 'Metric',
            },
            title='Utilization & Usage Trend',
        )
        chart_fig.update_yaxes(title='Average Utilization (%) / Used Licenses')
        st.plotly_chart(chart_fig, use_container_width=True)

        history_display = history_df.rename(
            columns={
                'snapshot_id': 'Snapshot #',
                'created_at': 'Captured',
                'total_tools': 'Tools',
                'total_licenses': 'Licenses',
                'used_licenses': 'In Use',
                'avg_utilization': 'Avg Util (%)',
                'total_cost': 'Total Cost (USD)',
                'unused_cost': 'Idle Cost (USD)',
            }
        )
        history_display['Captured'] = pd.to_datetime(history_display['Created']).dt.strftime('%Y-%m-%d %H:%M') # FIX: Renamed column used in dt.strftime
        with st.expander("📚 Historical Dashboard Snapshots", expanded=False):
            st.dataframe(
                history_display,
                hide_index=True,
                use_container_width=True,
            )

    # Spotlight the most constrained and most underused tools
    st.markdown("### 🔎 Risk Spotlight")
    risk_cols = [col for col in display_cols if col in ['Tool', 'Vendor', 'Total Licenses', 'Used Licenses', 'Available', 'Utilization (%)', 'Unused Cost']]

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
        st.write(
            "These tools are nearly saturated—initiate procurement or re-allocation discussions before the next tapeout build."
        )
        st.dataframe(
            high_pressure[risk_cols].assign(**{
                'Buffer Seats': high_pressure['Available'],
            })[
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
        st.markdown(
            "- **Buffer After Purchase:** ``{}`` seats".format(projected_total - current_used)
        )
        if seats_needed_for_target > 0:
            st.markdown(
                f"- To hit the target utilization of {target_util}%, plan for at least ``{seats_needed_for_target}`` additional seats."
            )
        else:
            st.markdown("- Current capacity already meets the utilization goal.")

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
                current_df = st.session_state.license_data
                if tool_name.title() not in current_df['Tool'].tolist():
                    upsert_license_record(new_data)
                    refresh_license_session_state()
                    st.success(f"Tool '{tool_name.title()}' added, persisted to the database, and dashboard updated!")
                    st.rerun()
                else:
                    st.warning(f"Tool '{tool_name.title()}' already exists.")


def render_bug_tracker():
    st.header("🐞 Known EDA Tool Issues & Workarounds")
    st.markdown("""
    > Central repository for tool bugs, workarounds, and status—persisted in the CAD database for cross-session visibility.
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
            new_id = generate_bug_id()
            new_bug = {
                "ID": new_id,
                "Tool": tool.title(),
                "Version": version,
                "Issue": issue,
                "Workaround": workaround if workaround else "None yet.",
                "Reported By": reported_by,
                "Status": "Open"
            }

            insert_bug_record(new_bug)
            refresh_bug_registry_session_state()
            st.success(f"Bug {new_id} reported successfully and saved to the database!")
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


def render_run_tracker():
    st.header("🚀 Flow Run Tracker")
    st.markdown(
        "> Keep tabs on long-running PnR, STA, and verification jobs so bottlenecks are surfaced before schedules slip."
    )

    run_df = st.session_state.get('run_history', fetch_run_history_from_db())
    if run_df.empty:
        st.info("No EDA runs have been logged yet. Use the form below to capture the first one.")
    else:
        run_df = run_df.copy()
        run_df['Start'] = pd.to_datetime(run_df['Start'], errors='coerce')
        run_df['Duration (hrs)'] = pd.to_numeric(run_df['Duration (hrs)'], errors='coerce')

        projects = sorted(run_df['Project'].dropna().unique().tolist())
        statuses = sorted(run_df['Status'].dropna().unique().tolist())

        with st.expander("🔍 Filter runs", expanded=True):
            selected_projects = st.multiselect(
                "Project", projects, default=projects
            )
            status_defaults = [status for status in RUN_STATUS_OPTIONS if status in statuses] or statuses
            selected_statuses = st.multiselect(
                "Status", statuses or RUN_STATUS_OPTIONS, default=status_defaults if status_defaults else RUN_STATUS_OPTIONS
            )
            lookback_days = st.slider("Lookback window (days)", min_value=3, max_value=90, value=30, step=1)

        filtered_df = run_df.copy()
        if selected_projects:
            filtered_df = filtered_df[filtered_df['Project'].isin(selected_projects)]
        if selected_statuses:
            filtered_df = filtered_df[filtered_df['Status'].isin(selected_statuses)]

        cutoff = datetime.now() - timedelta(days=lookback_days)
        filtered_df = filtered_df[filtered_df['Start'].isna() | (filtered_df['Start'] >= cutoff)]

        if filtered_df.empty:
            st.warning("No runs match the chosen filters or time window.")
        else:
            total_runs = len(filtered_df)
            completed_runs = filtered_df['Status'].isin(['Passed']).sum()
            active_runs = filtered_df['Status'].isin(['Running', 'Queued']).sum()
            success_rate = (completed_runs / total_runs * 100) if total_runs else 0
            average_duration = filtered_df['Duration (hrs)'].dropna().mean()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Runs tracked", total_runs)
            col2.metric("Pass rate", f"{success_rate:.1f}%")
            col3.metric("Active jobs", active_runs)
            col4.metric("Avg duration", f"{average_duration:.1f} h" if not np.isnan(average_duration) else "--")

            stage_summary = (
                filtered_df.groupby(['Flow Stage', 'Status'], dropna=False)['Project']
                .count()
                .unstack(fill_value=0)
                .sort_index()
            )
            with st.expander("Stage health", expanded=False):
                st.dataframe(stage_summary, use_container_width=True)

            timeline_df = filtered_df.dropna(subset=['Start']).copy()
            if not timeline_df.empty:
                timeline_df['Finish'] = timeline_df.apply(
                    lambda row: row['Start'] + pd.to_timedelta(max(row.get('Duration (hrs)') or 0.5, 0.5), unit='h'),
                    axis=1,
                )
                timeline_fig = px.timeline(
                    timeline_df,
                    x_start='Start',
                    x_end='Finish',
                    y='Project',
                    color='Status',
                    hover_data=['Tool', 'Flow Stage', 'Owner', 'Duration (hrs)', 'Iteration', 'Blockers'],
                    title='Recent run timeline',
                )
                timeline_fig.update_yaxes(autorange="reversed")
                st.plotly_chart(timeline_fig, use_container_width=True)

                editor_columns = [
                    'run_id', 'Project', 'Tool', 'Flow Stage', 'Owner', 'Status', 'Start', 'Duration (hrs)', 'Iteration', 'Blockers'
                ]
                editor_df = filtered_df[editor_columns].set_index('run_id')
                status_choices = list(dict.fromkeys(RUN_STATUS_OPTIONS + statuses))

                edited_df = st.data_editor(
                    editor_df,
                    num_rows="fixed",
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Status': st.column_config.SelectboxColumn('Status', options=status_choices),
                        'Duration (hrs)': st.column_config.NumberColumn('Duration (hrs)', min_value=0.0, step=0.25),
                        'Start': st.column_config.DatetimeColumn('Start', disabled=True),
                        'Project': st.column_config.TextColumn('Project', disabled=True),
                        'Tool': st.column_config.TextColumn('Tool', disabled=True),
                        'Flow Stage': st.column_config.TextColumn('Flow Stage', disabled=True),
                        'Owner': st.column_config.TextColumn('Owner', disabled=True),
                        'Iteration': st.column_config.NumberColumn('Iteration', disabled=True),
                    },
                )

                if st.button("💾 Save run updates", use_container_width=True):
                    changes = 0
                    for run_id, edited_row in edited_df.iterrows():
                        original_row = editor_df.loc[run_id]
                        status_changed = edited_row['Status'] != original_row['Status']
                        duration_changed = not pd.isna(edited_row['Duration (hrs)']) and (
                            pd.isna(original_row['Duration (hrs)'])
                            or float(edited_row['Duration (hrs)']) != float(original_row['Duration (hrs)'])
                        )
                        blockers_changed = (edited_row['Blockers'] or '') != (original_row['Blockers'] or '')

                        if status_changed or duration_changed or blockers_changed:
                            update_run_status(
                                run_id,
                                edited_row['Status'],
                                blockers=edited_row['Blockers'] if blockers_changed else None,
                                duration_hours=float(edited_row['Duration (hrs)']) if duration_changed else None,
                            )
                            changes += 1

                    if changes:
                        refresh_run_history_session_state()
                        st.success(f"Updated {changes} run{'s' if changes != 1 else ''}.")
                        st.rerun()
                    else:
                        st.info("No run updates detected.")

                st.download_button(
                    "⬇️ Export filtered runs",
                    data=filtered_df.drop(columns=['updated_at'], errors='ignore').to_csv(index=False),
                    file_name="eda_run_tracker.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with st.expander("➕ Log a new run"):
            with st.form("new_run_form"):
                run_state_df = st.session_state.get('run_history', pd.DataFrame())
                if isinstance(run_state_df, pd.DataFrame) and 'Project' in run_state_df.columns:
                    existing_projects = sorted(set(run_state_df['Project'].dropna()))
                else:
                    existing_projects = []
                default_project = existing_projects[0] if existing_projects else ""
                project = st.text_input("Project", value=default_project)
                tool = st.text_input("Tool", placeholder="e.g., Cadence Innovus")
                flow_stage = st.selectbox(
                    "Flow stage",
                    [
                        "Synthesis",
                        "Floorplanning",
                        "Place & Route",
                        "Timing Sign-off",
                        "Power Analysis",
                        "DRC",
                        "LVS",
                        "Custom",
                    ],
                )
                owner = st.text_input("Owner", value=st.session_state.get('user', ""))
                status = st.selectbox("Status", RUN_STATUS_OPTIONS, index=RUN_STATUS_OPTIONS.index("Queued"))
                start_time = st.datetime_input("Start time", value=datetime.now())
                duration_hours = st.number_input("Duration (hrs)", min_value=0.0, step=0.25, value=1.0)
                iteration = st.number_input("Iteration", min_value=0, step=1, value=1)
                blockers = st.text_area("Blockers / Notes", height=80)

                submitted = st.form_submit_button("Log run")
                if submitted:
                    required = [project.strip(), tool.strip(), flow_stage.strip(), owner.strip()]
                    if not all(required):
                        st.warning("Project, tool, flow stage, and owner are required to log a run.")
                    else:
                        record = {
                            'run_id': generate_run_id(),
                            'Project': project.strip(),
                            'Tool': tool.strip(),
                            'Flow Stage': flow_stage,
                            'Owner': owner.strip(),
                            'Status': status,
                            'Start': start_time,
                            'Duration (hrs)': float(duration_hours),
                            'Iteration': int(iteration),
                            'Blockers': blockers.strip(),
                        }
                        log_run_event(record)
                        refresh_run_history_session_state()
                        st.success("Run logged successfully.")
                        st.rerun()


def render_tapeout_checklist():
    st.header("✅ Tapeout Readiness Checklist")
    st.markdown(
        "> Align CAD deliverables, sign-off tasks, and foundry checklists so engineering knows exactly what remains."
    )

    checklist_df = st.session_state.get('design_checklist', fetch_design_checklist_from_db())
    if checklist_df.empty:
        st.info("No checklist items yet—use the form below to seed your tapeout plan.")
    else:
        checklist_df = checklist_df.copy()
        checklist_df['Due'] = pd.to_datetime(checklist_df['Due'], errors='coerce')

        projects = sorted(checklist_df['Project'].dropna().unique().tolist())
        milestones = sorted(checklist_df['Milestone'].dropna().unique().tolist())

        with st.expander("🔍 Filter checklist", expanded=True):
            selected_projects = st.multiselect("Project", projects, default=projects)
            selected_milestones = st.multiselect("Milestone", milestones, default=milestones)
            show_only_open = st.checkbox("Show only open items", value=False)

        filtered_df = checklist_df.copy()
        if selected_projects:
            filtered_df = filtered_df[filtered_df['Project'].isin(selected_projects)]
        if selected_milestones:
            filtered_df = filtered_df[filtered_df['Milestone'].isin(selected_milestones)]
        if show_only_open:
            filtered_df = filtered_df[filtered_df['Status'] != 'Complete']

        if filtered_df.empty:
            st.warning("No checklist items match the current filters.")
        else:
            total_tasks = len(filtered_df)
            completed = (filtered_df['Status'] == 'Complete').sum()
            blocked = filtered_df['Status'].str.contains('Blocked', case=False, na=False).sum()
            due_soon = filtered_df[
                (filtered_df['Status'] != 'Complete')
                & (filtered_df['Due'].notna())
                & (filtered_df['Due'] <= datetime.now() + timedelta(days=3))
            ]

            completion_rate = (completed / total_tasks * 100) if total_tasks else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Tasks tracked", total_tasks)
            col2.metric("Completion", f"{completion_rate:.1f}%")
            col3.metric("Blocked", blocked)
            col4.metric("Due in 3 days", len(due_soon))

            progress_weights = {
                'Complete': 1.0,
                'In Progress': 0.5,
                'Blocked': 0.0,
                'Not Started': 0.0,
                'Needs Review': 0.25,
            }
            filtered_df['Progress'] = filtered_df['Status'].map(progress_weights).fillna(0.5)
            milestone_summary = (
                filtered_df.groupby('Milestone')
                .agg(
                    Tasks=('Task', 'count'),
                    Complete=('Status', lambda s: (s == 'Complete').sum()),
                    Progress=('Progress', 'mean'),
                )
                .reset_index()
            )

            with st.expander("Milestone progress", expanded=False):
                for _, row in milestone_summary.iterrows():
                    st.markdown(f"**{row['Milestone']}** — {int(row['Complete'])}/{int(row['Tasks'])} complete")
                    st.progress(min(max(row['Progress'], 0), 1.0))

            editor_columns = ['task_id', 'Project', 'Milestone', 'Task', 'Owner', 'Status', 'Due', 'Notes']
            editor_df = filtered_df[editor_columns].set_index('task_id')
            status_choices = list(dict.fromkeys(CHECKLIST_STATUS_OPTIONS + filtered_df['Status'].dropna().tolist()))

            edited_df = st.data_editor(
                editor_df,
                num_rows="fixed",
                hide_index=True,
                use_container_width=True,
                column_config={
                    'Status': st.column_config.SelectboxColumn('Status', options=status_choices),
                    'Due': st.column_config.DateColumn('Due date'),
                    'Project': st.column_config.TextColumn('Project', disabled=True),
                    'Milestone': st.column_config.TextColumn('Milestone', disabled=True),
                    'Task': st.column_config.TextColumn('Task', disabled=True),
                },
            )

            if st.button("💾 Save checklist updates", use_container_width=True):
                changes = 0
                for task_id, edited_row in edited_df.iterrows():
                    original_row = editor_df.loc[task_id]
                    has_changes = False
                    record = {
                        'task_id': task_id,
                        'Project': original_row['Project'],
                        'Milestone': original_row['Milestone'],
                        'Task': original_row['Task'],
                        'Owner': edited_row['Owner'],
                        'Status': edited_row['Status'],
                        'Due': edited_row['Due'],
                        'Notes': edited_row['Notes'],
                    }

                    if edited_row['Status'] != original_row['Status']:
                        has_changes = True
                    if (edited_row['Owner'] or '') != (original_row['Owner'] or ''):
                        has_changes = True
                    if pd.to_datetime(edited_row['Due']) != pd.to_datetime(original_row['Due']):
                        has_changes = True
                    if (edited_row['Notes'] or '') != (original_row['Notes'] or ''):
                        has_changes = True

                    if has_changes:
                        upsert_checklist_task(record)
                        changes += 1

                if changes:
                    refresh_checklist_session_state()
                    st.success(f"Updated {changes} checklist item{'s' if changes != 1 else ''}.")
                    st.rerun()
                else:
                    st.info("No checklist edits detected.")

            st.download_button(
                "⬇️ Export checklist",
                data=filtered_df.drop(columns=['Progress'], errors='ignore').to_csv(index=False),
                file_name="tapeout_checklist.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with st.expander("➕ Add checklist item"):
        with st.form("new_checklist_item"):
            project = st.text_input("Project")
            milestone = st.text_input("Milestone")
            task = st.text_area("Task")
            owner = st.text_input("Owner")
            status = st.selectbox("Status", CHECKLIST_STATUS_OPTIONS, index=0)
            due = st.date_input("Due date", value=datetime.now().date())
            notes = st.text_area("Notes", height=80)

            submitted = st.form_submit_button("Add item")
            if submitted:
                required = [project.strip(), milestone.strip(), task.strip()]
                if not all(required):
                    st.warning("Project, milestone, and task description are required.")
                else:
                    record = {
                        'task_id': generate_task_id(),
                        'Project': project.strip(),
                        'Milestone': milestone.strip(),
                        'Task': task.strip(),
                        'Owner': owner.strip(),
                        'Status': status,
                        'Due': due,
                        'Notes': notes.strip(),
                    }
                    upsert_checklist_task(record)
                    refresh_checklist_session_state()
                    st.success("Checklist item added.")
                    st.rerun()


def render_database_manager():
    st.header("🗄️ Database Manager")
    st.markdown(
        """
        > Inspect, back up, and maintain the persisted state that powers the Celestial AI CAD command center.
        """
    )

    if not os.path.exists(DATABASE_PATH):
        st.warning("Database not found—initializing with baseline records now.")
        init_database()

    tables, overview_df = get_database_overview()

    db_size_mb = os.path.getsize(DATABASE_PATH) / (1024 ** 2) if os.path.exists(DATABASE_PATH) else 0
    last_modified = (
        datetime.fromtimestamp(os.path.getmtime(DATABASE_PATH)).strftime("%Y-%m-%d %H:%M")
        if os.path.exists(DATABASE_PATH)
        else "N/A"
    )

    total_rows = int(overview_df["Rows"].sum()) if not overview_df.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("SQLite Size", f"{db_size_mb:.2f} MB")
    col2.metric("Tracked Tables", len(tables))
    col3.metric("Total Rows", f"{total_rows:,}")

    st.caption(f"Database located at: `{DATABASE_PATH}` (last modified {last_modified})")

    if overview_df.empty:
        st.info("No user tables detected yet—interact with the infrastructure tools to start populating the datastore.")
        return

    st.markdown("### Inventory")
    st.dataframe(overview_df, hide_index=True, use_container_width=True)

    st.markdown("### Table Explorer")
    selected_table = st.selectbox("Choose a table to inspect", tables, index=0)
    row_limit = st.slider("Rows to display", min_value=10, max_value=500, value=100, step=10)

    preview_df = fetch_table_preview(selected_table, limit=row_limit)

    if preview_df.empty:
        st.warning("This table is currently empty.")
    else:
        st.dataframe(preview_df, hide_index=True, use_container_width=True)

        csv_bytes = preview_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download preview as CSV",
            data=csv_bytes,
            file_name=f"{selected_table}_preview.csv",
            mime="text/csv",
        )

    with get_db_connection() as conn:
        schema_rows = conn.execute(f'PRAGMA table_info("{selected_table}")').fetchall()

    if schema_rows:
        schema_df = pd.DataFrame(schema_rows, columns=["cid", "Column", "Type", "Not Null", "Default", "Primary Key"])
        st.markdown("#### Schema")
        st.dataframe(schema_df[["Column", "Type", "Not Null", "Primary Key"]], hide_index=True)

    with st.expander("Maintenance & Utilities"):
        st.markdown("Keep the datastore lean and ensure application caches stay in sync after manual interventions.")

        maintenance_col, backup_col, refresh_col = st.columns(3)

        if maintenance_col.button("Run VACUUM + ANALYZE", use_container_width=True):
            perform_database_maintenance()
            st.success("Database maintenance completed successfully.")
            st.rerun()

        if os.path.exists(DATABASE_PATH):
            with open(DATABASE_PATH, "rb") as db_file:
                backup_col.download_button(
                    "Download SQLite backup",
                    data=db_file.read(),
                    file_name=f"celestial_ai_datastore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db",
                    mime="application/x-sqlite3",
                    use_container_width=True,
                )

        if refresh_col.button("Refresh session caches", use_container_width=True):
            refresh_license_session_state()
            refresh_bug_registry_session_state()
            refresh_tool_registry_session_state()
            refresh_run_history_session_state()
            refresh_checklist_session_state()
            st.success("Session state refreshed from the latest database contents.")
            st.rerun()

    with st.expander("Read-only SQL Workbench"):
        st.markdown("Run ad-hoc SELECT queries when you need deeper analytics without leaving the dashboard.")
        default_query = f'SELECT * FROM "{selected_table}" LIMIT 10'
        query = st.text_area("SQL", value=default_query, height=120)

        if st.button("Execute query", use_container_width=True):
            lowered = query.strip().lower()
            if not lowered.startswith("select") and not lowered.startswith("with"):
                st.error("Only read-only SELECT statements are allowed.")
            else:
                try:
                    with get_db_connection() as conn:
                        result_df = pd.read_sql_query(query, conn)
                    if result_df.empty:
                        st.info("Query executed successfully but returned no rows.")
                    else:
                        st.dataframe(result_df, hide_index=True, use_container_width=True)
                        st.download_button(
                            "⬇️ Download result as CSV",
                            data=result_df.to_csv(index=False).encode("utf-8"),
                            file_name="query_result.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Query failed: {exc}")

💻 CAD Command Center
The CAD Command Center is a Streamlit-based web application designed for ASIC/FPGA physical design engineers and CAD infrastructure managers. It provides unified dashboards for optimizing physical design trade-offs (Floorplanning) and monitoring high-value EDA tool licenses and known tool issues.

🚀 Features
This application is split into two main dashboards:

1. 🧱 Floorplan Strategy Analyzer
Netlist Simulation: Generates synthetic netlist data (blocks, connectivity, timing, area) for a target project (e.g., "AI Accelerator").

Strategy Comparison: Simulates and compares multiple floorplan strategies (Flat, Functional, Random, and optimized Connectivity-Driven partition via Louvain algorithm).

Multi-Objective Analysis: Visualizes trade-offs between critical metrics: Timing (Critical Path), Area Utilization, Power Proxy, and Routability Score (congestion risk).

Physical Visualization: Generates physical floorplan layouts and inter-module connectivity graphs.

Export: Allows export of metrics (CSV) and the final floorplan (DEF/TCL format proxy).

2. 📊 EDA Infrastructure Monitor
License Utilization: Monitors real-time license usage, highlighting critical shortages and displaying overall usage metrics.

Cost Efficiency: Calculates and highlights the most cost-inefficient tools (highest annual unused license cost).

Vendor Filtering: Allows filtering license metrics by EDA vendor (Synopsys, Cadence, etc.).

Bug Tracker: Provides a centralized, persistent registry for known EDA tool bugs, versions, workarounds, and status updates.

Tool Registry: Tracks approved, stable tool versions for specific projects to ensure design reproducibility.

⚙️ Installation and Local Setup
To run this application locally, you need Python 3.9+ and the required dependencies.

Prerequisites
Clone the Repository:

git clone [https://github.com/tinouchemassinissa/CelestialAi_app.git](https://github.com/tinouchemassinissa/CelestialAi_app.git)
cd CelestialAi_app

Create Virtual Environment (Recommended):

python -m venv venv
.\venv\Scripts\activate  # On Windows
# source venv/bin/activate # On Linux/macOS

Install Dependencies
Install all required libraries listed in requirements.txt:

pip install -r requirements.txt

The application relies on the following key libraries:

streamlit

pandas

numpy

plotly

networkx

python-louvain

Run the Application
Execute the main Streamlit file:

streamlit run streamlit_app.py

The application will automatically open in your browser at http://localhost:8501.

☁️ Cloud Deployment (AWS App Runner)
This application is designed for Continuous Delivery (CD) via AWS App Runner, which automatically deploys changes pushed to this GitHub repository.

Configuration Details
To configure the App Runner service, use the following settings:

Setting

Value

Rationale

Source

GitHub Repository

Uses automatic CI/CD trigger.

Runtime

Python 3

Required Python version.

Port

8080

Required port for the App Runner container environment.

Start Command

streamlit run streamlit_app.py --server.port 8080 --server.address 0.0.0.0

CORRECTED: Forces Streamlit to listen on the required port (8080) and all network interfaces.

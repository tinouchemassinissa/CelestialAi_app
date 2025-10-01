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

## ☁️ Cloud Deployment (AWS App Runner)

This application supports **Continuous Delivery (CD)** via AWS App Runner, automatically deploying updates when changes are pushed to GitHub.

### Configuration
| Setting       | Value                                                                                     | Notes                           |
|---------------|-------------------------------------------------------------------------------------------|---------------------------------|
| Source        | GitHub Repository                                                                         | Enables automatic CI/CD trigger |
| Runtime       | Python 3                                                                                  | Required version                |
| Port          | 8080                                                                                      | App Runner default              |
| Start Command | `streamlit run streamlit_app.py --server.port 8080 --server.address 0.0.0.0`              | Ensures correct binding         |

### Deploying GitHub updates to AWS

Once the App Runner service is connected to your GitHub repository, every push to the tracked branch (for example, `main` or `work`) can automatically redeploy the latest Streamlit build. Use the following workflow to keep AWS in sync:

1. **Commit and push your changes to GitHub.**
   ```bash
   git add .
   git commit -m "Describe your change"
   git push origin <branch>
   ```
2. **Verify the App Runner deployment.** In the AWS console, open **App Runner → Services → [Your Service] → Deployments** to confirm a new deployment started from the recent GitHub push. The service automatically builds the new container image and rolls it out when the health checks pass.
3. **Trigger a manual redeploy (optional).** If automatic deployments are disabled, choose **Deploy → Deploy latest commit** within the service console to force a redeploy from the most recent GitHub revision.
4. **Monitor application logs.** Use the **Logs** tab in App Runner or stream them via CloudWatch Logs to ensure the Streamlit server starts successfully with `streamlit_app.py` bound to port `8080`.

> 💡 **Tip:** For more advanced workflows (staging vs. production), create separate App Runner services pointing to dedicated branches or wire the repository into AWS CodePipeline/CodeBuild to run automated tests before App Runner receives the artifact.

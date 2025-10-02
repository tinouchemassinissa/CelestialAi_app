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

### Post-connection validation checklist

After wiring your GitHub repository to AWS App Runner, run through the checks below to confirm the integration is healthy before relying on automated deployments:

1. **Confirm the webhook handshake.** In App Runner, open your service, navigate to **Source → Repository details**, and ensure the connection status reads `Connected`. If it shows an error, click **Reconnect** to re-authorize GitHub.
2. **Validate the default deployment.** Trigger a manual deploy of the latest commit and wait for the status to become `Running`. This proves the build container and Streamlit start command are accepted by App Runner.
3. **Inspect environment variables and secrets.** Verify `Settings → Environment variables` contains the values the app expects (license paths, feature flags, etc.) so database features work once traffic hits the service.
4. **Smoke test the live endpoint.** Visit the App Runner default domain (or your custom domain) and navigate through the CAD dashboards to make sure database-backed tabs load without errors.
5. **Review CloudWatch metrics.** Use **Monitoring → CloudWatch alarms/metrics** to confirm the service is sending health data. Set up an alarm on `5XX` errors so future deploy issues alert your team quickly.

Documenting the results of this checklist in your deployment runbook makes it easier for CAD engineers to troubleshoot if automated releases stop flowing from GitHub to AWS.

## EC2 Deployment via GitHub Actions

This repository ships with a GitHub Actions workflow that deploys the Streamlit service to your EC2 host whenever commits land on the `main` branch.

### Required GitHub Secrets
| Secret | Description |
| ------ | ----------- |
| `EC2_HOST` | Public DNS name or IP address of the EC2 instance running the Streamlit service. |
| `EC2_USER` | SSH username (for example, `ubuntu` or `ec2-user`). |
| `EC2_SSH_KEY` | Private SSH key with permission to pull from GitHub and manage `/opt/CelestialAi_app`. |

> Optional: add `EC2_PORT` if your SSH daemon listens on a non-standard port and update the workflow accordingly.

### Deployment Flow
1. **Provision the EC2 host.** Install Git, Python 3, and ensure `/opt/CelestialAi_app` contains this repository with a `deploy.sh` script that restarts `streamlit.service` after pulling new code.
2. **Authorize the SSH key.** Append the public key that pairs with `EC2_SSH_KEY` to the EC2 instance's `~/.ssh/authorized_keys` file and restrict permissions to the deployment user.
3. **Add GitHub secrets.** In your repository, navigate to **Settings → Secrets and variables → Actions**, then create the secrets listed above with the appropriate values.
4. **Push to `main`.** Every push to `main` triggers `.github/workflows/deploy.yml`, which SSHes into the EC2 host, refreshes dependencies with `requirements.txt`, runs `/opt/CelestialAi_app/deploy.sh`, and restarts `streamlit.service`.
5. **Verify the deployment.** Tail `/var/log/syslog` or `journalctl -u streamlit.service` on the EC2 instance to ensure the Streamlit app restarted cleanly, and open the public URL to confirm the latest commit banner appears in the UI.

If a deployment fails, check the GitHub Actions run logs for SSH errors and ensure the EC2 host can reach GitHub over HTTPS to fetch repository updates.

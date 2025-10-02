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
4. **Push to `main`.** Every push to `main` triggers `.github/workflows/deploy.yml`, which SSHes into the EC2 host, **syncs the repository**, installs `requirements.txt`, runs `/opt/CelestialAi_app/deploy.sh`, and restarts `streamlit.service`.
5. **Verify the deployment.** Tail `/var/log/syslog` or `journalctl -u streamlit.service` on the EC2 instance to ensure the Streamlit app restarted cleanly, and open the public URL to confirm the latest commit banner appears in the UI.

If a deployment fails, check the GitHub Actions run logs for SSH errors and ensure the EC2 host can reach GitHub over HTTPS to fetch repository updates.
#!/usr/bin/env bash
# Start (or attach to) the NVIDIA CloudXR runtime + WSS proxy from the isaacteleop pip package.
# Usage: scripts/cloudxr_service.sh [--accept-eula] [--host-client]   (keep this terminal open; it prints a cloudxr.env to source)
# First run downloads the CloudXR Web Client SDK to ~/.cloudxr and asks you to accept the NVIDIA CloudXR EULA.
# Headset: open https://nvidia.github.io/IsaacTeleop/client/release-1.3.x in the Quest browser, enter this PC's IP
# (<PC_IP>), accept the self-signed cert (https://<PC_IP>:48322/), Connect.
# Ports (LAN): UDP 47998, TCP 49100, TCP 48322.
set -euo pipefail
QWS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWS"
exec .venv_cloudxr/bin/python -m isaacteleop.cloudxr "$@"

#!/usr/bin/env bash
# CloudXR path: .venv_cloudxr = a venv layered on the Isaac Lab env (--system-site-packages) holding NVIDIA's
# isaacteleop package, which ships the CloudXR runtime + WSS proxy (scripts/cloudxr_service.sh runs it).
# The Isaac Lab env itself stays untouched.
set -euo pipefail
QWS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWS"
source "$QWS/scripts/_isaac_env.sh"
python -m venv --system-site-packages .venv_cloudxr
.venv_cloudxr/bin/pip install "isaacteleop[cloudxr]==1.3.131"
.venv_cloudxr/bin/python -c "import isaacteleop; print('isaacteleop ok')"
echo "[setup] done: $QWS/.venv_cloudxr  (next: scripts/cloudxr_service.sh)"

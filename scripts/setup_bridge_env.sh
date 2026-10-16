#!/usr/bin/env bash
# Browser (WebXR) path only: creates .venv (host python3 >= 3.10) with the ROBOTIS vuer fork + zmq/msgpack,
# and a TLS cert/key in certs/ for this PC's LAN IP (the Quest browser needs https/wss for WebXR).
# Usage: scripts/setup_bridge_env.sh [LAN_IP]     (default: the first address of `hostname -I`)
set -euo pipefail
QWS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWS"
IP="${1:-$(hostname -I | awk '{print $1}')}"
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip wheel
pip install -r requirements-bridge.txt
python -c "import vuer, zmq, msgpack; print('vuer', getattr(vuer, '__version__', '?'), 'ok')"
mkdir -p certs
if [ ! -s certs/key.pem ]; then
  if command -v mkcert >/dev/null; then
    mkcert -cert-file certs/cert.pem -key-file certs/key.pem "$IP" localhost 127.0.0.1
  else
    openssl req -x509 -newkey rsa:2048 -nodes -days 825 -subj "/CN=$IP" \
      -addext "subjectAltName=IP:$IP,DNS:localhost,IP:127.0.0.1" \
      -keyout certs/key.pem -out certs/cert.pem
  fi
  chmod 600 certs/key.pem
  echo "[setup] TLS cert/key for $IP written to certs/ (accept it once in the Quest browser)"
fi
echo "[setup] done: $QWS/.venv"

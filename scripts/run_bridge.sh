#!/usr/bin/env bash
# Usage: scripts/run_bridge.sh [extra args for vuer_server.py]
set -euo pipefail
QWS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWS"
. .venv/bin/activate
exec python -m quest_bridge.vuer_server "$@"

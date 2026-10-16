#!/usr/bin/env bash
# Fetch the release v1.0 files:
#   sim_assets.tar.gz                  -> sim/assets/  (robot, table, object USDs for the bundled handover env)
#   vr_demos_handover_20260821.tar.gz  -> demos/vr_cloudxr/handover_tshape/20260821_185201/  (6 recorded episodes)
# Usage: scripts/download_assets.sh [sim|demos|all]   (default: all)
set -euo pipefail
QWS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="https://github.com/ismael-esp/quest3-dexterous-teleop/releases/download/v1.0"
WHAT="${1:-all}"
fetch() {  # fetch <file> <dest dir>
  mkdir -p "$2"
  echo "[download] $1 -> $2"
  curl -fL --retry 3 "$URL/$1" | tar xz -C "$2"
}
if [ "$WHAT" = sim ] || [ "$WHAT" = all ]; then
  fetch sim_assets.tar.gz "$QWS/sim"
fi
if [ "$WHAT" = demos ] || [ "$WHAT" = all ]; then
  fetch vr_demos_handover_20260821.tar.gz "$QWS/demos/vr_cloudxr"
fi
echo "[download] done"

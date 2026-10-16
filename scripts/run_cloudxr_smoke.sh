#!/usr/bin/env bash
# Usage: [GPU=<n>] [DEVICE=cpu] scripts/run_cloudxr_smoke.sh [--seconds 120 --anchor_pos 0 0.05 0 --anchor_yaw 180 ...]
# Native XR render of the thesis env into the Quest 3 via the CloudXR runtime (start scripts/cloudxr_service.sh first).
# Runs headless (the view is in the headset). Default: no GPU pinning -> same GPU as the CloudXR runtime (GPU 0).
set -euo pipefail
QWS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$QWS/scripts/_isaac_env.sh"   # conda env with Isaac Lab (ISAACLAB_ENV, default env_isaaclab)
cd "$QWS"
EXTRA=()
if [ -n "${GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
  EXTRA=(--device "${DEVICE:-cuda:0}" --kit_args "--/renderer/activeGpu=$GPU --/renderer/multiGpu/enabled=false")
elif [ -n "${DEVICE:-}" ]; then
  EXTRA=(--device "$DEVICE")
fi
exec python isaac_teleop/cloudxr_render_smoke.py --headless "${EXTRA[@]}" "$@"

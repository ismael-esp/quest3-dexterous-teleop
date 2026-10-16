#!/usr/bin/env bash
# Usage: [GPU=1] [DEVICE=cpu] scripts/run_teleop.sh [--fake_vr] [--vr_origin 0 1.2 0] ...   (adds --headless --enable_cameras)
# DEVICE=cpu runs PhysX on the CPU (Isaac Lab's recommendation for single-env teleop; ~30% faster loop here).
# GPU=<n> pins BOTH physics (CUDA) and the RTX renderer (Vulkan) to physical GPU n — needed when another
# GPU is busy (e.g. a thesis training run on GPU 0), otherwise the renderer picks GPU 0 and OOMs.
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
exec python isaac_teleop/teleop_handover_vr.py --headless --enable_cameras "${EXTRA[@]}" "$@"

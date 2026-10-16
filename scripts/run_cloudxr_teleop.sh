#!/usr/bin/env bash
# Usage: [GPU=<n>] [DEVICE=cpu] scripts/run_cloudxr_teleop.sh [--robot_opacity 0.5 --anchor_pos 0 0.05 0 ...]
# Full VR teleop via the native Kit XR render + CloudXR (start scripts/cloudxr_service.sh first).
# Keys: A=start demo | F=finish | S=save OK | N=save FAIL | D=discard | SPACE=engage | 1/2=per-arm | C=re-center | K=view hold | R=reset | Q=quit
set -euo pipefail
QWS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$QWS/scripts/_isaac_env.sh"   # conda env with Isaac Lab (ISAACLAB_ENV, default env_isaaclab)
cd "$QWS"
# async rendering can sample link transforms mid-physics-update (torn articulations) -- off by default;
# ASYNC=1 re-enables it if XR frame pacing suffers.
if [ -n "${ASYNC:-}" ]; then
  KIT="--/app/asyncRendering=true --/app/asyncRenderingLowLatency=true --/rtx/raytracing/fractionalCutoutOpacity=true"
else
  KIT="--/app/asyncRendering=false --/app/asyncRenderingLowLatency=false --/rtx/raytracing/fractionalCutoutOpacity=true"
fi
# KIT_EXTRA appends extra Kit settings without clobbering the ones above (argparse keeps only the
# last --kit_args), e.g. KIT_EXTRA="--/log/channels/omni.kit.xr.core.plugin=verbose" for XR debugging.
KIT="$KIT ${KIT_EXTRA:-}"
EXTRA=(--kit_args "$KIT")
if [ -n "${GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
  EXTRA=(--device "${DEVICE:-cuda:0}" --kit_args "$KIT --/renderer/activeGpu=$GPU --/renderer/multiGpu/enabled=false")
elif [ -n "${DEVICE:-}" ]; then
  EXTRA=(--device "$DEVICE" --kit_args "$KIT")
fi
# Isaac's Ctrl-C handler can kill the process before Python restores the terminal -- always re-sane it.
trap 'stty sane 2>/dev/null || true' EXIT
python isaac_teleop/teleop_handover_cloudxr.py --headless "${EXTRA[@]}" "$@"

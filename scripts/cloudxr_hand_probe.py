"""Phase-0 probe: Quest 3 head + hand tracking through IsaacTeleop / CloudXR (no Isaac Sim involved).

Terminal 1:  scripts/cloudxr_service.sh            # CloudXR runtime + WSS proxy (EULA prompt on first run); keep open
Terminal 2:  .venv_cloudxr/bin/python scripts/cloudxr_hand_probe.py [--seconds 60]
             (attaches to the running runtime by loading ~/.cloudxr/run/cloudxr.env; --launch starts one in-process)
Headset:     CloudXR.js client in the Quest browser -> enter this PC's IP -> accept the cert -> Connect.
Prints the raw head / hand outputs at 2 Hz. Success = values change when you move your hands.
"""
from __future__ import annotations

import argparse
import os
import time

from isaacteleop.cloudxr import CloudXRLauncher
from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource, HeadSource
from isaacteleop.retargeting_engine.interface import OutputCombiner
from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig


def _load_cloudxr_env(path: str) -> None:
    """Equivalent of `source ~/.cloudxr/run/cloudxr.env` (written by `python -m isaacteleop.cloudxr`)."""
    if os.environ.get("XR_RUNTIME_JSON"):
        return
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: start the runtime first (scripts/cloudxr_service.sh) or pass --launch")
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    print(f"[probe] loaded {path}: XR_RUNTIME_JSON={os.environ.get('XR_RUNTIME_JSON')}", flush=True)


def _brief(v):
    """OptionalTensorGroup -> short text: absent, or wrist/head position + #valid joints.
    Fields are addressed by index (names in group_type.types): hands = [positions(26,3), orientations,
    radii, valid]; head = [position, orientation, is_valid]."""
    import numpy as np
    try:
        if v.is_none:
            return "absent"
        names = [getattr(t, "name", str(i)) for i, t in enumerate(v.group_type.types)]
        vals = {n: v[i] for i, n in enumerate(names)}
        if "hand_joint_positions" in vals:
            a = np.asarray(vals["hand_joint_positions"]).reshape(-1, 3)
            valid = np.asarray(vals.get("hand_joint_valid", [])).ravel()
            return f"wrist={np.round(a[1], 3).tolist()} valid={int(valid.sum()) if valid.size else '?'}/{a.shape[0]}"
        if "head_position" in vals:
            return f"pos={np.round(np.asarray(vals['head_position']).ravel()[:3], 3).tolist()} valid={vals.get('head_is_valid')}"
        return ", ".join(f"{n}={str(vals[n])[:20]}" for n in names)[:120]
    except Exception as e:
        return f"{type(v).__name__}({e!r})"[:100]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--launch", action="store_true", help="start the CloudXR runtime in-process instead of attaching")
    ap.add_argument("--accept-eula", action="store_true", help="[--launch] accept the NVIDIA CloudXR EULA non-interactively")
    ap.add_argument("--cloudxr-install-dir", default="~/.cloudxr")
    args = ap.parse_args()
    if not args.launch:
        _load_cloudxr_env(os.path.join(os.path.expanduser(args.cloudxr_install_dir), "run", "cloudxr.env"))

    hands = HandsSource(name="hands")
    head = HeadSource(name="head")
    outs = {"left": hands.output(HandsSource.LEFT), "right": hands.output(HandsSource.RIGHT)}
    try:
        outs["head"] = head.output(list(head.output_spec().keys())[0])
    except Exception as e:  # head source API may differ; hands are the important part
        print(f"[probe] head source not wired: {e!r}")
    pipeline = OutputCombiner(outs)

    import contextlib
    # isaacteleop 1.3.x: CloudXRLauncher starts runtime + WSS proxy on construction (EULA prompt on first run)
    ctx = CloudXRLauncher(install_dir=args.cloudxr_install_dir, accept_eula=args.accept_eula) if args.launch else contextlib.nullcontext()
    with ctx:
        cfg = TeleopSessionConfig(app_name="Quest3HandProbe", trackers=[], pipeline=pipeline)
        # A clientless CloudXR runtime advertises no HMD -> xrGetSystem fails with -35
        # (XR_ERROR_FORM_FACTOR_UNAVAILABLE) until the headset pressed Connect. Poll until it does.
        session_cm = None
        t_wait = time.time()
        while session_cm is None:
            try:
                session_cm = TeleopSession(cfg)
                session = session_cm.__enter__()
            except RuntimeError as e:
                if "-35" not in str(e) or time.time() - t_wait > 600:
                    raise
                session_cm = None
                print("[probe] no headset yet (OpenXR -35): press Connect in the CloudXR.js client ... retrying in 2 s", flush=True)
                time.sleep(2.0)
        try:
            t0 = time.time(); last = 0.0
            print("[probe] session up -- headset connected", flush=True)
            while time.time() - t0 < args.seconds:
                res = session.step()
                now = time.time()
                if now - last >= 0.5:
                    last = now
                    print(f"[probe] t={now - t0:5.1f}s " + " | ".join(f"{k}: {_brief(v)}" for k, v in res.items()), flush=True)
                time.sleep(0.005)
        finally:
            session_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()

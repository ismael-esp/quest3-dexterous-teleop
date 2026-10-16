"""Phase 1 (quest3-cloudxr): render the thesis handover_tshape env NATIVELY into the Quest 3 via
Isaac Lab 2.3 `--xr` (Kit OpenXR) against the CloudXR runtime started by `scripts/cloudxr_service.sh`.

Terminal 1:  scripts/cloudxr_service.sh                        (CloudXR runtime, keep open)
Terminal 2:  scripts/run_cloudxr_smoke.sh [--seconds 120] [--anchor_pos 0 0.05 0] [--anchor_yaw 180]
Headset:     CloudXR.js client -> Connect (can be before or after this script starts).

What it does: loads ~/.cloudxr/run/cloudxr.env (XR_RUNTIME_JSON), launches Isaac Lab with the XR
experience, selects the *system* OpenXR runtime, builds the env (1 env, full-hand action), places the
XR anchor (where the headset's floor origin lands in the cell), enables the AR profile, and steps the
sim with zero actions while printing FPS and the tracked head / wrist positions (Isaac world frame)
from `OpenXRDevice`. No control yet -- this is the go/no-go for architecture A.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

QWS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if QWS not in sys.path:
    sys.path.insert(0, QWS)
from isaac_teleop.paths import sim_root  # noqa: E402

TWS = sim_root()  # the handover env: $THESIS_WS, ../thesis_ws, or the bundled ./sim
for p in (QWS, TWS):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_cloudxr_env(path: str) -> None:
    if os.environ.get("XR_RUNTIME_JSON"):
        return
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: start the CloudXR runtime first (scripts/cloudxr_service.sh)")
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="handover_tshape")
parser.add_argument("--seconds", type=float, default=0.0, help="run time in s (0 = until Q / Ctrl-C)")
parser.add_argument("--anchor_pos", type=float, nargs=3, default=[0.0, 0.05, 0.0],
                    help="Isaac-world position where the headset's floor origin lands (x y z); default = under the "
                         "robot head / between the arm mounts")
parser.add_argument("--foveation", choices=["none", "warped"], default="none",
                    help="Kit XR foveation mode; must match the CloudXR runtime's PQW submit mode")
parser.add_argument("--anchor_yaw", type=float, default=180.0,
                    help="yaw [deg] of the XR frame about world z. 180 -> OpenXR -z (forward) = world +y (facing the "
                         "table); adjust if the table is behind you")
parser.add_argument("--cloudxr_env", default=os.path.expanduser("~/.cloudxr/run/cloudxr.env"))
parser.add_argument("--print_every", type=float, default=1.0)
parser.add_argument("--no_enable_profile", action="store_true",
                    help="do not call XRCore.request_enable_profile('ar') (the headless XR experience may auto-start it)")
_load_cloudxr_env(parser.parse_known_args()[0].cloudxr_env)

# pinocchio (dex_retargeting) MUST be imported before AppLauncher (see teleop_handover_vr.py)
import pinocchio  # noqa: E402,F401

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.xr = True          # -> isaaclab.python.xr.openxr[.headless].kit experience
# Offscreen rendering (enable_cameras AND headless) makes IsaacLab switch off the active viewport,
# which is exactly the surface the XR compositor renders the headset image through -> black HMD with
# "Viewport does not have the right aov". The arg alone does not close the door: AppLauncher falls
# back to the ENABLE_CAMERAS env var whenever the arg is False, so pin that too.
os.environ["ENABLE_CAMERAS"] = "0"
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print(f"[xr] render flags: enable_cameras={getattr(app_launcher, '_enable_cameras', '?')} "
      f"offscreen_render={getattr(app_launcher, '_offscreen_render', '?')}", flush=True)

import math  # noqa: E402

import carb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

os.chdir(TWS)  # env cfgs use BASE_DIR="" -> assets/... relative to the env root
from task_registry import make_env  # noqa: E402

from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg, XrCfg  # noqa: E402

from isaac_teleop.keyboard import TerminalKeys  # noqa: E402


def main() -> None:
    settings = carb.settings.get_settings()
    # Use the runtime pointed to by XR_RUNTIME_JSON (CloudXR from isaacteleop), not Kit's bundled one.
    settings.set_string("/persistent/xr/system/openxr/runtime", "system")
    settings.set_string("/persistent/xr/profile/ar/system/display", "OpenXR")
    print(f"[xr] XR_RUNTIME_JSON={os.environ.get('XR_RUNTIME_JSON')}", flush=True)
    print(f"[xr] app.xr.enabled={settings.get('/app/xr/enabled')} runtime={settings.get('/persistent/xr/system/openxr/runtime')}", flush=True)

    steps_per_s = 60.0
    overrides = {"handover_full_hand_action": True, "handover_rand_obs_noise": True, "handover_rand_pd_gains": True,
                 "handover_domain_rand": False, "episode_length": 7200, "episode_length_s": 120.0}
    env, raw, env_cfg = make_env(args_cli.task, 1, device=str(args_cli.device), env_cfg_overrides=overrides)
    if hasattr(raw.cfg, "terminate_on_success"):
        raw.cfg.terminate_on_success = False
    raw.cfg.sim.render_interval = int(raw.cfg.decimation)      # render every control step (XR wants every frame)
    env.reset()
    print(f"[xr] env ready: act={env.num_actions} sim device={raw.device}", flush=True)
    # the XR compositor renders through the active viewport; IsaacLab may have just disabled it
    try:
        from omni.kit.viewport.utility import get_active_viewport
        _vp = get_active_viewport()
        if _vp is not None and not bool(getattr(_vp, "updates_enabled", True)):
            _vp.updates_enabled = True
            print("[xr] active viewport updates were OFF -> re-enabled (this is what blanks the HMD)", flush=True)
    except Exception as _exc:
        print(f"[xr] viewport check unavailable: {_exc}", flush=True)

    # XR anchor: the headset's floor origin appears at anchor_pos with yaw anchor_yaw about world z.
    yaw = math.radians(args_cli.anchor_yaw)
    anchor_rot = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))      # (w, x, y, z)
    env_origin = raw.env_origins[0].detach().cpu().numpy().astype(float)
    anchor_pos = tuple((env_origin + np.asarray(args_cli.anchor_pos, dtype=float)).tolist())
    xr_cfg = XrCfg(anchor_pos=anchor_pos, anchor_rot=anchor_rot, near_plane=0.05)
    device = OpenXRDevice(OpenXRDeviceCfg(xr_cfg=xr_cfg))
    print(f"[xr] anchor pos={np.round(anchor_pos, 3).tolist()} yaw={args_cli.anchor_yaw} deg", flush=True)

    if not args_cli.no_enable_profile:
        from omni.kit.xr.core import XRCore
        core = XRCore.get_singleton()
        try:
            cur = core.get_current_xr_profile_name()
        except Exception:
            cur = "?"
        print(f"[xr] current profile: {cur!r}; requesting 'ar' ...", flush=True)
        # AR profile defaults to foveation "warped"; this runtime submits unwarped -> black HMD
        # ("XRComposition (warped) input resource is null"). Keep the two in sync.
        _xr_settings = carb.settings.get_settings()
        # Kit reads profile settings from /xr/..., /persistent/xr/... or /defaults/xr/... depending on
        # how the profile was loaded (the AR experience uses persistent.* for renderQuality), so set
        # all three and print the readback rather than guessing which one wins.
        _fov_keys = ("/xr/profile/ar/foveation/mode",
                     "/persistent/xr/profile/ar/foveation/mode",
                     "/defaults/xr/profile/ar/foveation/mode")
        for _key in _fov_keys:
            _xr_settings.set(_key, str(args_cli.foveation))
        print(f"[xr] foveation mode -> {args_cli.foveation} | "
              + " ".join(f"{k}={_xr_settings.get(k)!r}" for k in _fov_keys), flush=True)
        core.request_enable_profile("ar")

    act = torch.zeros((2, 1, env.num_actions), device=raw.device)
    t0 = time.time(); t_print = t0; n = 0; n_print = 0
    print("[xr] running -- put the headset on. Keys (this terminal): Q = quit, R = reset the env", flush=True)
    try:
      with TerminalKeys() as keys:
        while simulation_app.is_running() and (args_cli.seconds <= 0 or time.time() - t0 < args_cli.seconds):
            k = keys.poll()
            if k and k.lower() == "q":
                break
            if k and k.lower() == "r":
                env.reset(); print("[xr] env reset", flush=True)
            env.step(act, n)
            n += 1; n_print += 1
            now = time.time()
            if now - t_print >= args_cli.print_every:
                fps = n_print / (now - t_print); t_print = now; n_print = 0
                try:
                    from omni.kit.xr.core import XRCore
                    prof = XRCore.get_singleton().get_current_xr_profile_name()
                except Exception:
                    prof = "?"
                try:
                    rawd = device._get_raw_data()
                    head = rawd[OpenXRDevice.TrackingTarget.HEAD]
                    lw = rawd[OpenXRDevice.TrackingTarget.HAND_LEFT].get("wrist")
                    rw = rawd[OpenXRDevice.TrackingTarget.HAND_RIGHT].get("wrist")
                    txt = (f"head={np.round(head[:3], 2).tolist()} "
                           f"Lwrist={None if lw is None else np.round(lw[:3], 2).tolist()} "
                           f"Rwrist={None if rw is None else np.round(rw[:3], 2).tolist()}")
                except Exception as e:
                    txt = f"tracking n/a ({e!r})"
                print(f"[xr] {now - t0:6.1f}s  {fps:5.1f} steps/s  profile={prof!r}  {txt}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            from omni.kit.xr.core import XRCore
            XRCore.get_singleton().request_disable_profile()
        except Exception:
            pass
        env.close()
        simulation_app.close()
        os._exit(0)


if __name__ == "__main__":
    main()

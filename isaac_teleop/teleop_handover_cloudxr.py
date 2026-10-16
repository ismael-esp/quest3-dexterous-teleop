"""Full VR teleoperation of the dual-arm handover_tshape env through the
NATIVE Kit XR render + CloudXR, recording demos in the same pickle format as the Vuer bridge.

Terminal 1:  scripts/cloudxr_service.sh                      (CloudXR runtime, keep open)
Terminal 2:  DEVICE=cpu scripts/run_cloudxr_teleop.sh [--robot_opacity 0.5 --anchor_pos 0 0.05 0 ...]
Headset:     CloudXR.js client -> Connect; put the controllers down for hand tracking.

Differences from the bridge teleop: the scene is rendered natively (walk / look around freely); hand and
head poses arrive from Isaac Lab's OpenXRDevice already in Isaac world frame (the XR anchor replaces the
old --embody/T_world_vr); the robot is drawn semi-transparent and the T-shaped mount frame is hidden
(visual only -- physics unchanged); K toggles VIEW HOLD (the world follows you, so your viewpoint stays
put while you walk); while held, the arrow keys glide the viewpoint (up/down = forward/back along your
facing, left/right = strafe) and O/L move it up/down.

Demo collection: the env starts in a FREE phase (nothing recorded -- set up the view, put the headset
on, test the arms). A resets the env and starts a demo (episode data + front-view mp4 together);
F finishes it and PAUSES the sim for the verdict: S saves ep_XXX_{success,fail}.pickle + the matching
.mp4, D discards both; the env then resets back to FREE for the next A. 3 records a standalone mp4
outside demos (2K 60fps by default; frames are captured on the wall clock so the video plays at
real-life speed even though the sim runs slightly faster than real time).
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
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="handover_tshape")
parser.add_argument("--max_episode_seconds", type=float, default=40.0)
parser.add_argument("--object", default="rod",
                    help="object catalog id (objects/catalog.py of the env): rod | rod_slim | tumbler")
parser.add_argument("--session_note", default="", help="free text stored in session.json (operator, setup, ...)")
parser.add_argument("--steps", type=int, default=0, help="stop after N steps (0 = until Q); used by tests")
parser.add_argument("--auto_engage", action="store_true", help="engage both arms as soon as tracking is present")
parser.add_argument("--anchor_pos", type=float, nargs=3, default=[0.0, 0.05, 0.0],
                    help="Isaac-world floor point where the headset's local origin lands")
parser.add_argument("--anchor_yaw", type=float, default=180.0, help="180 = facing the table (+y)")
parser.add_argument("--foveation", choices=["none", "warped"], default="none",
                    help="Kit XR foveation mode. MUST match the CloudXR runtime's PQW submit mode; 'warped' with an unwarped-submit runtime gives a black headset (see notes at request_enable_profile).")
parser.add_argument("--eye_pos", type=float, nargs=3, default=[0.0, 0.05, 1.65],
                    help="[embody] env-local position your eyes are placed at on start / key C "
                         "(default = the robot's head, between the arm mounts)")
parser.add_argument("--eye_yaw", type=float, default=90.0,
                    help="[embody] world heading [deg] you face after centering (90 = +y = toward the table)")
parser.add_argument("--no_auto_center", action="store_true",
                    help="do not auto-center on the first head pose (key C still works)")
parser.add_argument("--robot_opacity", type=float, default=1.0,
                    help="1.0 (default) = normal robot colors; <1 draws both robots see-through (visual only; the "
                         "dithered look was judged worse than normal colors live 2026-08-20)")
parser.add_argument("--ghost_mode", choices=["cutout", "translucent"], default="cutout",
                    help="cutout = dithered transparency that writes depth (stable under CloudXR reprojection); "
                         "translucent = true alpha (smoother look but smears/detaches while moving)")
parser.add_argument("--sim_steps_per_frame", type=int, default=2,
                    help="physics control steps per rendered XR frame; 2 brings the sim from ~0.5x to ~1x real time "
                         "(recorded data stays 60 Hz sim-time)")
parser.add_argument("--show_frame", action="store_true",
                    help="keep the T-shaped mount frame visible (hidden by default; physics unchanged)")
parser.add_argument("--pos_scale", type=float, default=1.0)
parser.add_argument("--max_ee_speed", type=float, default=6.0,
                    help="max EE target advance per second (with --clamp_pos sets the IK lead; see bridge README)")
parser.add_argument("--clamp_pos", type=float, default=0.10)
parser.add_argument("--clamp_rot", type=float, default=0.50)
parser.add_argument("--rot_weight", type=float, default=1.0)
parser.add_argument("--ff_tau", type=float, default=0.10,
                    help="velocity feed-forward horizon [s] on the EE target (0 = off)")
parser.add_argument("--hand_alpha", type=float, default=0.5, help="EMA on retargeted hand joints (1=no smoothing)")
parser.add_argument("--hand_profile", choices=["chain", "dexpilot"], default="chain",
                    help="chain (default) tracks every finger joint (full fist closes); dexpilot = fingertips only")
parser.add_argument("--finger_gain", type=float, default=1.15,
                    help="flexion gain about the open limit after retargeting (1.0 = off; fingers only)")
parser.add_argument("--thumb_mode", choices=["angle", "ik"], default="angle",
                    help="angle (default) maps the human thumb joint-to-joint (opposition->j12, abduction->j13, "
                         "MCP/IP flexion->j14/j15) -- thumb extension works; ik = optimizer only")
parser.add_argument("--view_comp", choices=["snap90", "continuous", "off"], default="snap90",
                    help="rotate the hand->EE mapping by your facing direction at engage time so controls stay "
                         "intuitive from the front/side (snap90 = nearest 90 deg; off = raw world frame)")
parser.add_argument("--hand_follow_clutch", action="store_true", help="only retarget fingers while the arm is engaged")
parser.add_argument("--max_joint_speed", type=float, default=3.1416,
                    help="per-joint rate limit on the commanded UR arm targets [rad/s] (UR3 spec: 180 deg/s\n"
                         "on the large joints); keeps recorded demos hardware-safe. 0 disables")
parser.add_argument("--nudge_step", type=float, default=0.05,
                    help="meters the held view moves per arrow / O / L key press (key-repeat glides)")
parser.add_argument("--video_res", default="2560x1440", help="front-view recording resolution")
parser.add_argument("--video_capture_fps", type=float, default=30.0,
                    help="wall-clock capture rate: the video plays at real-life speed even though the sim\n"
                         "runs faster than real time (XR frame pacing)")
parser.add_argument("--video_fps", type=float, default=60.0, help="mp4 playback fps")
parser.add_argument("--video_cam_pos", type=float, nargs=3, default=[0.0, 1.9, 1.5],
                    help="video camera position (env frame), in front of the robots looking back")
parser.add_argument("--video_cam_target", type=float, nargs=3, default=[0.0, 0.25, 1.1],
                    help="point the video camera looks at (env frame)")
parser.add_argument("--video_encoder", default="h264_nvenc", help="ffmpeg encoder (falls back to libx264)")
parser.add_argument("--video_autostart", action="store_true",
                    help="start recording immediately and save on exit (used by tests)")
parser.add_argument("--no_episode_video", action="store_true",
                    help="do not record the front-view mp4 alongside each demo episode")
parser.add_argument("--out_dir", default=os.path.join(QWS, "demos", "vr_cloudxr"))
parser.add_argument("--session_tag", default=time.strftime("%Y%m%d_%H%M%S"))
parser.add_argument("--require_success", action="store_true")
parser.add_argument("--domain_rand", action="store_true", default=False)
parser.add_argument("--cloudxr_env", default=os.path.expanduser("~/.cloudxr/run/cloudxr.env"))
parser.add_argument("--profile", action="store_true")
parser.add_argument("--no_status_cue", action="store_true",
                    help="disable the color cube above the table that mirrors the teleop state in-headset")
_load_cloudxr_env(parser.parse_known_args()[0].cloudxr_env)

# pinocchio (dex_retargeting) MUST be imported before AppLauncher
import pinocchio  # noqa: E402,F401
import dex_retargeting  # noqa: E402,F401

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.xr = True
# The XR compositor sources the headset image from the ACTIVE VIEWPORT, and IsaacLab switches that
# viewport off when offscreen rendering is on (enable_cameras AND headless, simulation_context.py
# L218). That shows up as "[omni.kit.xr.core.plugin] Viewport does not have the right aov" followed
# by "XRComposition (warped) input resource is null" every frame -- black HMD while tracking and the
# sim keep running fine (first fixed in 4568cb2).
# Setting the ARG to False is not enough on its own: AppLauncher only honours `enable_cameras` when
# it is True, and otherwise falls back to the ENABLE_CAMERAS env var (app_launcher.py
# _resolve_render_settings), so a stray `ENABLE_CAMERAS=1` exported in the shell silently turns
# offscreen rendering back on. Pin the env var as well.
os.environ["ENABLE_CAMERAS"] = "0"
args_cli.enable_cameras = False     # the key-3 video render product renders independently of this.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print(f"[xr] render flags: enable_cameras={getattr(app_launcher, '_enable_cameras', '?')} "
      f"offscreen_render={getattr(app_launcher, '_offscreen_render', '?')} "
      f"xr={getattr(app_launcher, '_xr', '?')} headless={getattr(app_launcher, '_headless', '?')}", flush=True)

import math  # noqa: E402

import carb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

os.chdir(TWS)  # env cfgs use BASE_DIR="" -> assets/... relative to the env root
from task_registry import make_env  # noqa: E402

from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg, XrCfg  # noqa: E402
from isaacsim.core.prims import SingleXFormPrim  # noqa: E402

from isaac_teleop.arm_ik import DualArmVRController  # noqa: E402
from isaac_teleop.frames import (look_at_camera_pose, quat_wxyz_to_mat, recenter_camera_pose,  # noqa: E402
                                 view_correction_from_head, view_nudge_delta)
from isaac_teleop.video_recorder import VideoRecorder  # noqa: E402

XR_FORWARD = "-z"   # Kit XR /user/head uses the camera convention, same as the WebXR bridge
                    # (confirmed live: "+z" made view-relative control invert when facing the robots)
from isaac_teleop.keyboard import TerminalKeys  # noqa: E402
from isaac_teleop.snapshot import SnapshotSource  # noqa: E402
from objects.catalog import get_object  # noqa: E402
from trl_utils.palm_frame import WRIST3_TO_PALM_POS, WRIST3_TO_PALM_QUAT, palm_from_wrist3  # noqa: E402
from isaac_teleop.openxr_source import OpenXRHandsSource  # noqa: E402
from isaac_teleop.recorder import EpisodeRecorder  # noqa: E402
from isaac_teleop.xr_hand import AllegroDexPilot, canonical_to_slots, mp21_to_operator_wrist, xr25_to_mp21  # noqa: E402

def ensure_xr_viewport(tag: str = "") -> bool:
    """Re-enable the active viewport -- the XR compositor renders the headset image through it.

    IsaacLab disables it during SimulationContext init whenever offscreen rendering is on; without
    this repair the HMD stays black ("Viewport does not have the right aov") even though tracking,
    physics and the episode recorder all work. Safe to call repeatedly.
    """
    try:
        from omni.kit.viewport.utility import get_active_viewport
        vp = get_active_viewport()
    except Exception as exc:
        print(f"[xr] viewport check unavailable {tag}: {exc}", flush=True)
        return False
    if vp is None:
        print(f"[xr] WARNING: no active viewport {tag} -- the headset will stay black", flush=True)
        return False
    was = bool(getattr(vp, "updates_enabled", True))
    if not was:
        vp.updates_enabled = True
        print(f"[xr] active viewport updates were OFF {tag} -> re-enabled (this is what blanks the HMD)",
              flush=True)
    else:
        print(f"[xr] active viewport updates on {tag}", flush=True)
    return bool(getattr(vp, "updates_enabled", True))


XR_WRIST = 0
KEY_HELP = ("keys: A=start demo | F=finish (then S=save OK, N=save as FAIL, D=discard) | SPACE=engage/disengage both arms | "
            "1=left arm | 2=right arm | C=re-center (eyes -> robot head, facing the table) | "
            "K=hold/release the view | 3=standalone video | R=reset (discard) | Q=quit")


def _git_sha(path: str) -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def apply_ghost_visuals(robot_paths, frame_paths, opacity: float, hide_frame: bool, mode: str = "cutout") -> int:
    """See-through robots + hidden mount frame. Visual only: materials/visibility, physics untouched.

    mode="cutout": fractional (dithered) cutout -- the surface is stochastically opaque, so it writes
    depth and stays put under CloudXR's depth-based reprojection. mode="translucent": true alpha; looks
    smoother but translucent pixels carry no depth, so they smear/detach from opaque parts in motion
    (seen live 2026-08-20). The robot USDs use instanced meshes: the binding is applied inside each
    instance prototype as well, otherwise tips/links keep their original opaque material."""
    import omni.usd
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    if opacity < 1.0:
        carb.settings.get_settings().set_bool("/rtx/raytracing/fractionalCutoutOpacity", mode == "cutout")
        mat = UsdShade.Material.Define(stage, "/World/Looks/GhostRobot")
        sh = UsdShade.Shader.Define(stage, "/World/Looks/GhostRobot/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.6, 0.75, 0.9))
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        if mode == "cutout":
            sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.01)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

        def bind(prim):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

        n_proto = 0
        for path in robot_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim:
                print(f"[xr][warn] robot prim not found: {path}", flush=True)
                continue
            bind(prim)
            # instanced meshes ignore bindings inherited from outside the instance: de-instance the
            # subtree so the override reaches every link (visual scene only, physics unaffected)
            for p_ in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                if p_.IsInstance():
                    p_.SetInstanceable(False)
                    n_proto += 1
            print(f"[xr] ghost material ({opacity:.2f}, {mode}) on {path}", flush=True)
        if n_proto:
            print(f"[xr] de-instanced {n_proto} prims so the ghost material applies everywhere", flush=True)
    else:
        n_proto = 0
    if hide_frame:
        for path in frame_paths:
            prim = stage.GetPrimAtPath(path)
            if prim:
                UsdGeom.Imageable(prim).MakeInvisible()
                print(f"[xr] hidden: {path}", flush=True)
            else:
                print(f"[xr][warn] frame prim not found: {path}", flush=True)
    return n_proto


def _state_zero() -> np.ndarray:
    return np.zeros(4, np.float32)


class StatusCue:
    """Small cube above the table whose color mirrors the teleop state (the terminal HUD is invisible
    in the headset): grey=idle, orange=one arm engaged, green=both engaged, blue=view hold,
    magenta=SUCCESS latched. Visual only."""

    COLORS = {"idle": (0.45, 0.45, 0.45), "one": (1.0, 0.55, 0.1), "both": (0.1, 0.85, 0.2),
              "hold": (0.2, 0.45, 1.0), "success": (1.0, 0.1, 0.8)}

    def __init__(self, world_pos):
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        cube = UsdGeom.Cube.Define(stage, "/World/StatusCue")
        cube.GetSizeAttr().Set(0.05)
        UsdGeom.XformCommonAPI(cube.GetPrim()).SetTranslate(tuple(float(x) for x in world_pos))
        self._attr = cube.GetDisplayColorAttr()
        self._state = None
        self.set("idle")

    def set(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self._attr.Set([self.COLORS[state]])

    def update(self, n_engaged: int, hold: bool, success: bool) -> None:
        self.set("success" if success else "hold" if hold else
                 "both" if n_engaged == 2 else "one" if n_engaged == 1 else "idle")


def main() -> None:
    settings = carb.settings.get_settings()
    settings.set_string("/persistent/xr/system/openxr/runtime", "system")
    settings.set_string("/persistent/xr/profile/ar/system/display", "OpenXR")
    print(f"[xr] XR_RUNTIME_JSON={os.environ.get('XR_RUNTIME_JSON')}", flush=True)

    steps_per_s = 60.0
    ep_steps = int(round(args_cli.max_episode_seconds * steps_per_s))
    overrides = {
        "handover_full_hand_action": True,
        "handover_object_id": str(args_cli.object),
        "handover_rand_obs_noise": True, "handover_rand_pd_gains": True,
        "handover_domain_rand": bool(args_cli.domain_rand),
        "episode_length": ep_steps, "episode_length_s": ep_steps / steps_per_s,
    }
    env, raw, env_cfg = make_env(args_cli.task, 1, device=str(args_cli.device), env_cfg_overrides=overrides)
    if hasattr(raw.cfg, "terminate_on_success"):
        raw.cfg.terminate_on_success = False
    n_sub = max(1, int(args_cli.sim_steps_per_frame))
    raw.cfg.sim.render_interval = int(raw.cfg.decimation) * n_sub    # render once per n_sub control steps
    ctrl_dt = float(getattr(raw, "step_dt", 1.0 / steps_per_s))
    print(f"[teleop] env ready: obs={env.num_observations} act={env.num_actions} ctrl_dt={ctrl_dt:.4f}", flush=True)
    ensure_xr_viewport("after env creation")

    # --- XR anchor + device (hand/head tracking in world frame) ---
    yaw = math.radians(args_cli.anchor_yaw)
    anchor_rot = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    env_origin = raw.env_origins[0].detach().cpu().numpy().astype(float)
    anchor_pos = tuple((env_origin + np.asarray(args_cli.anchor_pos, dtype=float)).tolist())
    device = OpenXRDevice(OpenXRDeviceCfg(xr_cfg=XrCfg(anchor_pos=anchor_pos, anchor_rot=anchor_rot, near_plane=0.05)))
    anchor_prim = SingleXFormPrim("/XRAnchor")
    print(f"[xr] anchor pos={np.round(anchor_pos, 3).tolist()} yaw={args_cli.anchor_yaw}", flush=True)

    from omni.kit.xr.core import XRCore
    from pxr import Gf
    # Kit's AR profile ships `defaults.xr.profile.ar.foveation.mode = "warped"`
    # (omni.kit.xr.profile.ar/config/extension.toml), so the XR composition pipeline runs an
    # "Image Resample/Unwarp" stage that expects a WARPED source image. The CloudXR runtime here
    # reports NV_CXR_APP_PQW_SUBMIT_MODE=unwarped with PQW_WARPED_WIDTH=0, so that warped image is
    # never produced -> "[XR Composition pipeline] Unwarp/resample failed to execute: XRComposition
    # (warped) input resource is null" every frame and a BLACK HMD, while head/hand tracking, physics
    # and the recorder all keep working normally. "none" submits plain stereo, which is what this
    # runtime expects. Flip back with --foveation warped if a future client/runtime negotiates it.
    _xr_settings = carb.settings.get_settings()
    # Kit reads profile settings from /xr/..., /persistent/xr/... or /defaults/xr/... depending on
    # how the profile was loaded (the AR experience uses persistent.* for renderQuality), so set
    # all three and print the readback rather than guessing which one wins.
    _fov_keys = ("/xr/profile/ar/foveation/mode",
                 "/persistent/xr/profile/ar/foveation/mode",
                 "/defaults/xr/profile/ar/foveation/mode")
    for _key in _fov_keys:
        _xr_settings.set(_key, str(args_cli.foveation))
    print(f"[xr] foveation mode -> {args_cli.foveation} "
          "readback " + " ".join(f"{k}={_xr_settings.get(k)!r}" for k in _fov_keys), flush=True)
    XRCore.get_singleton().request_enable_profile("ar")

    # --- visuals: ghost robots, hide the T-shaped mount frame ---
    env_root = "/World/envs/env_0"
    n_deinst = apply_ghost_visuals([f"{env_root}/RightRobot", f"{env_root}/LeftRobot"], [f"{env_root}/tshape"],
                                   opacity=float(args_cli.robot_opacity), hide_frame=not args_cli.show_frame,
                                   mode=args_cli.ghost_mode)
    if n_deinst:
        # de-instancing changes prim paths -> the PhysX tensor views are stale; a full sim reset
        # (stop + play) re-initializes every articulation/asset view.
        print("[xr] sim.reset() to rebuild physics views after de-instancing ...", flush=True)
        raw.sim.reset()

    # --- control stack (same as the bridge) ---
    print("[teleop] building arm controller + DexPilot ...", flush=True)
    ctrl = DualArmVRController(raw, pos_scale=args_cli.pos_scale, clamp_pos=args_cli.clamp_pos,
                               clamp_rot=args_cli.clamp_rot, rot_weight=args_cli.rot_weight,
                               max_ee_speed=args_cli.max_ee_speed, dt=ctrl_dt, ff_tau=args_cli.ff_tau,
                               max_joint_speed=args_cli.max_joint_speed)
    alpha = None if args_cli.hand_alpha >= 1.0 else float(args_cli.hand_alpha)
    retarget = {s_: AllegroDexPilot(s_, ema_alpha=alpha, profile=args_cli.hand_profile,
                                    finger_gain=args_cli.finger_gain, thumb_mode=args_cli.thumb_mode)
                for s_ in ("right", "left")}
    src = OpenXRHandsSource(stale_after=0.5)
    snap = SnapshotSource(raw, ctrl)
    # Palm-frame parity: if the USD kept *_palm_link as a body (fixed joints not merged), compare the
    # constant used by the recorder against the sim; otherwise report n/a.
    for side_, robot_ in (("right", raw.right_robot), ("left", raw.left_robot)):
        names_ = list(robot_.data.body_names)
        palm_name = f"{'ra' if side_ == 'right' else 'la'}_palm_link"
        if palm_name in names_:
            pR_, qR_, pL_, qL_ = ctrl.teleop._compute_frame_pose_world()
            pw_, qw_ = (pR_, qR_) if side_ == "right" else (pL_, qL_)
            pp_, _ = palm_from_wrist3(pw_[0].cpu().numpy(), qw_[0].cpu().numpy(), side_)
            ps_ = robot_.data.body_pos_w[0, names_.index(palm_name)].cpu().numpy()
            print(f"[teleop] palm check {side_}: |const - sim| = {float(np.linalg.norm(pp_ - ps_)):.4f} m", flush=True)
        else:
            print(f"[teleop] palm check {side_}: n/a ({palm_name} merged into wrist_3)", flush=True)

    session_meta = {
        "cli": " ".join(sys.argv[1:]), "quest3_ws_sha": _git_sha(QWS), "thesis_ws_sha": _git_sha(TWS),
        "mode": "cloudxr_native_xr", "anchor_pos": list(anchor_pos), "anchor_yaw": args_cli.anchor_yaw,
        "hand_profile": args_cli.hand_profile, "finger_gain": args_cli.finger_gain,
        "thumb_mode": args_cli.thumb_mode,
        "view_comp": args_cli.view_comp,
        "dexpilot_cfg": {s: r.cfg_path for s, r in retarget.items()},
        "urdf": {s: r.urdf_path for s, r in retarget.items()}, "ctrl_dt": ctrl_dt,
        "obs_dim": int(env.num_observations), "share_obs_dim": int(env.num_share_observations),
        "act_dim": int(env.num_actions),
        "object_id": str(args_cli.object), "object_spec": get_object(args_cli.object).to_dict(),
        "T_wrist3_palm": {s: {"pos": WRIST3_TO_PALM_POS[s].tolist(), "quat_wxyz": WRIST3_TO_PALM_QUAT[s].tolist()}
                          for s in ("right", "left")},
        "sim_dt": float(raw.physics_dt), "decimation": int(raw.cfg.decimation),
        "episode_steps_cap": int(ep_steps), "env_cfg_overrides": overrides,
        "session_note": str(args_cli.session_note),
    }
    rec = EpisodeRecorder(args_cli.out_dir, task=args_cli.task, session_tag=args_cli.session_tag,
                          session_meta=session_meta, outcome_in_name=True)

    def new_episode():
        raw.reset()
        obs, share, _ = env.reset()
        ctrl.reset_hold()
        for r in retarget.values():
            r.reset()
        return obs, share

    def set_engage(side: str, on: bool) -> None:
        if not on:
            ctrl.disengage(side)
            return
        joints = src.hand(side)
        if joints is None:
            return
        R_view = None
        if args_cli.view_comp != "off" and src.head() is not None:
            R_view = view_correction_from_head(src.head(),
                                               snap_deg=90.0 if args_cli.view_comp == "snap90" else 0.0,
                                               forward_axis=XR_FORWARD)
        T = joints[XR_WRIST]
        ctrl.engage(side, T[:3, 3].copy(), T[:3, :3].copy(), R_view=R_view)

    nan_hand = np.full((25, 4, 4), np.nan, np.float32)

    def snapshot(obs, share, act, raw_act, t_now):
        head = src.head()
        d = snap.base(obs, share, act, raw_act, t_now)
        d.update({
            "vr_head": (head if head is not None else np.full((4, 4), np.nan)).astype(np.float32),
            "vr_left": (src.hand("left") if src.hand("left") is not None else nan_hand).astype(np.float32),
            "vr_right": (src.hand("right") if src.hand("right") is not None else nan_hand).astype(np.float32),
            "vr_left_state": _state_zero(), "vr_right_state": _state_zero(),
        })
        return d

    cue = None if args_cli.no_status_cue else StatusCue(env_origin + np.array([0.55, 0.435, 1.30]))
    eye_world = env_origin + np.asarray(args_cli.eye_pos, dtype=np.float64)

    obs, share = new_episode()
    step = 0
    t_ep0 = time.time()
    last_msg = "ready"
    hold_view = False
    hold_head_pos: np.ndarray | None = None
    centered = False
    t_head_lost = 0.0

    def recenter() -> bool:
        """Teleport the operator's eyes to eye_world facing eye_yaw (embodiment: you ARE the
        dual-arm robot). Goes through XRCore.schedule_set_camera -- writing the anchor prim only
        applies translation mid-session, its rotation is silently ignored (why auto-center used to
        leave the view 180 deg off). Disengages both arms (the clutch would jump otherwise)."""
        nonlocal hold_view, hold_head_pos, centered, last_msg
        H = src.head()
        if H is None or not src.head_live(time.time()):
            return False        # a static pose is the runtime's no-headset default, not the operator
        for side_ in ("right", "left"):
            ctrl.disengage(side_)
        hold_view = False; hold_head_pos = None
        M = recenter_camera_pose(H, eye_world, args_cli.eye_yaw, XR_FORWARD)
        XRCore.get_singleton().schedule_set_camera(Gf.Matrix4d(M.T.tolist()))
        centered = True
        print(f"[xr] centered: eyes -> {np.round(eye_world, 3).tolist()} facing {args_cli.eye_yaw:.0f} deg", flush=True)
        return True
    # --- front-view video recording (key 3) ---
    video = {"cam": None, "rec": None, "t_next": 0.0}
    video_w, video_h = (int(x) for x in args_cli.video_res.lower().split("x"))

    def video_camera():
        """Raw USD camera + replicator render product, created on record-start and destroyed on
        stop so the 4K render costs nothing while not recording. (The isaaclab Camera sensor
        cannot be created mid-run: its view initialises with 0 instances.)"""
        if video["cam"] is None:
            import omni.replicator.core as rep
            import omni.usd
            from pxr import UsdGeom
            print(f"[video] creating front camera {video_w}x{video_h} ...", flush=True)
            stage = omni.usd.get_context().get_stage()
            usd_cam = UsdGeom.Camera.Define(stage, "/World/VideoCam")
            usd_cam.GetFocalLengthAttr().Set(16.0)
            usd_cam.GetHorizontalApertureAttr().Set(20.955)
            usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 20.0))
            pos, quat = look_at_camera_pose(env_origin + np.asarray(args_cli.video_cam_pos),
                                            env_origin + np.asarray(args_cli.video_cam_target))
            M = np.eye(4)
            M[:3, :3] = quat_wxyz_to_mat(quat)
            M[:3, 3] = pos
            xf = UsdGeom.Xformable(usd_cam.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(Gf.Matrix4d(M.T.tolist()))
            rp = rep.create.render_product("/World/VideoCam", (video_w, video_h))
            annot = rep.AnnotatorRegistry.get_annotator("rgb")
            annot.attach(rp)
            video["cam"] = (rp, annot)
        return video["cam"]

    def toggle_video() -> str:
        if video["rec"] is None:
            video_camera()
            path = rec.dir / time.strftime("video_%Y%m%d_%H%M%S.mp4")
            video["rec"] = VideoRecorder(path, video_w, video_h, fps_in=args_cli.video_capture_fps,
                                         fps_out=args_cli.video_fps, encoder=args_cli.video_encoder)
            video["t_next"] = time.time()
            return f"video REC -> {path.name}"
        p = video["rec"].close()
        video["rec"] = None
        if video["cam"] is not None:
            rp, annot = video["cam"]
            video["cam"] = None
            try:
                annot.detach(rp)
                rp.destroy()
            except Exception:
                pass
        return f"video saved {p.name}" if p else "video: nothing recorded"

    if args_cli.video_autostart:
        print("[teleop] " + toggle_video(), flush=True)

    # --- demo state machine: free -> (A) -> recording -> (F / env end) -> decide -> (S/D) -> free ---
    phase = "free"
    demo_success_env = False

    def start_demo() -> str:
        nonlocal obs, share, step, t_ep0, phase
        if video["rec"] is not None:                 # a manual key-3 video is running: finish it first
            print("[teleop] " + toggle_video(), flush=True)
        for side_ in ("right", "left"):
            ctrl.disengage(side_)
        obs, share = new_episode(); step = 0; t_ep0 = time.time()
        rec.start({"anchor_pos": list(anchor_pos), "anchor_yaw": args_cli.anchor_yaw})
        if not args_cli.no_episode_video:
            try:
                video_camera()
                video["rec"] = VideoRecorder(rec.dir / "_recording.mp4", video_w, video_h,
                                             fps_in=args_cli.video_capture_fps, fps_out=args_cli.video_fps,
                                             encoder=args_cli.video_encoder)
                video["t_next"] = time.time()
            except Exception as e:
                print(f"[teleop] episode video unavailable: {e}", flush=True)
        phase = "recording"
        return "DEMO started -- SPACE to engage, F to finish"

    def finish_demo(reason: str = "") -> str:
        nonlocal phase
        for side_ in ("right", "left"):
            ctrl.disengage(side_)
        phase = "decide"
        return (f"demo finished{reason} | env success={'YES' if demo_success_env else 'no'} "
                f"stage R{int(raw.subgoal[0, 0].item())}/L{int(raw.subgoal[0, 1].item())} "
                "-- S=save OK  N=save as FAIL  D=discard")

    def decide_demo(save: bool, manual_ok: bool = True) -> str:
        nonlocal obs, share, step, t_ep0, phase
        vp = None
        if video["rec"] is not None:
            try:
                vp = video["rec"].close()
            except Exception as e:
                print(f"[teleop] video close failed: {e}", flush=True)
            video["rec"] = None
        if save:
            p = rec.finish(success_env=demo_success_env, success_manual=bool(manual_ok))
            msg = f"saved {p.name}" if p else "nothing recorded"
            if p and vp is not None:
                vp = vp.rename(p.with_suffix(".mp4"))
                msg += f" + {vp.name}"
        else:
            rec.discard()
            if vp is not None:
                vp.unlink(missing_ok=True)
            msg = "discarded"
        obs, share = new_episode(); step = 0; t_ep0 = time.time()
        phase = "free"
        return msg + " -- A starts the next demo"

    prof = {"input": 0.0, "compute": 0.0, "env_step": 0.0, "n": 0}
    t_print = time.time()
    print(KEY_HELP, flush=True)
    ensure_xr_viewport("before the render loop")
    print("[teleop] running -- Connect in the headset, put the controllers down, then SPACE to engage", flush=True)

    with TerminalKeys() as keys:
        while simulation_app.is_running():
            t_now = time.time()
            # ---- keys ----
            k = keys.poll()
            if k:
                kl = k.lower()
                if kl == "q":
                    break
                elif phase == "decide":
                    if kl == "s":
                        if args_cli.require_success and not demo_success_env:
                            last_msg = "not saved: env success missing (N saves it as FAIL, D discards)"
                        else:
                            last_msg = decide_demo(save=True, manual_ok=True)
                            print(f"[teleop] {last_msg}", flush=True)
                    elif kl == "n":
                        last_msg = decide_demo(save=True, manual_ok=False)
                        print(f"[teleop] {last_msg}", flush=True)
                    elif kl in ("d", "r"):
                        last_msg = decide_demo(save=False)
                        print(f"[teleop] {last_msg}", flush=True)
                    else:
                        last_msg = "demo paused: S=save OK  N=save FAIL  D=discard"
                elif kl == "a":
                    last_msg = "demo already running (F to finish)" if phase == "recording" else start_demo()
                    print(f"[teleop] {last_msg}", flush=True)
                elif kl == "f":
                    if phase == "recording":
                        demo_success_env = bool(raw.success_buf[0].item())
                        last_msg = finish_demo()
                        print(f"[teleop] {last_msg}", flush=True)
                    else:
                        last_msg = "no demo running (A starts one)"
                elif k == " ":
                    tgt = not (ctrl.engaged("right") and ctrl.engaged("left"))
                    for side in ("right", "left"):
                        set_engage(side, tgt)
                    last_msg = "engage" if tgt else "disengage"
                elif kl in ("1", "2"):
                    side = "left" if kl == "1" else "right"
                    set_engage(side, not ctrl.engaged(side))
                    last_msg = f"{side}: {'engaged' if ctrl.engaged(side) else 'idle'}"
                elif kl == "c":
                    last_msg = "re-centered on the robot head" if recenter() else "cannot center: no live head tracking"
                elif kl == "k":
                    hold_view = not hold_view
                    hold_head_pos = src.head()[:3, 3].copy() if (hold_view and src.head() is not None) else None
                    if hold_view and hold_head_pos is None:
                        hold_view = False
                        last_msg = "cannot hold view: no head pose"
                    else:
                        last_msg = "VIEW HOLD on (walk freely)" if hold_view else "view hold off"
                elif k in ("UP", "DOWN", "LEFT", "RIGHT") or kl in ("o", "l"):
                    if hold_view and hold_head_pos is not None:
                        nudge_key = kl if kl in ("o", "l") else k
                        hold_head_pos += view_nudge_delta(nudge_key, src.head(), args_cli.eye_yaw,
                                                          args_cli.nudge_step, forward_axis=XR_FORWARD)
                        last_msg = "view moved " + {"UP": "forward", "DOWN": "back", "LEFT": "left",
                                                    "RIGHT": "right", "o": "up", "l": "down"}[nudge_key]
                    else:
                        last_msg = "view move ignored: press K first (view hold)"
                elif kl == "s":
                    last_msg = ("press F to finish the demo first" if phase == "recording"
                                else "no demo to save (A starts one)")
                elif kl == "3":
                    if phase == "recording":
                        last_msg = "video is tied to the demo (F to finish it)"
                    else:
                        try:
                            last_msg = toggle_video()
                        except Exception as e:
                            last_msg = f"video error: {e}"
                        print(f"[teleop] {last_msg}", flush=True)
                elif kl == "d":
                    if phase == "recording":
                        last_msg = "demo cancelled: " + decide_demo(save=False)
                        print(f"[teleop] {last_msg}", flush=True)
                    else:
                        last_msg = "nothing to discard (A starts a demo)"
                elif kl == "r":
                    if phase == "recording":
                        last_msg = "demo cancelled: " + decide_demo(save=False)
                        print(f"[teleop] {last_msg}", flush=True)
                    else:
                        obs, share = new_episode(); step = 0; t_ep0 = time.time(); last_msg = "env reset"

            # ---- decide phase: sim paused for the save/discard verdict, XR view stays alive ----
            if phase == "decide":
                try:
                    raw.sim.render()
                except Exception:
                    simulation_app.update()
                if t_now - t_print >= 2.0:
                    t_print = t_now
                    print(f"[teleop] PAUSED (demo review) | {last_msg}", flush=True)
                continue

            # ---- XR input ----
            _t0 = time.perf_counter()
            src.update(device._get_raw_data(), t_now)
            if not centered and not args_cli.no_auto_center and recenter():
                last_msg = "auto-centered on the robot head"
                print(f"[teleop] {last_msg}", flush=True)
            elif centered and not args_cli.no_auto_center and not src.head_live(t_now):
                if t_head_lost == 0.0:
                    t_head_lost = t_now
                elif t_now - t_head_lost > 3.0:     # headset gone: re-center when it comes back
                    centered = False
                    print("[teleop] head tracking lost -> will auto-center on reconnect", flush=True)
            else:
                t_head_lost = 0.0
            hand_pose_w = {"right": None, "left": None}
            tracked = {"right": False, "left": False}
            for side in ("right", "left"):
                joints = src.hand(side)
                if joints is None:
                    continue
                if not src.fresh(side, t_now):
                    if ctrl.engaged(side) and (t_now - src.last_change(side)) > 1.0:
                        ctrl.disengage(side); last_msg = f"auto-disengage ({side}: tracking stale)"
                    continue
                tracked[side] = True
                T = joints[XR_WRIST]
                hand_pose_w[side] = (T[:3, 3].copy(), T[:3, :3].copy())
                if (not args_cli.hand_follow_clutch) or ctrl.engaged(side):
                    mp = mp21_to_operator_wrist(xr25_to_mp21(joints), side)
                    ctrl.set_hand(side, canonical_to_slots(retarget[side].retarget(mp), ctrl.slot_of_canon[side]))
            if args_cli.auto_engage:
                for side in ("right", "left"):
                    if tracked[side] and not ctrl.engaged(side):
                        set_engage(side, True)

            # ---- view hold: move the XR anchor so the (virtual) head stays where it was ----
            if hold_view and hold_head_pos is not None and src.head() is not None:
                err = hold_head_pos - src.head()[:3, 3]
                if float(np.linalg.norm(err)) > 0.005:
                    cur_pos, cur_rot = anchor_prim.get_world_pose()
                    anchor_prim.set_world_pose(np.asarray(cur_pos, dtype=np.float32) + 0.5 * err.astype(np.float32),
                                               cur_rot)

            # ---- act: n_sub control steps per rendered XR frame (only the last one renders) ----
            _t1 = time.perf_counter()
            _t2 = _t1
            stop = False
            for _ in range(n_sub):
                act, raw_act = ctrl.compute(hand_pose_w)
                assert torch.isfinite(act).all(), "non-finite action"
                step_dict = snapshot(obs, share, act, raw_act, t_now) if rec.active else None
                _t2 = time.perf_counter()
                next_obs, _, next_share, _, done, _, _ = env.step(act.transpose(0, 1).contiguous(), step)
                if step_dict is not None:
                    rec.append(step_dict)
                obs, share = next_obs, next_share
                step += 1

                # auto done (episode time limit / object fell)
                is_done = bool(done[0].any()) if hasattr(done, "any") else bool(done)
                if is_done:
                    if phase == "recording":
                        demo_success_env = bool(raw.success_buf[0].item())
                        last_msg = finish_demo(" (ended by env)")
                    else:
                        obs, share = new_episode(); step = 0; t_ep0 = time.time()
                        last_msg = "env reset (episode limit)"
                    print(f"[teleop] {last_msg}", flush=True)
                    break
                if args_cli.steps and step >= args_cli.steps:
                    if rec.active:
                        p = rec.finish(success_env=bool(raw.success_buf[0].item()), success_manual=None)
                        print(f"[teleop] steps reached, saved {p}", flush=True)
                    else:
                        print("[teleop] steps reached", flush=True)
                    stop = True
                    break
            _t3 = time.perf_counter()
            if stop:
                break

            if video["rec"] is not None and t_now >= video["t_next"]:
                # wall-clock pacing: write one frame per elapsed 1/fps slot (duplicating when the
                # loop runs slower than fps, skipping when faster) so playback speed == real life
                due = min(1 + int((t_now - video["t_next"]) * args_cli.video_capture_fps), 30)
                video["t_next"] += due / args_cli.video_capture_fps
                try:
                    rgb = np.asarray(video["cam"][1].get_data())
                    if rgb.ndim == 3 and rgb.shape[:2] == (video_h, video_w):
                        frame = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
                        for _ in range(due):
                            video["rec"].add(frame)
                    else:
                        video["t_next"] = t_now      # render product still warming up: restart the clock
                except Exception as e:
                    video["rec"] = None
                    last_msg = f"video stopped: {e}"
                    print(f"[teleop] {last_msg}", flush=True)

            if args_cli.profile:
                prof["input"] += _t1 - _t0; prof["compute"] += _t2 - _t1; prof["env_step"] += _t3 - _t2; prof["n"] += 1
            if cue is not None:
                cue.update(int(ctrl.engaged("right")) + int(ctrl.engaged("left")), hold_view,
                           bool(raw.success_buf[0].item()))
            if t_now - t_print >= 1.0:
                t_print = t_now
                hud = (f"[teleop] step {step:6d} t={time.time() - t_ep0:6.1f}s "
                       f"stage R{int(raw.subgoal[0, 0].item())}/L{int(raw.subgoal[0, 1].item())} "
                       f"{'REC ' + str(rec.length) if phase == 'recording' else 'FREE (A=start demo)'} | "
                       f"L:{'ENG' if ctrl.engaged('left') else 'idle'} R:{'ENG' if ctrl.engaged('right') else 'idle'} "
                       f"track L{'+' if tracked['left'] else '-'} R{'+' if tracked['right'] else '-'} | "
                       + "".join(f"{sd[0].upper()}-arm:{fl} | " for sd, fl in ctrl.limits_hit.items() if fl)
                       + f"{'HOLD ' if hold_view else ''}{'SUCCESS ' if bool(raw.success_buf[0].item()) else ''}{last_msg}")
                if args_cli.profile and prof["n"]:
                    n = prof["n"]
                    hud += (f" | ms: in {1e3*prof['input']/n:.1f} cmp {1e3*prof['compute']/n:.1f} "
                            f"step {1e3*prof['env_step']/n:.1f}")
                print(hud, flush=True)

    if rec.active and rec.length:
        p = rec.abort()
        print(f"[teleop] open episode written: {p}", flush=True)
    if video["rec"] is not None:
        try:
            print(f"[teleop] video saved {video['rec'].close()}", flush=True)
        except Exception as e:
            print(f"[teleop] video close failed: {e}", flush=True)
    try:
        XRCore.get_singleton().request_disable_profile()
    except Exception:
        pass
    for closer in (getattr(env, "close", None), getattr(raw, "close", None)):
        if closer:
            try:
                closer()
            except Exception:
                pass
            break
    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()

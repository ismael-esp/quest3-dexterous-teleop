"""Quest 3 VR teleop of the thesis handover_tshape env (44-D dexterous actions) + demo recording.

Run via scripts/run_teleop.sh (env_isaaclab). Needs --enable_cameras (stereo view) and the bridge
(quest_bridge/vuer_server.py) unless --fake_vr / --replay is used.
"""
from __future__ import annotations

import argparse
import os
import subprocess
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

# pinocchio (used by dex_retargeting) MUST be imported before AppLauncher so the pip version is
# used instead of the copy bundled with Isaac Sim (same trick as Isaac Lab's --enable_pinocchio).
import pinocchio  # noqa: E402,F401
import dex_retargeting  # noqa: E402,F401

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="handover_tshape")
parser.add_argument("--max_episode_seconds", type=float, default=120.0)
parser.add_argument("--rt_factor", type=float, default=1.0, help="wall-clock pacing; <=0 = as fast as possible")
parser.add_argument("--steps", type=int, default=0, help="stop after N steps (0 = until Q); used by tests")
parser.add_argument("--vr_sub", default="tcp://127.0.0.1:5555")
parser.add_argument("--stereo_pub", default="tcp://127.0.0.1:5556")
parser.add_argument("--fake_vr", action="store_true")
parser.add_argument("--object", default="rod", help="object catalog id: rod | rod_slim | tumbler")
parser.add_argument("--replay", default="", help="episode pickle whose vr stream is re-fed")
parser.add_argument("--auto_engage", action="store_true", help="engage both arms as soon as tracking is present")
parser.add_argument("--embody", dest="embody", action="store_true", default=True,
                    help="(default) place the operator's eyes at --eye_pos facing --eye_yaw when the first head pose "
                         "arrives (and on key C); hands move 1:1 relative to the head")
parser.add_argument("--no_embody", dest="embody", action="store_false",
                    help="use the fixed --vr_origin/--vr_yaw mapping instead of --embody")
parser.add_argument("--eye_pos", type=float, nargs=3, default=[0.0, 0.05, 1.65],
                    help="[embody] world position (env-local) of the eyes; default = just above/in front of the top of "
                         "the torso column, between the arm mounts (0, 0, 1.49) -- a human head on the robot's shoulders "
                         "(the env's own head camera (0,-0.42,1.34) sits behind the column, which blocks the view)")
parser.add_argument("--eye_yaw", type=float, default=90.0,
                    help="[embody] world heading [deg] the operator faces (90 = +y = toward the table)")
parser.add_argument("--vr_origin", type=float, nargs=3, default=[0.0, 1.2, 0.0],
                    help="[--no_embody] where the VR floor origin sits in the world")
parser.add_argument("--vr_yaw", type=float, default=0.0, help="[--no_embody] extra yaw of the VR frame [deg]")
parser.add_argument("--pos_scale", type=float, default=1.0)
parser.add_argument("--max_ee_speed", type=float, default=6.0,
                    help="max EE target advance per second; with --clamp_pos it sets the 'lead' (target minus actual "
                         "EE, per step) = min(clamp_pos, max_ee_speed/60). The arm PD (K/D ~ 3.7/s) turns the lead into "
                         "speed (~0.37 m/s at 10 cm); 1.5 (2.5 cm lead) made the arms crawl at ~0.09 m/s")
parser.add_argument("--clamp_pos", type=float, default=0.10, help="max EE position lead per step [m]")
parser.add_argument("--clamp_rot", type=float, default=0.50)
parser.add_argument("--rot_weight", type=float, default=1.0)
parser.add_argument("--ff_tau", type=float, default=0.10,
                    help="velocity feed-forward horizon [s] on the EE target (0 = off)")
parser.add_argument("--ee_link_right", default="ra_wrist_3_link")
parser.add_argument("--ee_link_left", default="la_wrist_3_link")
parser.add_argument("--hand_alpha", type=float, default=0.5, help="EMA on retargeted hand joints (1=no smoothing)")
parser.add_argument("--hand_follow_clutch", action="store_true", help="only retarget fingers while the arm is engaged")
parser.add_argument("--eye_res", default="640x480")
parser.add_argument("--ipd", type=float, default=0.063)
parser.add_argument("--hfov", type=float, default=105.0,
                    help="horizontal FOV [deg] of each eye camera (bridge --hfov must match)")
parser.add_argument("--focal_length", type=float, default=None, help="override the eye camera focal length [mm]")
parser.add_argument("--jpeg_quality", type=int, default=80)
parser.add_argument("--image_every", type=int, default=2, help="publish stereo every N control steps")
parser.add_argument("--out_dir", default=os.path.join(QWS, "demos", "vr"))
parser.add_argument("--session_tag", default=time.strftime("%Y%m%d_%H%M%S"))
parser.add_argument("--require_success", action="store_true", help="S refuses to save unless success_buf is set")
parser.add_argument("--domain_rand", action="store_true", default=False, help="enable handover_domain_rand")
parser.add_argument("--profile", action="store_true", help="print per-stage timing every 120 steps")
parser.add_argument("--debug_stacks", type=float, default=0.0, help="if >0, dump all thread stacks to stderr every N seconds")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.enable_cameras:
    print("[teleop] forcing --enable_cameras (stereo view needs the camera pipeline)")
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
if args_cli.debug_stacks > 0:
    import faulthandler
    faulthandler.dump_traceback_later(args_cli.debug_stacks, repeat=True, file=sys.stderr)

# ---- everything below runs inside the kit app -------------------------------------------------
import numpy as np   # noqa: E402
import torch         # noqa: E402

os.chdir(TWS)  # env cfgs use BASE_DIR="" -> assets/... relative to the env root
from task_registry import make_env  # noqa: E402

from isaac_teleop.arm_ik import DualArmVRController  # noqa: E402
from isaac_teleop.fake_vr import FakeVRSource, ReplayVRSource  # noqa: E402
from isaac_teleop.snapshot import SnapshotSource  # noqa: E402
from objects.catalog import get_object  # noqa: E402
from trl_utils.palm_frame import WRIST3_TO_PALM_POS, WRIST3_TO_PALM_QUAT, palm_from_wrist3  # noqa: E402
from isaac_teleop.frames import T_world_vr, embody_T_world_vr, transform_pose  # noqa: E402
from isaac_teleop.keyboard import KEY_HELP, TerminalKeys  # noqa: E402
from isaac_teleop.recorder import EpisodeRecorder  # noqa: E402
from isaac_teleop.stereo_view import StereoView  # noqa: E402
from isaac_teleop.vr_source import StereoPublisher, ZmqVRSource  # noqa: E402
from isaac_teleop.xr_hand import AllegroDexPilot, canonical_to_slots, mp21_to_operator_wrist, xr25_to_mp21  # noqa: E402

XR_WRIST = 0


def _git_sha(path: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _state_vec(s: dict) -> np.ndarray:
    return np.array([float(s.get("pinch", 0)), float(s.get("squeeze", 0)),
                     float(s.get("pinch_value", 0.0)), float(s.get("squeeze_value", 0.0))], dtype=np.float32)


def _set_engage(ctrl, side, on, frame, T_wv):
    if not on:
        ctrl.disengage(side)
        return
    joints = getattr(frame, side) if frame is not None else None
    if joints is None:
        return
    Tw = transform_pose(T_wv, joints[XR_WRIST])
    ctrl.engage(side, Tw[:3, 3].copy(), Tw[:3, :3].copy())


def _snapshot(snap, obs, share, act, raw_act, frame, ctrl, t_now):
    nan_hand = np.full((25, 4, 4), np.nan, np.float32)
    d = snap.base(obs, share, act, raw_act, t_now)
    d.update({
        "vr_head": (frame.head if (frame is not None and frame.head is not None) else np.full((4, 4), np.nan)).astype(np.float32),
        "vr_left": (frame.left if (frame is not None and frame.left is not None) else nan_hand).astype(np.float32),
        "vr_right": (frame.right if (frame is not None and frame.right is not None) else nan_hand).astype(np.float32),
        "vr_left_state": _state_vec(frame.left_state) if frame is not None else np.zeros(4, np.float32),
        "vr_right_state": _state_vec(frame.right_state) if frame is not None else np.zeros(4, np.float32),
    })
    return d


def main():
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
    # Render only as often as we publish stereo frames (DirectRLEnv reads cfg.sim.render_interval every step).
    raw.cfg.sim.render_interval = int(raw.cfg.decimation) * max(1, int(args_cli.image_every))
    ctrl_dt = float(getattr(raw, "step_dt", 1.0 / steps_per_s))
    print(f"[teleop] env ready: obs={env.num_observations} share={env.num_share_observations} act={env.num_actions} "
          f"ctrl_dt={ctrl_dt:.4f} max_episode_steps={ep_steps}", flush=True)
    print(f"[teleop] right bodies: {list(raw.right_robot.body_names)}", flush=True)

    T_wv = T_world_vr(args_cli.vr_origin, args_cli.vr_yaw)
    env_origin = raw.env_origins[0].detach().cpu().numpy().astype(np.float64)
    eye_pos_world = env_origin + np.asarray(args_cli.eye_pos, dtype=np.float64)
    centered = False   # [embody] becomes True once T_wv has been fitted to a head pose
    print("[teleop] building arm controller ...", flush=True)
    ctrl = DualArmVRController(raw, ee_link_right=args_cli.ee_link_right, ee_link_left=args_cli.ee_link_left,
                               pos_scale=args_cli.pos_scale, clamp_pos=args_cli.clamp_pos, clamp_rot=args_cli.clamp_rot,
                               rot_weight=args_cli.rot_weight, max_ee_speed=args_cli.max_ee_speed, dt=ctrl_dt, ff_tau=args_cli.ff_tau)
    snap = SnapshotSource(raw, ctrl)
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
    print("[teleop] building DexPilot retargeters ...", flush=True)
    alpha = None if args_cli.hand_alpha >= 1.0 else float(args_cli.hand_alpha)
    retarget = {"right": AllegroDexPilot("right", ema_alpha=alpha), "left": AllegroDexPilot("left", ema_alpha=alpha)}
    w, h = (int(x) for x in args_cli.eye_res.lower().split("x"))
    print(f"[teleop] creating stereo cameras {w}x{h} ...", flush=True)
    focal = args_cli.focal_length if args_cli.focal_length else StereoView.focal_length_for_hfov(args_cli.hfov)
    view = StereoView(raw.sim, width=w, height=h, ipd=args_cli.ipd, focal_length=focal,
                      jpeg_quality=args_cli.jpeg_quality)
    print(f"[teleop] cameras ok (hfov {view.hfov_deg:.1f} deg, focal {focal:.2f} mm)", flush=True)
    stereo_pub = StereoPublisher(args_cli.stereo_pub)

    if args_cli.fake_vr:
        vr = FakeVRSource(hz=60.0)
    elif args_cli.replay:
        vr = ReplayVRSource(args_cli.replay)
    else:
        vr = ZmqVRSource(args_cli.vr_sub, stale_after=0.5)

    session_meta = {
        "cli": " ".join(sys.argv[1:]), "quest3_ws_sha": _git_sha(QWS), "thesis_ws_sha": _git_sha(TWS),
        "T_world_vr": T_wv.tolist(), "dexpilot_cfg": {s: r.cfg_path for s, r in retarget.items()},
        "urdf": {s: r.urdf_path for s, r in retarget.items()}, "ctrl_dt": ctrl_dt,
        "obs_dim": int(env.num_observations), "share_obs_dim": int(env.num_share_observations),
        "act_dim": int(env.num_actions), "ee_links": [args_cli.ee_link_right, args_cli.ee_link_left],
        "object_id": str(args_cli.object), "object_spec": get_object(args_cli.object).to_dict(),
        "T_wrist3_palm": {s: {"pos": WRIST3_TO_PALM_POS[s].tolist(), "quat_wxyz": WRIST3_TO_PALM_QUAT[s].tolist()}
                          for s in ("right", "left")},
        "sim_dt": float(raw.physics_dt), "decimation": int(raw.cfg.decimation),
    }
    rec = EpisodeRecorder(args_cli.out_dir, task=args_cli.task, session_tag=args_cli.session_tag, session_meta=session_meta)

    def new_episode():
        raw.reset()
        obs, share, _ = env.reset()
        ctrl.reset_hold()
        for r in retarget.values():
            r.reset()
        rec.start({"vr_origin": list(args_cli.vr_origin), "vr_yaw": args_cli.vr_yaw, "embody": bool(args_cli.embody),
                   "eye_pos": list(args_cli.eye_pos), "eye_yaw": args_cli.eye_yaw, "T_world_vr": T_wv.tolist()})
        return obs, share

    def recenter(frame):
        """[embody] fit T_world_vr so the current head is at eye_pos_world facing eye_yaw; disengages both arms
        (the anchored clutch would jump otherwise)."""
        nonlocal T_wv, centered
        if frame is None or frame.head is None:
            return False
        for side in ("right", "left"):
            ctrl.disengage(side)
        T_wv = embody_T_world_vr(frame.head, eye_pos_world, args_cli.eye_yaw)
        centered = True
        rec.set_meta({"T_world_vr": T_wv.tolist()})
        print(f"[teleop] re-centered: eyes at {np.round(eye_pos_world, 3).tolist()} facing {args_cli.eye_yaw:.0f} deg", flush=True)
        return True

    obs, share = new_episode()
    prof = {"input": 0.0, "compute": 0.0, "env_step": 0.0, "capture": 0.0, "n": 0}
    last_frame = None
    last_msg = "ready"
    step = 0
    t_ep0 = time.time()
    print(KEY_HELP, flush=True)

    with TerminalKeys() as keys:
        while simulation_app.is_running():
            t_now = time.time()
            # ---- keys ----
            k = keys.poll()
            if k:
                kl = k.lower()
                if kl == "q":
                    break
                elif k == " ":
                    tgt = not (ctrl.engaged("right") and ctrl.engaged("left"))
                    for side in ("right", "left"):
                        _set_engage(ctrl, side, tgt, last_frame, T_wv)
                    last_msg = "engage" if tgt else "disengage"
                elif kl in ("1", "2"):
                    side = "left" if kl == "1" else "right"
                    _set_engage(ctrl, side, not ctrl.engaged(side), last_frame, T_wv)
                    last_msg = f"{side}: {'engaged' if ctrl.engaged(side) else 'idle'}"
                elif kl == "c":
                    last_msg = "re-centered" if recenter(last_frame) else "cannot re-center: no head pose yet"
                elif kl == "r":
                    rec.discard(); obs, share = new_episode(); step = 0; t_ep0 = time.time(); last_msg = "reset"
                elif kl == "s":
                    ok = bool(raw.success_buf[0].item())
                    if args_cli.require_success and not ok:
                        last_msg = "not saved: no success"
                    else:
                        p = rec.finish(success=ok); last_msg = f"saved {p.name if p else 'nothing'}"
                        print(f"[teleop] {last_msg}", flush=True)
                        obs, share = new_episode(); step = 0; t_ep0 = time.time()
                elif kl == "d":
                    rec.discard(); obs, share = new_episode(); step = 0; t_ep0 = time.time(); last_msg = "discarded"

            # ---- VR input ----
            _t0 = time.perf_counter()
            f = vr.poll(t_now)
            if f is not None:
                last_frame = f
                if args_cli.embody and not centered and f.head is not None:
                    recenter(f); last_msg = "centered on robot head"
            stale = vr.stale(t_now)
            if stale:
                for side in ("right", "left"):
                    if ctrl.engaged(side) and (last_frame is None or (t_now - last_frame.t_wall) > 1.0):
                        ctrl.disengage(side); last_msg = "auto-disengage (VR stale)"
            hand_pose_w = {"right": None, "left": None}
            tracked = {"right": False, "left": False}
            if last_frame is not None:
                for side in ("right", "left"):
                    joints = getattr(last_frame, side)
                    if joints is None:
                        continue
                    tracked[side] = True
                    Tw = transform_pose(T_wv, joints[XR_WRIST])
                    hand_pose_w[side] = (Tw[:3, 3].copy(), Tw[:3, :3].copy())
                    # retarget fingers only when a NEW frame arrived (DexPilot ~5 ms per hand)
                    if f is not None and ((not args_cli.hand_follow_clutch) or ctrl.engaged(side)):
                        mp = mp21_to_operator_wrist(xr25_to_mp21(joints), side)
                        q_canon = retarget[side].retarget(mp)
                        ctrl.set_hand(side, canonical_to_slots(q_canon, ctrl.slot_of_canon[side]))
                if args_cli.auto_engage:
                    for side in ("right", "left"):
                        if tracked[side] and not ctrl.engaged(side):
                            _set_engage(ctrl, side, True, last_frame, T_wv)
                if last_frame.head is not None:
                    view.set_head_pose(transform_pose(T_wv, last_frame.head))

            # ---- act ----
            _t1 = time.perf_counter()
            act, raw_act = ctrl.compute(hand_pose_w)
            assert torch.isfinite(act).all(), "non-finite action"
            step_dict = _snapshot(snap, obs, share, act, raw_act, last_frame, ctrl, t_now)
            _t2 = time.perf_counter()
            next_obs, _, next_share, _, done, _, _ = env.step(act.transpose(0, 1).contiguous(), step)
            _t3 = time.perf_counter()
            rec.append(step_dict)
            obs, share = next_obs, next_share
            step += 1

            # ---- view ----
            if step % max(1, args_cli.image_every) == 0:
                hud = [
                    f"L:{'ENGAGED' if ctrl.engaged('left') else 'idle'}  R:{'ENGAGED' if ctrl.engaged('right') else 'idle'}"
                    f"   track L{'+' if tracked['left'] else '-'} R{'+' if tracked['right'] else '-'}",
                    f"step {step}  t={time.time() - t_ep0:5.1f}s  REC {rec.length}"
                    + ("  SUCCESS" if bool(raw.success_buf[0].item()) else ""),
                    ("VR STALE  " if stale else "") + ("NO HEAD (press C when tracked)  " if args_cli.embody and not centered else "") + last_msg,
                ]
                stereo_pub.publish(view.capture(ctrl_dt, hud))
            _t4 = time.perf_counter()
            if args_cli.profile:
                prof["input"] += _t1 - _t0; prof["compute"] += _t2 - _t1; prof["env_step"] += _t3 - _t2
                prof["capture"] += _t4 - _t3; prof["n"] += 1
                if prof["n"] % 120 == 0:
                    n = prof["n"]
                    print(f"[teleop][profile] per step ms: input {1e3*prof['input']/n:.1f} | compute+snapshot "
                          f"{1e3*prof['compute']/n:.1f} | env.step {1e3*prof['env_step']/n:.1f} | capture "
                          f"{1e3*prof['capture']/n:.1f} (avg over {n} steps)", flush=True)

            # ---- auto done (timeout / object fell) ----
            is_done = bool(done[0].any()) if hasattr(done, "any") else bool(done)
            if is_done:
                p = rec.finish(success=bool(raw.success_buf[0].item()))
                last_msg = f"episode ended by env -> {p.name if p else 'nothing'}"
                print(f"[teleop] {last_msg}", flush=True)
                obs, share = new_episode(); step = 0; t_ep0 = time.time()

            if args_cli.steps and step >= args_cli.steps:
                p = rec.finish(success=bool(raw.success_buf[0].item()))
                print(f"[teleop] steps reached, saved {p}", flush=True)
                break

            # ---- pacing ----
            if args_cli.rt_factor > 0:
                budget = ctrl_dt / args_cli.rt_factor
                dt_left = budget - (time.time() - t_now)
                if dt_left > 0:
                    time.sleep(dt_left)

    if rec.active:
        p = rec.abort()
        print(f"[teleop] aborted episode written to {p}", flush=True)
    vr.close(); stereo_pub.close()
    if hasattr(env, "close"):
        env.close()
    print("[teleop] bye", flush=True)


if __name__ == "__main__":
    rc = 0
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        rc = 1
    finally:
        # Kit teardown is known to hang (timeline/stop callbacks); give it a few seconds, then hard-exit.
        import threading
        threading.Timer(10.0, lambda: os._exit(rc)).start()
        try:
            simulation_app.close()
        finally:
            os._exit(rc)

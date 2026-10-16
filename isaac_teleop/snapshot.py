"""Per-step recorder snapshot shared by teleop_handover_cloudxr.py and teleop_handover_vr.py.

Everything the augmentation pipeline needs (spec 2026-09-09 §2.2): wrist AND palm pose, the wrist
target actually commanded, the FSM stage, fingertip positions and the env success flag -- all in the
env-local frame (env origin subtracted), agent0 = RIGHT, agent1 = LEFT. The VR-source keys
(vr_head / vr_left / ...) are added by each script since their sources differ.
Requires the env root (isaac_teleop.paths.sim_root()) on sys.path for trl_utils.palm_frame; no Isaac import here so it is unit-testable.
"""
from __future__ import annotations

import numpy as np
import torch

from trl_utils.palm_frame import FINGERTIP_LINKS, fingertips_from_links, palm_from_wrist3

_F32 = np.float32


class SnapshotSource:
    def __init__(self, raw_env, ctrl):
        self.raw = raw_env
        self.ctrl = ctrl
        self.tip_ids = {}
        for side, robot in (("right", raw_env.right_robot), ("left", raw_env.left_robot)):
            ids, _ = robot.find_bodies([f"^{n}$" for n in FINGERTIP_LINKS[side]], preserve_order=True)
            if len(ids) != 4:
                raise RuntimeError(f"{side}: expected 4 fingertip parent links {FINGERTIP_LINKS[side]}, got {ids}")
            self.tip_ids[side] = ids

    def _fingertips(self, side: str, off: torch.Tensor) -> np.ndarray:
        robot = self.raw.right_robot if side == "right" else self.raw.left_robot
        ids = self.tip_ids[side]
        p = (robot.data.body_pos_w[0, ids] - off).detach().cpu().numpy()
        q = robot.data.body_quat_w[0, ids].detach().cpu().numpy()
        return fingertips_from_links(p, q)

    def base(self, obs, share, act, raw_act, t_now: float) -> dict:
        raw = self.raw
        off = raw.env_origins[0]
        off_np = off.detach().cpu().numpy().astype(np.float64)
        jp = torch.stack([raw.right_robot.data.joint_pos[0], raw.left_robot.data.joint_pos[0]]).cpu().numpy()
        jv = torch.stack([raw.right_robot.data.joint_vel[0], raw.left_robot.data.joint_vel[0]]).cpu().numpy()
        pR, qR, pL, qL = self.ctrl.teleop._compute_frame_pose_world()
        ee_pos = torch.stack([pR[0] - off, pL[0] - off]).cpu().numpy().astype(np.float64)
        ee_quat = torch.stack([qR[0], qL[0]]).cpu().numpy().astype(np.float64)
        palm = [palm_from_wrist3(ee_pos[i], ee_quat[i], side) for i, side in enumerate(("right", "left"))]
        tgt = [self.ctrl.target_pose_world(side) for side in ("right", "left")]
        return {
            "observations": obs[0].detach().cpu().numpy().astype(_F32),
            "share_observations": share[0].detach().cpu().numpy().astype(_F32),
            "actions": act[0].detach().cpu().numpy().astype(_F32),
            "raw_actions": raw_act[0].detach().cpu().numpy().astype(_F32),
            "joint_positions": jp.astype(_F32), "joint_velocities": jv.astype(_F32),
            "object_pos": (raw.cube.data.root_pos_w[0, :3] - off).cpu().numpy().astype(_F32),
            "object_quat": raw.cube.data.root_quat_w[0].cpu().numpy().astype(_F32),
            "ee_pos": ee_pos.astype(_F32), "ee_quat": ee_quat.astype(_F32),
            "palm_pos": np.stack([p for p, _ in palm]).astype(_F32),
            "palm_quat": np.stack([q for _, q in palm]).astype(_F32),
            "target_wrist_pos": np.stack([np.asarray(p, dtype=np.float64) - off_np for p, _ in tgt]).astype(_F32),
            "target_wrist_quat": np.stack([np.asarray(q, dtype=np.float64) for _, q in tgt]).astype(_F32),
            "fingertip_pos": np.stack([self._fingertips("right", off), self._fingertips("left", off)]).astype(_F32),
            "subgoal": raw.subgoal[0].detach().cpu().numpy().astype(_F32),
            "success_env_step": bool(raw.success_buf[0].item()),
            "engaged": np.array([int(self.ctrl.engaged("right")), int(self.ctrl.engaged("left"))], np.int8),
            "t_wall": float(t_now),
        }

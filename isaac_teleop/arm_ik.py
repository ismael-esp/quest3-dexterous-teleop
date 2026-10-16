"""Wrist targets (world) + hand joint targets -> per-agent 22-D raw (rad) and scaled [-1,1] actions.

Reuses the thesis Differential-IK teleop (trl_utils.dualarm_ik_controller.DualArmTeleop) exactly
like trl_utils/targets_to_actions.py, extended to the 44-D dexterous contract.
Requires the env root (isaac_teleop.paths.sim_root()) on sys.path and an existing DirectRLEnv (raw_env).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import isaaclab.utils.math as math_utils

from trl_utils.dualarm_ik_controller import (
    DualArmTeleop, make_default_left_gripper_mapping, make_default_right_gripper_mapping,
)
from trl_utils.scaling import scale_joint_to_action

from .frames import AnchoredMapper, mat_to_quat_wxyz, quat_wxyz_to_mat


def _clamp_norm(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0:
        return vec
    n = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(1e-9)
    return vec * torch.clamp(max_norm / n, max=1.0)


@dataclass
class ArmState:
    mapper: AnchoredMapper
    last_q: Optional[torch.Tensor] = None      # (6,) last commanded arm joints [rad]
    last_hand: Optional[np.ndarray] = None     # (16,) last hand joints, Isaac slot order [rad]
    prev_tgt: Optional[np.ndarray] = None      # (3,) previous mapped target position (for velocity)
    vel: Optional[np.ndarray] = None           # (3,) EMA-filtered target velocity [m/s]
    last_target_p: Optional[np.ndarray] = None # (3,) wrist target actually sent to IK, world [m]
    last_target_q: Optional[np.ndarray] = None # (4,) its orientation, WXYZ


class DualArmVRController:
    def __init__(self, raw_env, ee_link_right: str = "ra_wrist_3_link", ee_link_left: str = "la_wrist_3_link",
                 pos_scale: float = 1.0, clamp_pos: float = 0.10, clamp_rot: float = 0.50,
                 rot_weight: float = 1.0, max_ee_speed: float = 1.5, dt: float = 1.0 / 60.0,
                 ff_tau: float = 0.0, max_joint_speed: float = 3.1416):
        self.raw = raw_env
        self.device = raw_env.device
        self.N = int(raw_env.num_envs)
        self.teleop = DualArmTeleop(right_robot=raw_env.right_robot, left_robot=raw_env.left_robot,
                                    device=self.device, num_envs=self.N,
                                    right_ee_body_name=ee_link_right, left_ee_body_name=ee_link_left)
        self.rmap = DualArmTeleop.build_joint_name_to_idx(raw_env.right_robot)
        self.lmap = DualArmTeleop.build_joint_name_to_idx(raw_env.left_robot)
        self.clamp_pos, self.clamp_rot, self.rot_weight = float(clamp_pos), float(clamp_rot), float(rot_weight)
        self.max_step = float(max_ee_speed) * float(dt)          # metres per control step
        self.max_dq = float(max_joint_speed) * float(dt)         # rad per control step per arm joint
        self.dt = float(dt)
        self.limits_hit = {"right": "", "left": ""}              # per-frame diagnostic ("" = free)
        self.ff_tau = float(ff_tau)          # velocity feed-forward horizon [s]: target += tau * vel
        self.arms = {"right": ArmState(AnchoredMapper(pos_scale)), "left": ArmState(AnchoredMapper(pos_scale))}
        f = lambda x: torch.as_tensor(x, device=self.device, dtype=torch.float32)
        self.r_lo, self.r_hi = f(raw_env.right_dof_lower_limits[:6]), f(raw_env.right_dof_upper_limits[:6])
        self.l_lo, self.l_hi = f(raw_env.left_dof_lower_limits[:6]), f(raw_env.left_dof_upper_limits[:6])
        self.h_lo, self.h_hi = f(raw_env.hand_dof_lower_limits), f(raw_env.hand_dof_upper_limits)
        self.slot_of_canon = {
            "right": [int(i) for i in np.asarray(raw_env.right_hand_slot_of_canon).tolist()],
            "left": [int(i) for i in np.asarray(raw_env.left_hand_slot_of_canon).tolist()],
        }
        self.reset_hold()

    # ---- state ----
    def reset_hold(self) -> None:
        """Hold the robots where they are now (after env reset)."""
        for side, robot in (("right", self.raw.right_robot), ("left", self.raw.left_robot)):
            ids = robot.actuators["ur3_arm"].joint_indices
            self.arms[side].last_q = robot.data.joint_pos[0, ids].clone()
            self.arms[side].last_hand = self.h_lo.cpu().numpy().copy()
            self.arms[side].mapper.disengage()

    def ee_pose_world(self, side: str) -> Tuple[np.ndarray, np.ndarray]:
        pR, qR, pL, qL = self.teleop._compute_frame_pose_world()
        p, q = (pR, qR) if side == "right" else (pL, qL)
        return p[0].detach().cpu().numpy().astype(np.float64), quat_wxyz_to_mat(q[0].detach().cpu().numpy())

    def engage(self, side: str, p_hand_w: np.ndarray, R_hand_w: np.ndarray, R_view=None) -> None:
        p_e, R_e = self.ee_pose_world(side)
        self.arms[side].mapper.engage(p_hand_w, R_hand_w, p_e, R_e, R_view=R_view)

    def disengage(self, side: str) -> None:
        self.arms[side].mapper.disengage()
        self.arms[side].prev_tgt = None
        self.arms[side].vel = None

    def engaged(self, side: str) -> bool:
        return self.arms[side].mapper.engaged

    def target_pose_world(self, side: str) -> Tuple[np.ndarray, np.ndarray]:
        """Wrist target last handed to the Diff-IK (world pos (3,), WXYZ quat (4,)).
        Falls back to the current EE pose before the first compute() (i.e. holding)."""
        st = self.arms[side]
        if st.last_target_p is None or st.last_target_q is None:
            p, R = self.ee_pose_world(side)
            return p, np.asarray(mat_to_quat_wxyz(R), dtype=np.float64)
        return st.last_target_p, st.last_target_q

    def set_hand(self, side: str, q_slots: Optional[np.ndarray]) -> None:
        """Latest hand target (16, Isaac slot order, rad) or None to hold."""
        if q_slots is not None:
            self.arms[side].last_hand = np.clip(np.asarray(q_slots, dtype=np.float64),
                                                self.h_lo.cpu().numpy(), self.h_hi.cpu().numpy())

    # ---- main ----
    @torch.no_grad()
    def compute(self, hand_pose_w: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """hand_pose_w: {'right': (p (3,), R (3,3)) | None, 'left': ...} current wrist poses in world.
        Returns (actions (1,2,22) in [-1,1], raw (1,2,22) rad). Disengaged/untracked arms hold."""
        pR, qR, pL, qL = self.teleop._compute_frame_pose_world()
        dpose = {}
        for side, (p_cur, q_cur) in (("right", (pR, qR)), ("left", (pL, qL))):
            st = self.arms[side]
            hp = hand_pose_w.get(side)
            if st.mapper.engaged and hp is not None:
                p_t, R_t = st.mapper.target(hp[0], hp[1])
                # velocity feed-forward: lead the target by ff_tau so the PD-limited arm chases less
                if self.ff_tau > 0.0:
                    if st.prev_tgt is not None:
                        v_raw = (p_t - st.prev_tgt) / self.dt
                        st.vel = v_raw if st.vel is None else 0.3 * v_raw + 0.7 * st.vel
                    st.prev_tgt = p_t.copy()
                    if st.vel is not None:
                        p_t = p_t + self.ff_tau * st.vel
                p_t_t = torch.as_tensor(p_t, device=self.device, dtype=torch.float32).view(1, 3)
                q_t_t = torch.as_tensor(mat_to_quat_wxyz(R_t), device=self.device, dtype=torch.float32).view(1, 4)
                q_t_t = torch.where((q_t_t * q_cur).sum(-1, keepdim=True) < 0, -q_t_t, q_t_t)
                st.last_target_p = np.asarray(p_t, dtype=np.float64).reshape(3).copy()
                st.last_target_q = q_t_t[0].detach().cpu().numpy().astype(np.float64)
                dp = _clamp_norm(p_t_t - p_cur, min(self.clamp_pos, self.max_step))
                dr = math_utils.axis_angle_from_quat(math_utils.quat_mul(q_t_t, math_utils.quat_inv(q_cur))) * self.rot_weight
                dr = _clamp_norm(dr, self.clamp_rot)
                dpose[side] = torch.cat([dp, dr], dim=-1)
            else:
                st.last_target_p = p_cur[0].detach().cpu().numpy().astype(np.float64)
                st.last_target_q = q_cur[0].detach().cpu().numpy().astype(np.float64)
                dpose[side] = None
        zero6 = torch.zeros((self.N, 6), device=self.device)
        one = torch.ones((self.N, 1), device=self.device)       # grip=+1 (open) -> ignored, we set hands ourselves
        act14 = torch.cat([dpose["right"] if dpose["right"] is not None else zero6, one,
                           dpose["left"] if dpose["left"] is not None else zero6, one], dim=-1)
        _, _, r_demo, l_demo = self.teleop.step(
            actions=act14, right_actions_buf=self.raw.right_actions.clone(), left_actions_buf=self.raw.left_actions.clone(),
            right_joint_name_to_idx=self.rmap, left_joint_name_to_idx=self.lmap,
            right_gripper_map=make_default_right_gripper_mapping(), left_gripper_map=make_default_left_gripper_mapping())
        for side, demo in (("right", r_demo), ("left", l_demo)):
            st = self.arms[side]
            flags = []
            if dpose[side] is not None:
                lo, hi = (self.r_lo, self.r_hi) if side == "right" else (self.l_lo, self.l_hi)
                q_new = demo[0, :6].clone()
                q_lim = torch.max(torch.min(q_new, hi), lo)
                if not torch.allclose(q_lim, q_new, atol=1e-6):
                    flags.append("joint-limit")
                if self.max_dq > 0 and st.last_q is not None:
                    dq = q_lim - st.last_q
                    dq_lim = dq.clamp(-self.max_dq, self.max_dq)
                    if not torch.allclose(dq_lim, dq, atol=1e-6):
                        flags.append("vel-limit")
                    q_lim = st.last_q + dq_lim
                st.last_q = q_lim
                # step saturated: operator's wrist is outrunning the arm
                # (out of reach, singular pose, or the joint/vel limits above)
                if float(torch.linalg.norm(dpose[side][0, :3])) >= 0.95 * min(self.clamp_pos, self.max_step):
                    flags.append("target-far")
            self.limits_hit[side] = "+".join(flags)
        r_arm = torch.max(torch.min(self.arms["right"].last_q.view(1, 6), self.r_hi), self.r_lo)
        l_arm = torch.max(torch.min(self.arms["left"].last_q.view(1, 6), self.l_hi), self.l_lo)
        r_hand = torch.as_tensor(self.arms["right"].last_hand, device=self.device, dtype=torch.float32).view(1, 16)
        l_hand = torch.as_tensor(self.arms["left"].last_hand, device=self.device, dtype=torch.float32).view(1, 16)

        raw = torch.zeros((1, 2, 22), device=self.device)
        raw[:, 0, :6], raw[:, 0, 6:] = r_arm, r_hand
        raw[:, 1, :6], raw[:, 1, 6:] = l_arm, l_hand
        act = torch.zeros((1, 2, 22), device=self.device)
        act[:, 0, :6] = scale_joint_to_action(r_arm, self.r_lo, self.r_hi)
        act[:, 1, :6] = scale_joint_to_action(l_arm, self.l_lo, self.l_hi)
        act[:, 0, 6:] = scale_joint_to_action(r_hand, self.h_lo, self.h_hi)
        act[:, 1, 6:] = scale_joint_to_action(l_hand, self.h_lo, self.h_hi)
        act = torch.nan_to_num(act.clamp(-1.0, 1.0), nan=0.0)
        return act, raw

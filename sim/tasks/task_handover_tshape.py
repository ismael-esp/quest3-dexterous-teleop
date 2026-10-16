# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn.functional as F
from typing import Sequence
import numpy as np
from gymnasium import spaces
from PIL import Image
import cv2
import os
import math
from pathlib import Path
from typing import Optional
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv, mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, quat_mul  # quat_rotate
from isaaclab.actuators import ImplicitActuatorCfg, ActuatorBaseCfg, ImplicitActuator
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.schemas import ArticulationRootPropertiesCfg
from trl_utils.traj_reward_mgr import TrajectoryRewardManager
from trl_utils.dense_shaping import potential_shaping, stage_potential
from trl_utils.common_cfg import (
    BaseDualArmTableEnvCfg,
    BaseTshapeEnvCfg,
    make_back_wall_cfg,
    make_left_wall_cfg,
    make_right_wall_cfg,
)
from trl_utils.EventCfg import EventCfg
from trl_utils.scaling import (
    scale_joint_to_action,
    unscale_action_to_joint,
    scale_hand_to_action,
    unscale_action_to_hand,
)

CURRENT_TASK_NAME = "handover_tshape"


def _quat_mul_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _jitter_quat_wxyz(quats: torch.Tensor, rot_deg_range: float) -> torch.Tensor:
    if rot_deg_range <= 0.0:
        return quats
    angles = (torch.rand((quats.shape[0], 3), device=quats.device, dtype=quats.dtype) * 2.0 - 1.0)
    angles = angles * math.radians(float(rot_deg_range))
    half = angles * 0.5
    sr, sp, sy = torch.sin(half).unbind(dim=-1)
    cr, cp, cy = torch.cos(half).unbind(dim=-1)
    delta = torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )
    return F.normalize(_quat_mul_wxyz(quats, delta), dim=-1)


def _make_handover_tshape_left_wall_cfg() -> RigidObjectCfg:
    cfg = make_left_wall_cfg()
    try:
        cfg.spawn.visual_material.diffuse_color = (0.161, 0.467, 0.796) #(0.35, 0.66, 1.00)
    except Exception:
        pass
    return cfg


def _make_handover_tshape_right_wall_cfg() -> RigidObjectCfg:
    cfg = make_right_wall_cfg()
    try:
        cfg.spawn.visual_material.diffuse_color = (0.74, 0.90, 1.00)
    except Exception:
        pass
    return cfg


def _make_handover_tshape_back_wall_cfg() -> RigidObjectCfg:
    cfg = make_back_wall_cfg()
    try:
        cfg.spawn.visual_material.diffuse_color = (0.03, 0.03, 0.03)
    except Exception:
        pass
    return cfg


@configclass
class HandoverTshapeEventCfg(EventCfg):
    goal_position_randomization = None

    handover_right_robot_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    handover_left_robot_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("left_robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    handover_object_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    handover_object_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    handover_right_robot_pd_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_robot", joint_names=".*"),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    handover_left_robot_pd_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("left_robot", joint_names=".*"),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    handover_right_robot_joint_friction = EventTermCfg(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_robot", joint_names=".*"),
            "friction_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    handover_left_robot_joint_friction = EventTermCfg(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("left_robot", joint_names=".*"),
            "friction_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


def _handover_robot_material_event(asset_name: str) -> EventTermCfg:
    return EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(asset_name, body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )


def _handover_object_material_event() -> EventTermCfg:
    return EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )


def _handover_object_mass_event() -> EventTermCfg:
    return EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


def _handover_pd_event(asset_name: str) -> EventTermCfg:
    return EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(asset_name, joint_names=".*"),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


def _handover_joint_friction_event(asset_name: str) -> EventTermCfg:
    return EventTermCfg(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(asset_name, joint_names=".*"),
            "friction_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


# --- Dexterous-hand dimension bookkeeping ------------------------------------------
# Referenced by TaskEnvCfg._sync_full_hand_action_dims(); kept here so the arithmetic
# behind the 44 -> 74 observation growth is stated once, in one place.
_LEGACY_OBS_DIM = 44                      # see the breakdown on observation_space below
_GRIP_OBS_DIM_PER_AGENT = 1               # legacy: one grip scalar per hand
_HAND_OBS_DIM_PER_AGENT = 16              # dexterous: all 16 Allegro joints per hand
_FULL_HAND_ACTION_DIM_PER_AGENT = 22      # arm6 + hand16
_RELATIVE_OBS_EXTRA_DIM = 47              # 30 rel vectors + 8 EE quat + 9 motion deltas
_SYNERGY_ACTION_DIM_PER_AGENT = 7         # arm6 + 1 grasp-synergy coefficient
# Allegro joints in CANONICAL (URDF) numbering: joint_<k>_0.
#   finger 0 = index (0..3), 1 = middle (4..7), 2 = ring (8..11), 3 = thumb (12..15)
#   within each finger, the first joint is abduction/opposition and the rest are flexion.
_CANON_ABDUCTION = (0, 4, 8)        # finger spread; aims the hand, does not close it
_CANON_THUMB_OPPOSE = 12            # positions the thumb before closing
_CANON_FLEX = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15)
#
# WARNING: Isaac does NOT lay the hand out as joint_0_0..joint_15_0. `robot.joint_names`
# is ordered breadth-first by joint depth, which INTERLEAVES the fingers:
#     slot 0..3   -> joint_0_0, joint_12_0, joint_4_0, joint_8_0
#     slot 4..7   -> joint_1_0, joint_13_0, joint_5_0, joint_9_0   ... etc.
# Hard-coding canonical numbers as slot indices therefore scrambles the posture -- the
# index finger's flexion joints silently receive abduction/opposition values, so the
# index never closes. Always resolve slots BY NAME via hand_slot_map().


def hand_slot_map(joint_names, prefix: str) -> list[int]:
    """Canonical joint_<k>_0 -> slot within the 16-wide hand slice, resolved by NAME.

    Returns a 16-list `m` where `m[k]` is the slot holding `<prefix>_joint_<k>_0`, so
    `hand_isaac[m] = hand_canonical` reorders a canonical posture into Isaac's layout.
    Module-level rather than a method because @configclass does not preserve
    staticmethods on the cfg class.
    """
    hand = list(joint_names)[6:22]
    try:
        return [hand.index(f"{prefix}_joint_{k}_0") for k in range(16)]
    except ValueError as exc:
        raise RuntimeError(
            f"could not resolve Allegro joint slots for prefix '{prefix}'; "
            f"hand slice was {hand}"
        ) from exc


@configclass
class TaskEnvCfg(BaseTshapeEnvCfg):
    task_name = CURRENT_TASK_NAME
    task_name = "handover_tshape"
    episode_length: int = 530
    # action_dim_per_agent: int = 7
    action_space: int = 7   # 14
    # 44 = ee_pos(3+3) + cube_pos(3) + progress/sin/cos(3) + cube_rotation_6d(6)
    #    + arm_qpos(12) + grip_qpos(2) + arm_qvel(12)
    observation_space: int = 44
    share_observation_space: int = 44

    # --- Dexterous hand action mode (2026-08-10) ----------------------------------
    # False (legacy): 7-D per agent = arm6 + ONE grip scalar fanned out to 16 Allegro
    #                 joints. The hand is effectively a parallel gripper.
    # True  (dexterous): 22-D per agent = arm6 + 16 independent Allegro joints, and the
    #                 observation carries the full 16-joint hand state per agent.
    #
    # Keeping both behind a flag preserves the 7-D baseline, which is exactly the
    # "is the dexterity load-bearing?" ablation the thesis needs. Dims are reconciled
    # in apply_runtime_mode() -> _sync_full_hand_action_dims(); do not hand-edit
    # action_space/observation_space to switch modes.
    handover_full_hand_action: bool = False
    # Relative/goal-conditioned features: per-EE vectors to object, other hand, handover
    # point, goal and home (+30), EE quaternions (+8), motion deltas (+9) = +47 dims.
    # The task's waypoints are otherwise unobservable -- they live only as thresholds in
    # _get_rewards -- so without this the policy must infer targets it cannot see.
    handover_relative_obs: bool = False
    # 1-D grasp synergy. Requires handover_full_hand_action (it keeps the dexterous
    # 74-dim observation and the dexterous reward gates) but collapses the ACTION back to
    # 7/agent: 6 arm + one scalar that travels the grasp library's pregrasp->grasp line
    # and still commands all 16 Allegro joints (trl_utils/dexterous_demo.py::
    # synergy_finger_targets). Motivation: the teacher scored 72% at 14-D total and 3-4%
    # at 44-D, and no demo ever moves the fingers off that line, so the 32 extra
    # dimensions are pure exploration cost. Observe 16, command 1.
    handover_synergy_action: bool = False
    # Mean flexion (rad) separating "open" from "closed" in dexterous mode. Sits between
    # the library's pregrasp (~0.32) and grasp (~1.29) postures. Only used when
    # handover_full_hand_action is True.
    handover_dex_flex_threshold: float = 0.80
    # Potential-based dense shaping over the staged reward (trl_utils/dense_shaping.py).
    # The staged reward is 13 binary bonuses and nothing in between, so there is no
    # gradient just outside a threshold. F = gamma*phi(s') - phi(s) with
    # phi = -coef * distance-to-current-stage-target is policy-preserving, so this
    # cannot change the optimum — only how reachable it is. Off preserves every
    # result recorded before 2026-08-12.
    # Stage 2->3 receive gate: how close the ROD (a point 11 cm along its axis, not the
    # hand) must come to the fixed receive point for the handover to register. Was a
    # hardcoded 0.03 -- the only sub-0.05 gate in the task, and tighter than the expert's
    # own repeatability (see the comment at the stage-2 reward).
    handover_receive_threshold: float = 0.05
    # Goal-placement gate: how close the ROD must be to the final target. The rod is
    # RELEASED here, not precision-placed, so exact position is not part of the task -- a
    # release from slightly off still drops it on target. Was `threshold` (0.05); after the
    # stage-2 gate was widened the failures relocated here (13-15 of 50 envs stuck at right
    # stage 3 carrying the rod but never registering arrival).
    handover_goal_threshold: float = 0.08
    handover_dense_shaping: bool = False
    handover_dense_shaping_coef: float = 0.5
    # Must match the PPO discount for the invariance proof to hold exactly.
    handover_dense_shaping_gamma: float = 0.99
    # Vision student contract with explicit arm velocity:
    # policy raw = right_hand(13) + left_hand(13) = 26   (finger qpos removed; arm_qpos6 + arm_qvel6 + grip1)
    # critic raw = right_hand(13) + left_hand(13) + shared_privileged(85) = 111
    vision_student_observation_space: int = 26
    vision_student_share_observation_space: int = 111
    # MAT vision-student proprioception ablation:
    # 14 = arm_qpos(12) + grip_open(2)
    # 26 = arm_qpos(12) + arm_qvel(12) + grip_open(2)
    # 46 = arm_qpos(12) + full_finger_qpos(32) + grip_open(2)
    handover_student_obs_dim: int = 26
    vision_student_hw_observation_space: int = 64
    vision_student_hw_share_observation_space: int = 149
    num_hand_obs: int = 0
    normalize_obs: bool = True
    joint_vel_scale: float = 10.0
    joint_obs_limits_path: str = "configs/handover_joint_obs_limits_p95.npz"
    handover_teacher_arm_velocity_obs: bool = True
    handover_arm_velocity_clip_to_obs_limits: bool = False
    use_finger_obs: bool = True
    handover_hw_small_obs: bool = False
    handover_no_walls: bool = False
    handover_use_real_scene: bool = False
    handover_real_scene_usd_path: str = "assets/background/final_without_robot.usd"
    handover_real_scene_prim_path: str = "/World/envs/env_0/RealScene"
    handover_real_scene_translate: tuple[float, float, float] = (0.83, 0.19, -0.13)
    handover_real_scene_rotate_zyx: tuple[float, float, float] = (4.7, -11.8, 13.5)
    handover_real_scene_scale: tuple[float, float, float] = (1.03, 1.03, 1.03)
    handover_table_visible: bool = True
    handover_left_wall_enabled: bool = True
    handover_back_wall_enabled: bool = True
    handover_randomize_tables: bool = False
    handover_table_asset_dir: str = "assets/motion_desk"
    handover_table_asset_glob: str = "*.usd"
    handover_randomize_wall_textures: bool = False
    handover_wall_texture_dir: str = "assets/texture_gray"
    handover_wall_texture_glob: str = "*.png"
    handover_wall_texture_scale: tuple[float, float] = (3.0, 3.0)
    handover_randomize_light: bool = False
    handover_light_intensity_range: tuple[float, float] = (1300.0, 1700.0)
    handover_light_color_range: tuple[float, float] = (0.90, 1.0)
    handover_randomize_object_pose: bool = True
    handover_object_pos_xy_range: tuple[float, ...] = (-0.05, 0.05, -0.05, 0.05)
    handover_object_pos_z_range: float = 0.005
    handover_object_yaw_deg_range: float | tuple[float, float] = 10.0
    handover_object_pose_bin_balanced: bool = False
    handover_object_pos_x_bins: int = 1
    handover_object_pos_y_bins: int = 1
    handover_object_yaw_bins: int = 1
    # --- object catalog (objects/catalog.py) ---
    # "rod" reproduces the historical cylinder exactly; "rod_slim" / "tumbler" are the P-A set.
    handover_object_id: str = "rod"
    # Length along the handover axis; set from the catalog in apply_runtime_mode(). The FSM
    # reference points scale with it (rod 0.30 -> receiver contact +0.11, giver offset -0.09).
    handover_object_axis_len: float = 0.30
    # >0 overrides episode_length (60 Hz steps). VR-demo BC passes the converter's T here.
    handover_episode_length: int = -1
    handover_domain_rand: bool = False
    handover_rand_object_mass: bool = False
    handover_object_mass_scale_range: tuple[float, float] = (0.75, 1.50)
    handover_rand_material: bool = False
    handover_object_static_friction_range: tuple[float, float] = (0.75, 1.30)
    handover_object_dynamic_friction_range: tuple[float, float] = (0.75, 1.30)
    handover_object_restitution_range: tuple[float, float] = (0.0, 0.10)
    handover_robot_static_friction_range: tuple[float, float] = (0.85, 1.20)
    handover_robot_dynamic_friction_range: tuple[float, float] = (0.85, 1.20)
    handover_robot_restitution_range: tuple[float, float] = (0.0, 0.05)
    handover_rand_pd_gains: bool = False
    handover_pd_stiffness_scale_range: tuple[float, float] = (0.90, 1.15)
    handover_pd_damping_scale_range: tuple[float, float] = (0.90, 1.15)
    handover_rand_joint_friction: bool = False
    handover_joint_friction_scale_range: tuple[float, float] = (0.85, 1.15)
    handover_rand_initial_joint_velocity: bool = False
    handover_initial_joint_velocity_range: float = 0.02
    handover_rand_obs_noise: bool = False
    handover_object_pos_noise_std: float = 0.002
    handover_object_pos_bias_range: float = 0.003
    handover_object_rot_noise_deg: float = 1.0
    handover_object_rot_bias_deg: float = 1.0
    handover_robot_pos_noise_std: float = 0.002
    handover_robot_pos_bias_range: float = 0.002
    handover_robot_vel_noise_std: float = 0.02
    handover_robot_vel_bias_range: float = 0.01
    # Sim/hardware gap fix: hardware thumb cannot extend below ~0.10 rad on the
    # opposition axis (ra_joint_12_0 / la_joint_12_0). Floor the simulated lower
    # bound so a fully-open grip command lands at this value instead of 0.0.
    handover_thumb_oppose_lower_limit: float = 0.5
    # ACTION SMOOTHING, applied in normalized action space before the joint mapping.
    # hw/view_policy_hw.py already applies exactly this (--view_action_ema 0.7 plus a
    # per-step slew clamp) before anything reaches the real robot, but sim has never had
    # it -- so the policy is trained and evaluated on raw 60 Hz targets and then deployed
    # through a filter it has never seen. Defaults keep sim behaviour unchanged.
    # Restore the pre-2026-08-28 dirty-reset behaviour (hand joints and object/support
    # velocities carried across episodes). Only for reproducing old runs; the carried
    # state made training episodes non-i.i.d. and let policies grow dependent on it.
    handover_legacy_dirty_reset: bool = False
    # Anchor the L_GRASP waypoint (subgoals 0/1 gates + their dense-shaping column) to the
    # rod's LIVE pose instead of the reset-time capture. The legacy initial-pose clobber
    # re-captured every env's anchor from the live rod whenever ANY env reset, so with
    # desynchronised resets L_GRASP effectively tracked the live rod -- that is what every
    # historical policy was trained and scored against. Pinning the anchor at the spawn
    # (the per-env capture fix) under-counts real handovers whenever the rod drifts >5 cm
    # before the grasp latches (teacher A/B 2026-08-28: 0.80 -> 0.64 from the metric alone).
    # Live anchoring is also the honest task semantics: grasp the rod where it IS.
    # MEASURED 2026-08-28 (teacher, fixed resets, 3 seeds): live anchor scored
    # 0.53-0.58 vs 0.63-0.66 spawn-anchored -- WORSE, likely because the full-quat
    # rotation of the 17 cm grasp offset swings the waypoint when contact tilts the
    # rod. Default stays OFF until understood; the flag is kept for experiments.
    handover_grasp_waypoint_live: bool = False
    # Decomposition control for the legacy initial-pose clobber (see _reset_idx):
    # -1 = follow handover_legacy_dirty_reset, 0 = per-env capture (fixed), 1 = legacy
    # all-env live re-capture. Lets eval separate the metric effect of the clobber
    # from the physical effects of the dirty resets.
    handover_legacy_initial_pose_clobber: int = -1
    # Decomposition controls for the two physical reset fixes.
    # handover_reset_hand_joints: -1 = follow handover_legacy_dirty_reset (fixed mode
    # -> hardware home pose, mode 4), 0 = no hand write (legacy behaviour), 1 = zeros
    # pos+vel, 2 = zeros position only, 3 = zero velocity only, 4 = hardware home
    # (fingers 0, thumb opposition 0.5). See the measured dose-response in _reset_idx.
    handover_reset_hand_joints: int = -1
    handover_reset_object_velocity: int = -1    # zero rod/supporter velocity at reset
    handover_action_ema: float = 0.0        # 0 = off; a_t = (1-w)*a_raw + w*a_prev
    handover_action_max_delta: float = 0.0  # 0 = off; per-step |a_t - a_prev| clamp
    handover_robot_collision_penalty: bool = True
    handover_robot_collision_arm_threshold: float = 0.03
    handover_robot_collision_hand_threshold: float = 0.005
    handover_robot_collision_arm_coef: float = 0.01
    handover_robot_collision_hand_coef: float = 0.005
    handover_black_tshape: bool = True
    handover_tshape_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")

    # ---------------------------------------------------------------------
    # Dual-mode execution contract
    #   use_vision=False : state teacher / PPO mode
    #   use_vision=True  : visual student / supervised distill mode
    # ---------------------------------------------------------------------
    use_vision: bool = False
    strict_vision_only_policy: bool = False
    enable_privileged_debug_dict: bool = False
    vision_mode: str = "rgbd_dual_resnet"
    vision_fusion: str = "concat"
    vision_feat_dim: int = 128
    vision_dropout: float = 0.0
    vision_frame_stack: int = 1
    vision_update_stride: int = 2
    vision_channels: int = 4
    vision_use_depth: bool = True
    vision_camera_mount: str = "head"
    vision_use_head_camera: bool = True
    vision_use_front_camera: bool = True
    vision_use_front_right_camera: bool = False
    vision_front_only: bool = True
    vision_student_raw_mode: bool = True
    vision_export_split_rgbd: bool = True
    distill_dual_obs: bool = False
    dextrah_faithful_mode: bool = False
    dextrah_trlp4_distill_mode: bool = False
    dextrah_right_img_flip: bool = False
    dextrah_export_stereo_masks: bool = True
    dextrah_camera_pose_rand_enable: bool = True
    dextrah_use_hardware_camera_pose: bool = True
    dextrah_hardware_camera_pose_mode: str = "view_target"
    dextrah_hardware_camera_direct_convention: str = "world"
    dextrah_hardware_camera_target_distance: float = 1.0
    dextrah_front_camera_rand_rot_deg: float = 1.0
    dextrah_front_camera_rand_pos_range: float = 0.01
    dextrah_front_right_camera_rand_rot_deg: float = 1.0
    dextrah_front_right_camera_rand_pos_range: float = 0.003
    vision_depth_noise_std: float = 0.0
    vision_depth_scale_jitter: float = 0.0
    vision_depth_dropout_prob: float = 0.0
    vision_depth_near: float = 0.30
    vision_depth_far: float = 1.20
    vision_camera_width: int = 320
    vision_camera_height: int = 240
    vision_resize_h: int = 240
    vision_resize_w: int = 320
    vision_obs_dim: int = 0
    vision_spatial_num_kp: int = 32
    vision_asymmetric_critic: bool = True
    vision_resnet_pretrained: bool = True
    vision_camera_eye_local: tuple[float, float, float] = (0.02, 0.13, 2.28)
    vision_camera_target_local: tuple[float, float, float] = (0.02, 0.42, 0.93)
    vision_camera_eye_jitter: tuple[float, float, float] = (0.00, 0.00, 0.00)
    vision_camera_target_jitter: tuple[float, float, float] = (0.00, 0.00, 0.00)
    vision_head_camera_eye_local: tuple[float, float, float] = (0.0005, 0.131, 1.644)
    vision_head_camera_target_local: tuple[float, float, float] = (0.02, 0.42, 0.93)
    vision_head_camera_eye_jitter: tuple[float, float, float] = (0.00, 0.00, 0.00)
    vision_head_camera_target_jitter: tuple[float, float, float] = (0.00, 0.00, 0.00)
    vision_head_camera_quat_wxyz: tuple[float, float, float, float] = (
        0.59224848,
        -0.41210622,
        0.40163118,
        0.56400585,
    )
    vision_front_camera_eye_local: tuple[float, float, float] = (0.018, 1.495, 1.544)
    vision_front_camera_target_local: tuple[float, float, float] = (0.134, 0.537, 1.281)
    vision_front_camera_eye_jitter: tuple[float, float, float] = (0.00, 0.00, 0.00)
    vision_front_camera_target_jitter: tuple[float, float, float] = (0.00, 0.00, 0.00)
    vision_front_camera_quat_wxyz: tuple[float, float, float, float] = (0.740, 0.104, 0.085, -0.659)
    vision_front_right_camera_eye_local: tuple[float, float, float] = (-0.218, 1.495, 1.544)
    vision_front_right_camera_target_local: tuple[float, float, float] = (-0.102, 0.537, 1.281)
    vision_front_right_camera_eye_jitter: tuple[float, float, float] = (0.00, 0.00, 0.00)
    vision_front_right_camera_target_jitter: tuple[float, float, float] = (0.00, 0.00, 0.00)
    vision_front_right_camera_quat_wxyz: tuple[float, float, float, float] = (0.740, 0.104, 0.085, -0.659)
    table_camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/WorkspaceCamera",
        update_period=1.0 / 60.0,
        height=48,
        width=48,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=33.2367757485,
            vertical_aperture=18.6661791289,
            clipping_range=(0.3, 10.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )
    head_camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/HeadWorkspaceCamera",
        update_period=1.0 / 60.0,
        height=48,
        width=48,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=33.2367757485,
            vertical_aperture=18.6661791289,
            clipping_range=(0.01, 10.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )
    front_camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/FrontWorkspaceCamera",
        update_period=1.0 / 60.0,
        height=48,
        width=48,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=33.2367757485,
            vertical_aperture=18.6661791289,
            clipping_range=(0.3, 10.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )
    front_right_camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/FrontRightWorkspaceCamera",
        update_period=1.0 / 60.0,
        height=48,
        width=48,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=33.2367757485,
            vertical_aperture=18.6661791289,
            clipping_range=(0.3, 10.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )

    front_wall_cfg: RigidObjectCfg | None = None
    left_wall_cfg: RigidObjectCfg | None = _make_handover_tshape_left_wall_cfg()
    right_wall_cfg: RigidObjectCfg | None = None
    back_wall_cfg: RigidObjectCfg | None = _make_handover_tshape_back_wall_cfg()

    # === object for task ===
    # cube
    # NOTE: despite the name, the handover object is the YELLOW T-SHAPED ROD
    # (see assets/). The identifier 'cube' is kept because it is a shared scene
    # key with the colorcube task; a rename is a separate mechanical pass.
    cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        spawn=sim_utils.CylinderCfg(
            radius=0.027,
            height=0.3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.0005,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color= (0.9, 0.8, 0.05),# (0.71, 0.42, 0.25),
            ),
                mass_props=sim_utils.MassPropertiesCfg(  # By default, mass is computed from volume with density=1, in kg.
                # Hard cardboard rod (30 cm x 5.4 cm). The real rod is a spiral-wound
                # tube; the collider here is a SOLID cylinder, so match the real object's
                # total MASS rather than a solid-cardboard density. A thick-walled hard
                # tube of this size is ~0.15-0.25 kg -> take the heavy end.
                # 0.25 kg / 687 cm^3 = 364 kg/m^3 effective (solid-equivalent).
                # Was 1 kg == 1456 kg/m^3, denser than glass.
                mass=0.25
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.2425, 0.45, 0.99),
            rot=(0.7071, 0.0, 0.7071, 0.0),
        ),
    )

    supporter_1_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/supporter_1",
        spawn=sim_utils.CuboidCfg(
            size=(0.005, 0.005, 0.007),
            visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.2425, 0.48, 0.95),
        ),
    )

    supporter_2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/supporter_2",
        spawn=sim_utils.CuboidCfg(
            size=(0.005, 0.005, 0.007),
            visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.2425, 0.42, 0.95),
        ),
    )

    # goal marker -> set position at _setup_scene func
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_markers",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.01,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0), opacity=0.5
                ),
            ),
        },
    )
    debug_success_region_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/debug/handover_tshape_success_region",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.5,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.85, 1.0), opacity=0.18
                ),
            ),
        },
    )
    debug_home_right_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/debug/handover_tshape_home_right",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.45, 0.1), opacity=0.75
                ),
            ),
        },
    )
    debug_home_left_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/debug/handover_tshape_home_left",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.35, 0.65, 1.0), opacity=0.75
                ),
            ),
        },
    )
    debug_object_receive_target_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/debug/handover_tshape_object_receive_target",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 1.0, 1.0), opacity=0.9
                ),
            ),
        },
    )
    debug_stage_left_grasp_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/debug/handover_tshape_stage_ball_1_left_grasp",
        markers={
            "ball": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.35, 0.65, 1.0), opacity=0.9
                ),
            ),
        },
    )
    debug_stage_handover_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/debug/handover_tshape_stage_ball_2_handover",
        markers={
            "ball": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.25, 0.9, 0.55), opacity=0.9
                ),
            ),
        },
    )
    debug_stage_right_grasp_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/debug/handover_tshape_stage_ball_3_right_grasp",
        markers={
            "ball": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.55, 0.15), opacity=0.9
                ),
            ),
        },
    )
    debug_stage_target_hold_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/debug/handover_tshape_stage_ball_4_target_hold",
        markers={
            "ball": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.85, 0.2), opacity=0.9
                ),
            ),
        },
    )
    debug_stage_success_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/debug/handover_tshape_stage_ball_5_success",
        markers={
            "ball": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.25, 1.0), opacity=0.9
                ),
            ),
        },
    )

    events: HandoverTshapeEventCfg = HandoverTshapeEventCfg()

    # action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    #     noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
    #     bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    # )
    # observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    #     noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
    #     bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    # )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=50, env_spacing=8, replicate_physics=True
    )

    simple_log_every: int = 0

    # scales and constants
    fall_dist = -1.0
    success_region_hold_steps: float = 5.0
    success_home_hold_steps: float = 10.0
    success_home_th: float = 0.08
    demo_handover_capture_step: int = 210
    terminate_on_success: bool = True
    show_goal_marker: bool = True
    enable_debug_markers: bool = False
    debug_stage_marker_z: float = 1.95
    debug_stage_marker_x_start: float = -0.35
    debug_stage_marker_x_gap: float = 0.10
    debug_stage_marker_y: float = -0.18

    def _sync_full_hand_action_dims(self):
        """Reconcile action/observation dims with `handover_full_hand_action`.

        Legacy (flag off): action 7/agent; obs 44, where the hand contributes a single
        grip scalar per agent (2 dims total).

        Dexterous (flag on): action 22/agent (arm6 + 16 Allegro joints); the hand's
        contribution to the observation grows from 1 to 16 dims per agent, so

            obs = 44 - (2 * 1) + (2 * 16) = 74

        The policy must be able to sense the hand it commands -- observing one grip
        scalar while commanding 16 joints leaves the finger state unobservable. (The
        2026-06-17 `task_handover_tshape_22.py` reference had exactly that asymmetry;
        it is deliberately not reproduced here.)
        """
        if not bool(getattr(self, "handover_full_hand_action", False)):
            return

        self.action_space = _FULL_HAND_ACTION_DIM_PER_AGENT
        if bool(getattr(self, "handover_synergy_action", False)):
            # Synergy: still the dexterous OBSERVATION (74, all 16 finger joints visible)
            # and the dexterous reward gates, but 7 action dims per agent. The hand stays
            # fully observable while the policy commands one closure coefficient.
            self.action_space = _SYNERGY_ACTION_DIM_PER_AGENT
        # NOTE: observation_space is NOT set here. Later branches of apply_runtime_mode
        # recompute it from the vision/state mode, so setting it here would be silently
        # overwritten. The state-mode branch adds the hand contribution itself.

    def _sync_wall_objects(self):
        self.handover_no_walls = False
        self.handover_left_wall_enabled = True
        self.handover_back_wall_enabled = True
        self.front_wall_cfg = None
        self.right_wall_cfg = None
        if self.left_wall_cfg is None:
            self.left_wall_cfg = _make_handover_tshape_left_wall_cfg()
        if self.back_wall_cfg is None:
            self.back_wall_cfg = _make_handover_tshape_back_wall_cfg()

        if not isinstance(getattr(self, "rigid_objects_cfg", None), dict):
            return

        wall_cfg_names = {
            "front_wall": "front_wall_cfg",
            "left_wall": "left_wall_cfg",
            "right_wall": "right_wall_cfg",
            "back_wall": "back_wall_cfg",
        }
        removed = []
        for object_name, cfg_name in wall_cfg_names.items():
            wall_cfg = getattr(self, cfg_name, None)
            if wall_cfg is None:
                if self.rigid_objects_cfg.pop(object_name, None) is not None:
                    removed.append(object_name)
            else:
                self.rigid_objects_cfg[object_name] = wall_cfg
        if removed:
            print(f"[handover_tshape] task walls disabled: {','.join(sorted(removed))}")

    def _sync_table_visibility_cfg(self):
        table_cfg = getattr(self, "table_cfg", None)
        table_spawn = getattr(table_cfg, "spawn", None)
        if table_spawn is None:
            return
        try:
            table_spawn.visible = bool(getattr(self, "handover_table_visible", True))
        except Exception:
            pass

    def _sync_vision_camera_cfg(self) -> tuple[int, float]:
        vision_stride = max(1, int(getattr(self, "vision_update_stride", 1)))
        control_dt = float(getattr(self.sim, "dt", 1.0 / 120.0)) * float(
            max(1, int(self.decimation))
        )
        camera_update_period = control_dt * float(vision_stride)
        for camera_cfg in (
            self.table_camera_cfg,
            self.head_camera_cfg,
            self.front_camera_cfg,
            self.front_right_camera_cfg,
        ):
            camera_cfg.update_period = camera_update_period
            camera_cfg.width = int(self.vision_camera_width)
            camera_cfg.height = int(self.vision_camera_height)
        self._vision_update_stride_effective = int(vision_stride)
        self._vision_update_hz_effective = float(1.0 / camera_update_period)
        return vision_stride, camera_update_period

    def _set_event_enabled(self, name: str, enabled: bool) -> None:
        if enabled:
            if getattr(self.events, name, None) is not None:
                return
            factories = {
                "handover_object_mass": _handover_object_mass_event,
                "handover_object_material": _handover_object_material_event,
                "handover_right_robot_material": lambda: _handover_robot_material_event("right_robot"),
                "handover_left_robot_material": lambda: _handover_robot_material_event("left_robot"),
                "handover_right_robot_pd_gains": lambda: _handover_pd_event("right_robot"),
                "handover_left_robot_pd_gains": lambda: _handover_pd_event("left_robot"),
                "handover_right_robot_joint_friction": lambda: _handover_joint_friction_event("right_robot"),
                "handover_left_robot_joint_friction": lambda: _handover_joint_friction_event("left_robot"),
            }
            factory = factories.get(name)
            if factory is not None:
                setattr(self.events, name, factory())
        elif getattr(self.events, name, None) is not None:
            setattr(self.events, name, None)

    def _sync_handover_domain_rand_events(self) -> None:
        enabled = bool(getattr(self, "handover_domain_rand", False))

        mass_on = enabled and bool(getattr(self, "handover_rand_object_mass", True))
        material_on = enabled and bool(getattr(self, "handover_rand_material", True))
        pd_on = enabled and bool(getattr(self, "handover_rand_pd_gains", True))
        joint_friction_on = enabled and bool(getattr(self, "handover_rand_joint_friction", True))

        self._set_event_enabled("handover_object_mass", mass_on)
        self._set_event_enabled("handover_object_material", material_on)
        self._set_event_enabled("handover_right_robot_material", material_on)
        self._set_event_enabled("handover_left_robot_material", material_on)
        self._set_event_enabled("handover_right_robot_pd_gains", pd_on)
        self._set_event_enabled("handover_left_robot_pd_gains", pd_on)
        self._set_event_enabled("handover_right_robot_joint_friction", joint_friction_on)
        self._set_event_enabled("handover_left_robot_joint_friction", joint_friction_on)

        def _range2(value, default):
            try:
                vals = tuple(float(v) for v in value)
            except TypeError:
                vals = (float(value), float(value))
            except Exception:
                vals = tuple(float(v) for v in default)
            if len(vals) < 2:
                vals = (vals[0], vals[0])
            return (float(vals[0]), float(vals[1]))

        if mass_on and getattr(self.events, "handover_object_mass", None) is not None:
            self.events.handover_object_mass.params["mass_distribution_params"] = _range2(
                getattr(self, "handover_object_mass_scale_range", (0.90, 1.10)),
                (0.90, 1.10),
            )

        if material_on:
            obj_params = {
                "static_friction_range": _range2(getattr(self, "handover_object_static_friction_range", (0.90, 1.10)), (0.90, 1.10)),
                "dynamic_friction_range": _range2(getattr(self, "handover_object_dynamic_friction_range", (0.90, 1.10)), (0.90, 1.10)),
                "restitution_range": _range2(getattr(self, "handover_object_restitution_range", (0.0, 0.05)), (0.0, 0.05)),
            }
            robot_params = {
                "static_friction_range": _range2(getattr(self, "handover_robot_static_friction_range", (0.90, 1.10)), (0.90, 1.10)),
                "dynamic_friction_range": _range2(getattr(self, "handover_robot_dynamic_friction_range", (0.90, 1.10)), (0.90, 1.10)),
                "restitution_range": _range2(getattr(self, "handover_robot_restitution_range", (0.0, 0.03)), (0.0, 0.03)),
            }
            if getattr(self.events, "handover_object_material", None) is not None:
                self.events.handover_object_material.params.update(obj_params)
            for name in ("handover_right_robot_material", "handover_left_robot_material"):
                event = getattr(self.events, name, None)
                if event is not None:
                    event.params.update(robot_params)

        if pd_on:
            pd_params = {
                "stiffness_distribution_params": _range2(getattr(self, "handover_pd_stiffness_scale_range", (0.95, 1.05)), (0.95, 1.05)),
                "damping_distribution_params": _range2(getattr(self, "handover_pd_damping_scale_range", (0.95, 1.05)), (0.95, 1.05)),
            }
            for name in ("handover_right_robot_pd_gains", "handover_left_robot_pd_gains"):
                event = getattr(self.events, name, None)
                if event is not None:
                    event.params.update(pd_params)

        if joint_friction_on:
            jf_params = {
                "friction_distribution_params": _range2(getattr(self, "handover_joint_friction_scale_range", (0.90, 1.10)), (0.90, 1.10)),
            }
            for name in ("handover_right_robot_joint_friction", "handover_left_robot_joint_friction"):
                event = getattr(self.events, name, None)
                if event is not None:
                    event.params.update(jf_params)

    def apply_runtime_mode(self):
        # Object catalog: mutate cube_cfg in place (rigid_objects_cfg["cube"] holds the same object).
        # This is the only cfg hook that runs AFTER task_registry.make_env applies CLI overrides.
        from objects.catalog import apply_to_cube_cfg, get_object
        _spec = get_object(str(getattr(self, "handover_object_id", "rod")))
        apply_to_cube_cfg(_spec, self.cube_cfg, self.BASE_DIR)
        # configclass deep-copies mutable fields, so the dict entry made in __post_init__ is a
        # stale COPY by now; the scene builds from rigid_objects_cfg, so re-point it.
        self.rigid_objects_cfg["cube"] = self.cube_cfg
        self.handover_object_axis_len = float(_spec.length_along_axis)
        if int(getattr(self, "handover_episode_length", -1)) > 0:
            self.episode_length = int(self.handover_episode_length)
            self.episode_length_s = float(self.episode_length) / 60.0
        self._sync_full_hand_action_dims()
        self._sync_wall_objects()
        self._sync_table_visibility_cfg()
        self._sync_handover_domain_rand_events()
        faithful_front_stereo = (
            bool(getattr(self, "dextrah_faithful_mode", False))
            or bool(getattr(self, "dextrah_faithful_profile", False))
        ) and bool(getattr(self, "vision_use_front_right_camera", False))
        export_stereo_masks = bool(getattr(self, "dextrah_export_stereo_masks", True))
        depth_stream_enabled = bool(getattr(self, "vision_use_depth", True)) or (
            faithful_front_stereo and export_stereo_masks
        )
        try:
            self.table_camera_cfg.data_types = ["rgb"] + (
                ["distance_to_image_plane"] if depth_stream_enabled else []
            )
        except Exception:
            pass
        try:
            self.head_camera_cfg.data_types = ["rgb"] + (
                ["distance_to_image_plane"] if depth_stream_enabled else []
            )
        except Exception:
            pass
        try:
            self.front_camera_cfg.data_types = ["rgb"] + (
                ["distance_to_image_plane"] if depth_stream_enabled else []
            )
        except Exception:
            pass
        try:
            self.front_right_camera_cfg.data_types = ["rgb"] + (
                ["distance_to_image_plane"] if depth_stream_enabled else []
            )
        except Exception:
            pass

        if bool(self.use_vision):
            self.strict_vision_only_policy = True
            self.enable_privileged_debug_dict = False
            student_obs_dim = int(getattr(self, "handover_student_obs_dim", 26))
            if student_obs_dim not in (14, 26, 46):
                raise ValueError(
                    f"handover_student_obs_dim must be one of 14, 26, 46; got {student_obs_dim}"
                )
            self.vision_student_observation_space = student_obs_dim
            # Preserve the same 85D privileged suffix for every policy-observation ablation.
            self.vision_student_share_observation_space = student_obs_dim + 85
            if bool(getattr(self, "dextrah_trlp4_distill_mode", False)):
                self.num_hand_obs = 0
                self.observation_space = 94
                self.share_observation_space = 94
            elif bool(getattr(self, "handover_hw_small_obs", False)):
                self.num_hand_obs = 32
                self.observation_space = int(self.vision_student_hw_observation_space)
                self.share_observation_space = int(self.vision_student_hw_share_observation_space)
            else:
                self.num_hand_obs = student_obs_dim // 2
                self.observation_space = int(self.vision_student_observation_space)
                self.share_observation_space = int(self.vision_student_share_observation_space)
            use_depth = bool(getattr(self, "vision_use_depth", True))
            front_only = bool(getattr(self, "vision_front_only", True))
            use_head = bool(getattr(self, "vision_use_head_camera", False))
            use_front = (front_only or bool(getattr(self, "vision_use_front_camera", False))) and (
                not (use_head and not bool(getattr(self, "vision_use_front_camera", False)))
            )
            use_top = (not front_only) and (not use_head)
            use_front_right = bool(getattr(self, "vision_use_front_right_camera", False))
            view_count = int(use_head) + int(use_top) + int(use_front) + int(use_front_right)
            if view_count <= 0:
                raise RuntimeError("Invalid vision camera setup: all camera streams are disabled.")
            channels_per_view = 3 if bool(getattr(self, "vision_mode", "")) == "dextrah_stereo" else (4 if use_depth else 3)
            self.vision_channels = int(channels_per_view)
            self.vision_obs_dim = int(
                max(1, int(self.vision_frame_stack))
                * int(self.vision_resize_h)
                * int(self.vision_resize_w)
                * int(channels_per_view)
                * int(view_count)
            )
            try:
                if hasattr(self, "num_rerenders_on_reset"):
                    self.num_rerenders_on_reset = max(
                        1, int(getattr(self, "num_rerenders_on_reset", 0))
                    )
            except Exception:
                pass
            try:
                if hasattr(self, "rerender_on_reset"):
                    self.rerender_on_reset = True
            except Exception:
                pass
        else:
            self.strict_vision_only_policy = False
            self.num_hand_obs = 0
            # State-mode teacher obs. Base is 44 with arm velocities, 32 without. The
            # hand's contribution is 1 dim per agent in legacy grip mode and 16 per
            # agent in dexterous mode, so the totals become 74 / 62 respectively.
            # (This branch runs AFTER _sync_full_hand_action_dims(), so it must stay
            # hand-aware or it silently clobbers the dexterous dims.)
            base_obs = 44 if bool(self.handover_teacher_arm_velocity_obs) else 32
            hand_extra = 0
            if bool(getattr(self, "handover_full_hand_action", False)):
                hand_extra = 2 * (_HAND_OBS_DIM_PER_AGENT - _GRIP_OBS_DIM_PER_AGENT)
            rel_extra = (
                _RELATIVE_OBS_EXTRA_DIM
                if bool(getattr(self, "handover_relative_obs", False)) else 0
            )
            self.observation_space = base_obs + hand_extra + rel_extra
            self.share_observation_space = int(self.observation_space)
            self.vision_obs_dim = 0
        self._sync_vision_camera_cfg()

    def __post_init__(self):
        super().__post_init__()

        self.rigid_objects_cfg["cube"] = self.cube_cfg
        self.rigid_objects_cfg["supporter_1"] = self.supporter_1_cfg
        self.rigid_objects_cfg["supporter_2"] = self.supporter_2_cfg
        if bool(getattr(self, "show_goal_marker", True)):
            self.markers_cfg["goal_marker"] = self.goal_marker_cfg
        else:
            self.markers_cfg.pop("goal_marker", None)
        self.apply_runtime_mode()
        vision_stride, camera_update_period = self._sync_vision_camera_cfg()

        print(
            f"[{self.task_name}EnvCfg]\nBASE_DIR={self.BASE_DIR}\nepisode_length={self.episode_length}\n"
            f"action_space={self.action_space}\nobservation_space={self.observation_space}\n"
            f"share_observation_space={self.share_observation_space}\n"
            f"use_vision={bool(self.use_vision)} vision_mode={self.vision_mode}\n"
            f"vision_raw={bool(self.vision_student_raw_mode)} vision_obs_dim={int(self.vision_obs_dim)}"
            f"\nvision_camera={int(self.vision_camera_width)}x{int(self.vision_camera_height)} "
            f"vision_update_stride={vision_stride} vision_update_hz={(1.0 / camera_update_period):.2f}"
        )


class TaskEnv(DirectRLEnv):         
    cfg: TaskEnvCfg                 

    def __init__(self, cfg: TaskEnvCfg, render_mode: str | None = None, **kwargs): 
        self.vision_camera: Optional[TiledCamera] = None
        self.vision_camera_head: Optional[TiledCamera] = None
        self.vision_camera_front: Optional[TiledCamera] = None
        self.vision_camera_front_right: Optional[TiledCamera] = None
        self._vision_buffers_ready = False
        self._vision_shape_checked = False
        self._vision_guard_checked = False
        self._faithful_mask_export_logged = False
        self._banned_obs_keys = {
            "obj_pos",
            "cube_pos",
            "cube_quat",
            "cubes_pos",
            "boxes_pos",
            "pad_pos",
            "privileged_obs",
        }

        # Add offsets for parallel environments.
        self.env_spacing = cfg.scene.env_spacing
        self.env_origins = torch.zeros((cfg.scene.num_envs, 3), device=cfg.sim.device)

        if cfg.scene.num_envs == 1:
            self.env_origins[0] = torch.tensor([0.0, 0.0, 0.0], device=cfg.sim.device)
        else:
            num_cols = int(np.floor(np.sqrt(cfg.scene.num_envs)))
            num_rows = int(np.ceil(cfg.scene.num_envs / num_cols))
            half_size_x = (num_rows - 1) * self.env_spacing / 2.0
            half_size_y = (num_cols - 1) * self.env_spacing / 2.0

            for i in range(cfg.scene.num_envs):
                row = i // num_cols
                col = i % num_cols
                x = half_size_x - row * self.env_spacing
                y = col * self.env_spacing - half_size_y
                self.env_origins[i] = torch.tensor([x, y, 0.0], device=cfg.sim.device)

        self.base_goal_pos = self.env_origins.new_tensor([0.627, 0.444, 1.2]).view(1, 3)

        self._articulations: dict[str, Articulation] = {}
        self._rigid_objects: dict[str, RigidObject] = {}
        self._markers: dict[str, VisualizationMarkers] = {}

        super().__init__(cfg, render_mode, **kwargs)

        self.right_actions = self.env_origins.new_zeros((self.num_envs, 22))
        self.left_actions = self.env_origins.new_zeros((self.num_envs, 22))
        self.right_demos = torch.zeros(
            [self.num_envs, cfg.action_space], device=self.device
        )
        self.left_demos = torch.zeros(
            [self.num_envs, cfg.action_space], device=self.device
        )

        self.initial_cube_pos = self.env_origins.new_zeros((self.num_envs, 3))
        # Initial cube orientation (WXYZ); updated on every reset and used for cube-aware demo wrist targets.
        self.initial_cube_quat = self.env_origins.new_zeros((self.num_envs, 4))
        self.initial_cube_quat[:, 0] = 1.0  # identity placeholder; overwritten on first reset
        self.right_handover_ref_pos = self.env_origins.new_zeros((self.num_envs, 3))
        self.right_handover_ref_quat = self.env_origins.new_zeros((self.num_envs, 4))
        self.right_handover_ref_quat[:, 0] = 1.0
        self.right_handover_ref_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.demo_handover_ref_pos = self.env_origins.new_zeros((self.num_envs, 3))
        self.demo_handover_ref_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.progress_buf = self.env_origins.new_zeros((self.num_envs, 1))
        self.subgoal = self.env_origins.new_zeros((self.num_envs, 2))
        self.hold_timer = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.last_grip_R = torch.ones(self.num_envs, device=self.device)
        self.last_grip_L = torch.ones(self.num_envs, device=self.device)
        # Potential-based shaping state: phi from the previous step, and whether
        # that phi belongs to the current episode (False right after a reset).
        self._shaping_phi_prev = self.env_origins.new_zeros((self.num_envs,))
        self._shaping_phi_valid = torch.zeros(self.num_envs, device=self.device,
                                              dtype=torch.bool)
        self._prev_obs_pos_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._prev_cube_local = self.env_origins.new_zeros((self.num_envs, 3))
        self._prev_rh_local = self.env_origins.new_zeros((self.num_envs, 3))
        self._prev_lh_local = self.env_origins.new_zeros((self.num_envs, 3))
        self._dr_cube_pos_bias = self.env_origins.new_zeros((self.num_envs, 3))
        self._dr_cube_rot_bias_deg = self.env_origins.new_zeros((self.num_envs, 3))
        self._dr_right_pos_bias = self.env_origins.new_zeros((self.num_envs, 3))
        self._dr_left_pos_bias = self.env_origins.new_zeros((self.num_envs, 3))
        self._dr_right_joint_pos_bias = self.env_origins.new_zeros((self.num_envs, 6))
        self._dr_left_joint_pos_bias = self.env_origins.new_zeros((self.num_envs, 6))
        self._dr_right_joint_vel_bias = self.env_origins.new_zeros((self.num_envs, 6))
        self._dr_left_joint_vel_bias = self.env_origins.new_zeros((self.num_envs, 6))

        self.in_success_region = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.target_held = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.hands_home = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.success_region_latch = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.success_region_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.home_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.success_latched = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._debug_markers_initialized = False
        self._debug_markers_enabled = bool(getattr(cfg, "enable_debug_markers", False))
        self._debug_vis_every = 1 if self._debug_markers_enabled else 10**9
        self.debug_success_region_marker = None
        self.debug_home_right_marker = None
        self.debug_home_left_marker = None
        self.debug_object_receive_target_marker = None
        self.debug_stage_ball_markers: list[VisualizationMarkers] = []
        self._debug_marker_enable_logged = False
        self._debug_marker_visualize_logged = False
        z_ball = float(getattr(self.cfg, "debug_stage_marker_z", 1.95))
        x_start = float(getattr(self.cfg, "debug_stage_marker_x_start", -0.35))
        x_gap = float(getattr(self.cfg, "debug_stage_marker_x_gap", 0.10))
        y_ball = float(getattr(self.cfg, "debug_stage_marker_y", -0.18))
        self._debug_stage_ball_legend = [
            "1 BLUE  : LEFT_GRASPED",
            "2 GREEN : HANDOVER_REACHED",
            "3 ORANGE: RIGHT_GRASPED_AFTER_Y_CLEARANCE",
            "4 YELLOW: TARGET_HELD_5_STEPS",
            "5 PINK  : HOME_HELD_10_STEPS_SUCCESS",
        ]
        self._debug_stage_ball_offsets = torch.tensor(
            [[x_start + x_gap * i, y_ball, z_ball] for i in range(5)],
            device=self.device,
            dtype=torch.float32,
        )
        if self._debug_markers_enabled:
            self._ensure_debug_markers()

        self.right_dof_lower_limits: Optional[torch.Tensor] = None
        self.right_dof_upper_limits: Optional[torch.Tensor] = None
        self.left_dof_lower_limits: Optional[torch.Tensor] = None
        self.left_dof_upper_limits: Optional[torch.Tensor] = None
        self._joint_obs_limits: Optional[dict[str, torch.Tensor]] = None
        self._joint_obs_limits_checked = False

        # save joint's index
        self.right_joint_name_to_idx = {
            name: i for i, name in enumerate(self.right_robot.joint_names)
        }
        self.left_joint_name_to_idx = {
            name: i for i, name in enumerate(self.left_robot.joint_names)
        }
        # Canonical joint_<k>_0 -> Isaac hand slot. Isaac interleaves the fingers, so
        # these permutations are the ONLY safe way to talk about "the index finger".
        self.right_hand_slot_of_canon = hand_slot_map(
            self.right_robot.joint_names, "ra"
        )
        self.left_hand_slot_of_canon = hand_slot_map(
            self.left_robot.joint_names, "la"
        )
        # Slots holding true flexion joints, for open/closed classification.
        self.right_flex_slots = [self.right_hand_slot_of_canon[k] for k in _CANON_FLEX]
        self.left_flex_slots = [self.left_hand_slot_of_canon[k] for k in _CANON_FLEX]

        # Thumb opposition axis (joint_12) gets a non-zero lower bound to mirror
        # the hardware floor; all other actuated finger joints keep the full
        # [0, 1.57] range and are driven lockstep by the single grip scalar.
        right_thumb_oppose_names = ["ra_joint_12_0"]
        right_grip_joint_names = [
            "ra_joint_1_0",
            "ra_joint_13_0",
            "ra_joint_5_0",
            "ra_joint_9_0",
            "ra_joint_3_0",
            "ra_joint_15_0",
            "ra_joint_7_0",
            "ra_joint_11_0",
            "ra_joint_2_0",
            "ra_joint_14_0",
            "ra_joint_6_0",
            "ra_joint_10_0",
        ]
        left_thumb_oppose_names = ["la_joint_12_0"]
        left_grip_joint_names = [
            "la_joint_1_0",
            "la_joint_13_0",
            "la_joint_5_0",
            "la_joint_9_0",
            "la_joint_3_0",
            "la_joint_15_0",
            "la_joint_7_0",
            "la_joint_11_0",
            "la_joint_2_0",
            "la_joint_14_0",
            "la_joint_6_0",
            "la_joint_10_0",
        ]
        self.right_thumb_oppose_ids = [
            self.right_joint_name_to_idx[n] for n in right_thumb_oppose_names
        ]
        self.right_grip_ids = [
            self.right_joint_name_to_idx[n] for n in right_grip_joint_names
        ]
        self.left_thumb_oppose_ids = [
            self.left_joint_name_to_idx[n] for n in left_thumb_oppose_names
        ]
        self.left_grip_ids = [
            self.left_joint_name_to_idx[n] for n in left_grip_joint_names
        ]
        # Keep value_ids covering every actuated hand joint for downstream code
        # (obs normalization etc.) that still expects the original contract.
        self.right_value_ids = self.right_grip_ids + self.right_thumb_oppose_ids
        self.right_none_ids = [
            self.right_joint_name_to_idx[n]
            for n in ["ra_joint_0_0", "ra_joint_4_0", "ra_joint_8_0"]
        ]
        self.left_value_ids = self.left_grip_ids + self.left_thumb_oppose_ids
        self.left_none_ids = [
            self.left_joint_name_to_idx[n]
            for n in ["la_joint_0_0", "la_joint_4_0", "la_joint_8_0"]
        ]

        cube_base_pos = torch.tensor(
            self.cfg.cube_cfg.init_state.pos, device=self.device
        )
        supp1_base_pos = torch.tensor(
            self.cfg.supporter_1_cfg.init_state.pos, device=self.device
        )
        supp2_base_pos = torch.tensor(
            self.cfg.supporter_2_cfg.init_state.pos, device=self.device
        )

        # Local offset vectors from the cube center to each supporter center.
        self.supporter_1_world_offset = supp1_base_pos - cube_base_pos
        self.supporter_2_world_offset = supp2_base_pos - cube_base_pos
        self.default_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self._obs_shape_checked = False
        self._vision_num_views = (
            int(self._vision_use_head_camera_stream())
            + int(self._vision_use_top_camera())
            + int(self._vision_use_front_camera_stream())
            + int(self._vision_use_front_right_camera_stream())
        )
        if bool(getattr(self.cfg, "use_vision", False)) and self._vision_num_views <= 0:
            raise RuntimeError("Invalid vision camera setup: all camera streams are disabled.")
        stack = max(1, int(getattr(self.cfg, "vision_frame_stack", 1)))
        self._vision_base_dim = (
            int(getattr(self.cfg, "vision_obs_dim", 0)) // stack
            if bool(getattr(self.cfg, "use_vision", False))
            else 0
        )
        if bool(getattr(self.cfg, "use_vision", False)):
            self.cfg.vision_obs_dim = int(self._vision_base_dim * stack)
            self._ensure_vision_buffers()
            try:
                all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
                self._configure_vision_camera_views(all_env_ids)
            except Exception as exc:
                print(f"[{self.cfg.task_name}][vision] initial camera pose setup skipped: {exc}")

    def _vision_use_top_camera(self) -> bool:
        return (not bool(getattr(self.cfg, "vision_front_only", False))) and (
            not bool(getattr(self.cfg, "vision_use_head_camera", False))
        )

    def _vision_use_head_camera_stream(self) -> bool:
        return bool(getattr(self.cfg, "vision_use_head_camera", False))

    def _vision_use_front_camera_stream(self) -> bool:
        if bool(getattr(self.cfg, "vision_use_head_camera", False)) and (
            not bool(getattr(self.cfg, "vision_use_front_camera", False))
        ):
            return False
        return bool(getattr(self.cfg, "vision_front_only", False)) or bool(
            getattr(self.cfg, "vision_use_front_camera", False)
        )

    def _vision_use_front_right_camera_stream(self) -> bool:
        return bool(getattr(self.cfg, "vision_use_front_right_camera", False))

    def _is_dextrah_faithful_mode(self) -> bool:
        return bool(getattr(self.cfg, "dextrah_faithful_mode", False)) or bool(
            getattr(self.cfg, "dextrah_faithful_profile", False)
        )

    def _is_faithful_front_stereo_mode(self) -> bool:
        return self._is_dextrah_faithful_mode() and bool(
            getattr(self.cfg, "vision_use_front_right_camera", False)
        )

    def _use_raw_student_vision_mode(self) -> bool:
        return bool(getattr(self.cfg, "vision_student_raw_mode", False)) and bool(
            getattr(self.cfg, "distill_dual_obs", False)
        )

    def _ensure_vision_buffers(self) -> None:
        if self._vision_buffers_ready:
            return
        if self._vision_base_dim <= 0:
            raise RuntimeError("vision buffers requested but vision base dim is not initialized")
        stack = max(1, int(getattr(self.cfg, "vision_frame_stack", 1)))
        max_delay = max(0, int(getattr(self.cfg, "vision_latency_steps_max", 0)))
        hist_len = max(1, stack + max_delay)
        self._vision_hist = torch.zeros(
            (self.num_envs, hist_len, self._vision_base_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self._vision_hist_valid = torch.zeros(
            (self.num_envs, hist_len), device=self.device, dtype=torch.bool
        )
        self._vision_delay_steps = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        self._vision_buffers_ready = True

    def _euler_xyz_to_rotation_matrix_batch(self, angles_rad: torch.Tensor) -> torch.Tensor:
        cx = torch.cos(angles_rad[:, 0])
        sx = torch.sin(angles_rad[:, 0])
        cy = torch.cos(angles_rad[:, 1])
        sy = torch.sin(angles_rad[:, 1])
        cz = torch.cos(angles_rad[:, 2])
        sz = torch.sin(angles_rad[:, 2])
        zeros = torch.zeros_like(cx)
        ones = torch.ones_like(cx)
        rx = torch.stack(
            (ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx),
            dim=-1,
        ).reshape(-1, 3, 3)
        ry = torch.stack(
            (cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy),
            dim=-1,
        ).reshape(-1, 3, 3)
        rz = torch.stack(
            (cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones),
            dim=-1,
        ).reshape(-1, 3, 3)
        return torch.bmm(rz, torch.bmm(ry, rx))

    def _sample_dextrah_style_view_locals(
        self,
        eye_local: torch.Tensor,
        target_local: torch.Tensor,
        batch_size: int,
        rot_deg_range: float,
        pos_range: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eyes_local = eye_local.unsqueeze(0).repeat(batch_size, 1)
        base_forward = target_local - eye_local
        base_dist = torch.linalg.norm(base_forward).clamp(min=1e-6)
        base_forward = base_forward / base_dist
        if pos_range > 0.0:
            pos_delta = (
                torch.rand((batch_size, 3), device=self.device) * 2.0 - 1.0
            ) * float(pos_range)
            eyes_local = eyes_local + pos_delta
        if rot_deg_range > 0.0:
            angles_deg = (
                torch.rand((batch_size, 3), device=self.device) * 2.0 - 1.0
            ) * float(rot_deg_range)
            angles_rad = angles_deg * (math.pi / 180.0)
            rot_mats = self._euler_xyz_to_rotation_matrix_batch(angles_rad)
            fwd = torch.bmm(
                rot_mats,
                base_forward.view(1, 3, 1).repeat(batch_size, 1, 1),
            ).squeeze(-1)
            fwd = F.normalize(fwd, dim=-1)
        else:
            fwd = base_forward.view(1, 3).repeat(batch_size, 1)
        targets_local = eyes_local + fwd * base_dist
        return eyes_local, targets_local

    def _sample_yaw_quats_wxyz(self, batch_size: int, yaw_deg_range: float | tuple[float, float]) -> torch.Tensor:
        yaw = self._sample_yaw_angles_rad(batch_size, yaw_deg_range)
        half = yaw * 0.5
        return torch.stack(
            (
                torch.cos(half),
                torch.zeros_like(half),
                torch.zeros_like(half),
                torch.sin(half),
            ),
            dim=-1,
        )

    def _sample_yaw_angles_rad(self, batch_size: int, yaw_deg_range: float | tuple[float, float]) -> torch.Tensor:
        if isinstance(yaw_deg_range, (tuple, list)):
            values = tuple(float(v) for v in yaw_deg_range)
            neg_range = abs(values[0]) if len(values) > 0 else 0.0
            pos_range = abs(values[1]) if len(values) > 1 else neg_range
            low = -math.radians(neg_range)
            high = math.radians(pos_range)
        else:
            half_range = abs(float(yaw_deg_range))
            low = -math.radians(half_range)
            high = math.radians(half_range)
        if bool(getattr(self.cfg, "handover_object_pose_bin_balanced", False)):
            bins = int(getattr(self.cfg, "handover_object_yaw_bins", 1))
            return self._sample_bin_balanced_uniform(batch_size, low, high, bins)
        return torch.empty((int(batch_size),), device=self.device, dtype=torch.float32).uniform_(low, high)

    def _sample_bin_balanced_uniform(self, batch_size: int, low: float, high: float, bins: int) -> torch.Tensor:
        batch_size = int(batch_size)
        bins = max(1, int(bins))
        if batch_size <= 0:
            return torch.empty((0,), device=self.device, dtype=torch.float32)
        if high <= low or bins <= 1:
            return torch.empty((batch_size,), device=self.device, dtype=torch.float32).uniform_(float(low), float(high))
        bin_ids = torch.arange(batch_size, device=self.device, dtype=torch.long) % bins
        bin_ids = bin_ids[torch.randperm(batch_size, device=self.device)]
        edges = torch.linspace(float(low), float(high), bins + 1, device=self.device, dtype=torch.float32)
        lo = edges[bin_ids]
        hi = edges[bin_ids + 1]
        return lo + torch.rand((batch_size,), device=self.device, dtype=torch.float32) * (hi - lo)

    def _configure_vision_camera_views(self, env_ids: torch.Tensor) -> None:
        if not bool(getattr(self.cfg, "use_vision", False)):
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        def _apply_view(
            camera: Optional[TiledCamera],
            eye_key: str,
            target_key: str,
            eye_jitter_key: str,
            target_jitter_key: str,
            tag: str,
            pose_rot_deg_key: Optional[str] = None,
            pose_pos_range_key: Optional[str] = None,
        ):
            if camera is None:
                return
            eye_local = torch.tensor(
                getattr(self.cfg, eye_key), device=self.device, dtype=torch.float32
            )
            target_local = torch.tensor(
                getattr(self.cfg, target_key), device=self.device, dtype=torch.float32
            )
            eye_jitter = torch.tensor(
                getattr(self.cfg, eye_jitter_key), device=self.device, dtype=torch.float32
            )
            target_jitter = torch.tensor(
                getattr(self.cfg, target_jitter_key), device=self.device, dtype=torch.float32
            )
            batch_size = int(env_ids.numel())
            eyes_local = eye_local.unsqueeze(0).repeat(batch_size, 1)
            targets_local = target_local.unsqueeze(0).repeat(batch_size, 1)
            quat_key = eye_key.replace("eye_local", "quat_wxyz")
            if (
                bool(getattr(self.cfg, "dextrah_use_hardware_camera_pose", False))
                and hasattr(self.cfg, quat_key)
            ):
                eyes = self.env_origins[env_ids] + eyes_local
                quats = torch.tensor(
                    getattr(self.cfg, quat_key),
                    device=self.device,
                    dtype=torch.float32,
                ).view(1, 4).repeat(batch_size, 1)
                quats = F.normalize(quats, dim=-1)
                if bool(getattr(self.cfg, "dextrah_camera_pose_rand_enable", False)):
                    rot_deg = (
                        float(getattr(self.cfg, pose_rot_deg_key, 0.0))
                        if pose_rot_deg_key is not None
                        else 0.0
                    )
                    pos_rng = (
                        float(getattr(self.cfg, pose_pos_range_key, 0.0))
                        if pose_pos_range_key is not None
                        else 0.0
                    )
                    if pos_rng > 0.0:
                        eyes = eyes + (torch.rand_like(eyes) * 2.0 - 1.0) * pos_rng
                    if rot_deg > 0.0:
                        quats = _jitter_quat_wxyz(quats, rot_deg)
                pose_mode = str(
                    getattr(self.cfg, "dextrah_hardware_camera_pose_mode", "view_target")
                ).strip().lower()
                try:
                    if pose_mode == "direct":
                        direct_convention = str(
                            getattr(self.cfg, "dextrah_hardware_camera_direct_convention", "world")
                        ).strip().lower()
                        camera.set_world_poses(
                            positions=eyes,
                            orientations=quats,
                            env_ids=env_ids,
                            convention=direct_convention,
                        )
                    else:
                        w, x, y, z = quats.unbind(dim=-1)
                        forward = torch.stack(
                            (
                                1.0 - 2.0 * (y * y + z * z),
                                2.0 * (x * y + z * w),
                                2.0 * (x * z - y * w),
                            ),
                            dim=-1,
                        )
                        target_dist = float(
                            getattr(self.cfg, "dextrah_hardware_camera_target_distance", 1.0)
                        )
                        targets = eyes + F.normalize(forward, dim=-1) * target_dist
                        camera.set_world_poses_from_view(eyes, targets, env_ids=env_ids)
                except Exception as exc:
                    if pose_mode == "direct":
                        raise RuntimeError(
                            f"[{self.cfg.task_name}][vision] {tag} direct hardware camera pose setup failed"
                        ) from exc
                    print(f"[{self.cfg.task_name}][vision] {tag} hardware camera pose setup skipped: {exc}")
                return
            if bool(getattr(self.cfg, "dextrah_camera_pose_rand_enable", False)):
                rot_deg = (
                    float(getattr(self.cfg, pose_rot_deg_key, 0.0))
                    if pose_rot_deg_key is not None
                    else 0.0
                )
                pos_rng = (
                    float(getattr(self.cfg, pose_pos_range_key, 0.0))
                    if pose_pos_range_key is not None
                    else 0.0
                )
                if rot_deg > 0.0 or pos_rng > 0.0:
                    eyes_local, targets_local = self._sample_dextrah_style_view_locals(
                        eye_local=eye_local,
                        target_local=target_local,
                        batch_size=batch_size,
                        rot_deg_range=rot_deg,
                        pos_range=pos_rng,
                    )
            eyes = self.env_origins[env_ids] + eyes_local
            targets = self.env_origins[env_ids] + targets_local
            if torch.any(eye_jitter > 0):
                eyes = eyes + torch.randn_like(eyes) * eye_jitter.unsqueeze(0)
            if torch.any(target_jitter > 0):
                targets = targets + torch.randn_like(targets) * target_jitter.unsqueeze(0)
            try:
                camera.set_world_poses_from_view(eyes, targets, env_ids=env_ids)
            except Exception as exc:
                print(f"[{self.cfg.task_name}][vision] {tag} camera pose setup skipped: {exc}")

        _apply_view(
            self.vision_camera,
            "vision_camera_eye_local",
            "vision_camera_target_local",
            "vision_camera_eye_jitter",
            "vision_camera_target_jitter",
            "top",
        )
        if self._vision_use_head_camera_stream():
            _apply_view(
                self.vision_camera_head,
                "vision_head_camera_eye_local",
                "vision_head_camera_target_local",
                "vision_head_camera_eye_jitter",
                "vision_head_camera_target_jitter",
                "head",
            )
        if self._vision_use_front_camera_stream():
            _apply_view(
                self.vision_camera_front,
                "vision_front_camera_eye_local",
                "vision_front_camera_target_local",
                "vision_front_camera_eye_jitter",
                "vision_front_camera_target_jitter",
                "front",
                "dextrah_front_camera_rand_rot_deg",
                "dextrah_front_camera_rand_pos_range",
            )
        if self._vision_use_front_right_camera_stream():
            _apply_view(
                self.vision_camera_front_right,
                "vision_front_right_camera_eye_local",
                "vision_front_right_camera_target_local",
                "vision_front_right_camera_eye_jitter",
                "vision_front_right_camera_target_jitter",
                "front_right",
                "dextrah_front_right_camera_rand_rot_deg",
                "dextrah_front_right_camera_rand_pos_range",
            )

    def _spawn_real_scene_reference(self) -> None:
        if not bool(getattr(self.cfg, "handover_use_real_scene", False)):
            return
        try:
            import omni.usd
            from pxr import Gf, Sdf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            usd_path = Path(str(getattr(self.cfg, "handover_real_scene_usd_path", ""))).expanduser()
            if not usd_path.is_absolute():
                usd_path = Path(self.cfg.BASE_DIR) / usd_path
            usd_path = usd_path.resolve()
            if not usd_path.is_file():
                raise FileNotFoundError(f"USD not found: {usd_path}")

            translate = tuple(float(v) for v in getattr(self.cfg, "handover_real_scene_translate", (0.83, 0.19, -0.13)))
            rotate_zyx = tuple(float(v) for v in getattr(self.cfg, "handover_real_scene_rotate_zyx", (4.7, -11.8, 13.5)))
            scale = tuple(float(v) for v in getattr(self.cfg, "handover_real_scene_scale", (1.03, 1.03, 1.03)))
            prim_path_template = str(getattr(self.cfg, "handover_real_scene_prim_path", "/World/envs/env_0/RealScene"))
            referenced = 0
            for env_id in range(int(self.num_envs)):
                prim_path = prim_path_template.replace("/env_0/", f"/env_{env_id}/")
                xform = UsdGeom.Xform.Define(stage, prim_path)
                prim = xform.GetPrim()
                prim.GetReferences().ClearReferences()
                prim.GetReferences().AddReference(str(usd_path))

                xformable = UsdGeom.Xformable(prim)
                xformable.ClearXformOpOrder()
                xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))
                xformable.AddRotateZYXOp().Set(Gf.Vec3d(*rotate_zyx))
                xformable.AddScaleOp().Set(Gf.Vec3d(*scale))
                prim.CreateAttribute("trl:visualBackground", Sdf.ValueTypeNames.Bool).Set(True)
                referenced += 1
            print(
                f"[handover_tshape] real scene referenced for {referenced} envs: {usd_path} -> {prim_path_template}",
                flush=True,
            )
        except Exception as exc:
            raise RuntimeError(f"failed to spawn handover real scene background: {exc}") from exc

    def _resolve_asset_glob(self, directory: str, pattern: str) -> list[str]:
        root = Path(str(directory or "")).expanduser()
        if not root.is_absolute():
            root = Path(self.cfg.BASE_DIR) / root
        if not root.is_dir():
            return []
        return sorted(str(path.resolve()).replace("\\", "/") for path in root.glob(str(pattern or "*")) if path.is_file())

    def _random_indices(self, count: int, high: int) -> list[int]:
        if count <= 0 or high <= 0:
            return []
        return torch.randint(0, int(high), (int(count),), device=self.device).detach().cpu().tolist()

    def _apply_table_asset_randomization(self) -> None:
        if not bool(getattr(self.cfg, "handover_randomize_tables", False)):
            return
        table_assets = self._resolve_asset_glob(
            str(getattr(self.cfg, "handover_table_asset_dir", "assets/motion_desk")),
            str(getattr(self.cfg, "handover_table_asset_glob", "*.usd")),
        )
        if not table_assets:
            print("[handover_tshape][table_rand] no table USD assets found; keeping default table.", flush=True)
            return
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            choices = self._random_indices(int(self.num_envs), len(table_assets))
            applied = 0
            for env_id, choice in enumerate(choices):
                prim_path = f"/World/envs/env_{env_id}/table"
                prim = stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    continue
                refs = prim.GetReferences()
                refs.ClearReferences()
                refs.AddReference(table_assets[int(choice) % len(table_assets)])
                applied += 1
            print(
                f"[handover_tshape][table_rand] applied {applied} env table references "
                f"from {len(table_assets)} assets in {getattr(self.cfg, 'handover_table_asset_dir', '')}",
                flush=True,
            )
        except Exception as exc:
            raise RuntimeError(f"failed to randomize handover table assets: {exc}") from exc

    def _shader_paths_under_prim(self, stage, prim_path: str) -> list[str]:
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim or not root_prim.IsValid():
            return []
        from pxr import Usd

        return [str(prim.GetPath()) for prim in Usd.PrimRange(root_prim) if prim.GetTypeName() == "Shader"]

    def _set_shader_texture_inputs(
        self,
        stage,
        shader_paths: list[str],
        texture_path: str,
        tint_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        if not shader_paths or not texture_path:
            return
        from pxr import Gf, Sdf, UsdShade

        asset_path = Sdf.AssetPath(texture_path)
        tint = Gf.Vec3f(float(tint_color[0]), float(tint_color[1]), float(tint_color[2]))
        for shader_path in shader_paths:
            shader_prim = stage.GetPrimAtPath(shader_path)
            if not shader_prim or not shader_prim.IsValid():
                continue
            shader = UsdShade.Shader(shader_prim)
            shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(asset_path)
            shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(tint)
            shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(tint)
            shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1.0, 1.0))
            shader.CreateInput("texture_translate", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.0, 0.0))
            shader.CreateInput("texture_rotate", Sdf.ValueTypeNames.Float).Set(0.0)

    def _bind_wall_texture_material(
        self,
        stage,
        prim_path: str,
        material_path: str,
        texture_path: str,
        tint_color: tuple[float, float, float],
    ) -> bool:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return False
        from pxr import Gf, Sdf, UsdShade

        if not stage.GetPrimAtPath(material_path).IsValid():
            material_cfg = sim_utils.MdlFileCfg(
                mdl_path="OmniPBR.mdl",
                project_uvw=True,
                texture_scale=tuple(float(v) for v in getattr(self.cfg, "handover_wall_texture_scale", (3.0, 3.0))),
            )
            material_cfg.func(material_path, material_cfg)
        shader_prim = stage.GetPrimAtPath(f"{material_path}/Shader")
        if not shader_prim or not shader_prim.IsValid():
            return False
        shader = UsdShade.Shader(shader_prim)
        tint = Gf.Vec3f(float(tint_color[0]), float(tint_color[1]), float(tint_color[2]))
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
        shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(tint)
        shader.CreateInput("albedo_brightness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(*tuple(float(v) for v in getattr(self.cfg, "handover_wall_texture_scale", (3.0, 3.0))))
        )
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.55)
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
        sim_utils.bind_visual_material(prim_path, material_path)
        return True

    def _apply_wall_texture_randomization(self) -> None:
        if not bool(getattr(self.cfg, "handover_randomize_wall_textures", False)):
            return
        textures = self._resolve_asset_glob(
            str(getattr(self.cfg, "handover_wall_texture_dir", "assets/texture")),
            str(getattr(self.cfg, "handover_wall_texture_glob", "*.png")),
        )
        if not textures:
            print("[handover_tshape][wall_tex] no wall textures found; keeping default wall materials.", flush=True)
            return
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            wall_names = ("left_wall", "back_wall")
            wall_tints = {
                "left_wall": (0.114, 0.188, 0.365),
                "back_wall": (0.03, 0.03, 0.03),
            }
            choices = self._random_indices(int(self.num_envs) * len(wall_names), len(textures))
            applied = 0
            choice_idx = 0
            for env_id in range(int(self.num_envs)):
                for wall_name in wall_names:
                    prim_path = f"/World/envs/env_{env_id}/{wall_name}"
                    texture_path = textures[int(choices[choice_idx]) % len(textures)]
                    choice_idx += 1
                    material_path = f"/World/Looks/handover_{wall_name}_tex_env_{env_id}"
                    if self._bind_wall_texture_material(
                        stage,
                        prim_path,
                        material_path,
                        texture_path,
                        wall_tints.get(wall_name, (1.0, 1.0, 1.0)),
                    ):
                        applied += 1
            print(
                f"[handover_tshape][wall_tex] applied {applied} wall texture bindings "
                f"from {len(textures)} textures in {getattr(self.cfg, 'handover_wall_texture_dir', '')}",
                flush=True,
            )
        except Exception as exc:
            raise RuntimeError(f"failed to randomize handover wall textures: {exc}") from exc

    def _apply_tshape_black_material(self) -> None:
        if not bool(getattr(self.cfg, "handover_black_tshape", False)):
            return
        try:
            import omni.usd
            from pxr import Usd

            stage = omni.usd.get_context().get_stage()
            material_path = "/World/Looks/handover_tshape_black"
            color = tuple(float(v) for v in getattr(self.cfg, "handover_tshape_color", (0.0, 0.0, 0.0)))
            if not stage.GetPrimAtPath(material_path).IsValid():
                material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
                material_cfg.func(material_path, material_cfg)

            applied = 0
            for env_id in range(int(self.num_envs)):
                root_path = f"/World/envs/env_{env_id}/tshape"
                root_prim = stage.GetPrimAtPath(root_path)
                if not root_prim or not root_prim.IsValid():
                    continue
                sim_utils.bind_visual_material(root_path, material_path)
                applied += 1
                for prim in Usd.PrimRange(root_prim):
                    type_name = prim.GetTypeName()
                    if type_name in ("Mesh", "Capsule", "Cube", "Cylinder", "Sphere"):
                        sim_utils.bind_visual_material(str(prim.GetPath()), material_path)
            print(f"[handover_tshape][tshape_mat] bound black material to {applied} tshape roots", flush=True)
        except Exception as exc:
            raise RuntimeError(f"failed to apply handover tshape material: {exc}") from exc

    def _apply_light_randomization(self) -> None:
        if not bool(getattr(self.cfg, "handover_randomize_light", False)):
            return
        try:
            import omni.usd
            from pxr import Gf

            stage = omni.usd.get_context().get_stage()
            light_prim = stage.GetPrimAtPath("/World/Light")
            if not light_prim or not light_prim.IsValid():
                return
            intensity_range = tuple(float(v) for v in getattr(self.cfg, "handover_light_intensity_range", (900.0, 2200.0)))
            color_range = tuple(float(v) for v in getattr(self.cfg, "handover_light_color_range", (0.75, 1.0)))
            intensity_min, intensity_max = min(intensity_range), max(intensity_range)
            color_min, color_max = min(color_range), max(color_range)
            intensity = float(
                torch.empty((), device=self.device).uniform_(intensity_min, intensity_max).detach().cpu().item()
            )
            color = (
                torch.empty((3,), device=self.device).uniform_(color_min, color_max).detach().cpu().tolist()
            )
            light_prim.GetAttribute("inputs:intensity").Set(intensity)
            light_prim.GetAttribute("inputs:color").Set(Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))
        except Exception as exc:
            raise RuntimeError(f"failed to randomize handover light: {exc}") from exc

    def _apply_table_visibility(self) -> None:
        visible = bool(getattr(self.cfg, "handover_table_visible", True))
        if visible:
            return
        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            hidden = 0
            for env_id in range(int(self.num_envs)):
                prim_path = f"/World/envs/env_{env_id}/table"
                prim = stage.GetPrimAtPath(prim_path)
                if prim and prim.IsValid():
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden += 1
            source_prim = stage.GetPrimAtPath("/World/envs/env_0/table")
            if source_prim and source_prim.IsValid():
                UsdGeom.Imageable(source_prim).MakeInvisible()
            print(f"[handover_tshape] table visibility disabled for {hidden} env tables", flush=True)
        except Exception as exc:
            raise RuntimeError(f"failed to apply handover table visibility: {exc}") from exc

    def _make_vision_camera(self, cam_cfg):
        """Instantiate a vision camera from a TiledCameraCfg.

        When the realscene (NuRec / 3DGS) background is active, use a STANDARD
        ``Camera`` instead of ``TiledCamera``: the tiled render path does not render
        NuRec volumes correctly, while the full-RTX standard Camera path does.
        All cfg options (resolution, focal length, clipping, offset, data_types) are
        preserved -- only the sensor class changes. Without realscene, keep
        TiledCamera for RL throughput.
        """
        if not bool(getattr(self.cfg, "handover_use_real_scene", False)):
            return TiledCamera(cam_cfg)
        std_cfg = CameraCfg(
            prim_path=cam_cfg.prim_path,
            update_period=cam_cfg.update_period,
            height=cam_cfg.height,
            width=cam_cfg.width,
            data_types=list(cam_cfg.data_types),
            spawn=cam_cfg.spawn,
            offset=CameraCfg.OffsetCfg(
                pos=cam_cfg.offset.pos,
                rot=cam_cfg.offset.rot,
                convention=cam_cfg.offset.convention,
            ),
        )
        print(f"[handover_tshape] realscene active -> standard Camera for {cam_cfg.prim_path}", flush=True)
        return Camera(std_cfg)

    def _setup_scene(self):
        # 1) Instantiate articulations, rigid objects, and markers registered by the base/task config.
        for name, cfg in self.cfg.articulations_cfg.items():
            art = Articulation(cfg)
            self._articulations[name] = art
            self.scene.articulations[name] = art
            setattr(self, name, art)  # self.right_robot, self.left_robot, etc.

        for name, cfg in self.cfg.rigid_objects_cfg.items():
            robj = RigidObject(cfg)
            self._rigid_objects[name] = robj
            self.scene.rigid_objects[name] = robj
            setattr(self, name, robj)  # self.table, self.cube, self.supporter_1 ...

        for name, cfg in self.cfg.markers_cfg.items():
            marker = VisualizationMarkers(cfg)
            self._markers[name] = marker
            setattr(self, name, marker)  # self.goal_marker, etc.

        # 2) ground plane + light
        sim_utils.spawn_ground_plane("/World/ground", cfg=sim_utils.GroundPlaneCfg())

        light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/Light", light_cfg)
        self._apply_light_randomization()

        # 3) Clone environments.
        self.scene.clone_environments(copy_from_source=True)
        self._spawn_real_scene_reference()
        self._apply_table_asset_randomization()
        self._apply_tshape_black_material()
        self._apply_wall_texture_randomization()
        self._apply_table_visibility()

        if bool(getattr(self.cfg, "use_vision", False)):
            if self._vision_use_top_camera():
                self.vision_camera = self._make_vision_camera(self.cfg.table_camera_cfg)
                self.scene.sensors["vision_camera"] = self.vision_camera
                self.scene.sensors["vision_camera_top"] = self.vision_camera
            if self._vision_use_head_camera_stream():
                self.vision_camera_head = self._make_vision_camera(self.cfg.head_camera_cfg)
                self.scene.sensors["vision_camera_head"] = self.vision_camera_head
                if self.vision_camera is None:
                    self.scene.sensors["vision_camera"] = self.vision_camera_head
            if self._vision_use_front_camera_stream():
                self.vision_camera_front = self._make_vision_camera(self.cfg.front_camera_cfg)
                self.scene.sensors["vision_camera_front"] = self.vision_camera_front
                if self.vision_camera is None:
                    self.scene.sensors["vision_camera"] = self.vision_camera_front
            if self._vision_use_front_right_camera_stream():
                self.vision_camera_front_right = self._make_vision_camera(self.cfg.front_right_camera_cfg)
                self.scene.sensors["vision_camera_front_right"] = self.vision_camera_front_right

        # 4) Compute and visualize goal positions.
        self.goal_positions = (
            self.base_goal_pos.repeat(self.num_envs, 1) + self.env_origins
        )
        if hasattr(self, "goal_marker"):
            self.goal_marker.visualize(translations=self.goal_positions)

    # Action-space-7 case with thumb-opposition lower-limit floor.
    def _smooth_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """EMA + per-step slew clamp in normalized action space.

        Mirrors the filter hw/view_policy_hw.py applies before commanding the real robot,
        so training, evaluation and deployment can share one action path. Both terms
        default to 0.0 (disabled), which reproduces every earlier number bit-for-bit.
        """
        w = float(getattr(self.cfg, "handover_action_ema", 0.0) or 0.0)
        md = float(getattr(self.cfg, "handover_action_max_delta", 0.0) or 0.0)
        if not getattr(self, "_action_filter_announced", False):
            self._action_filter_announced = True
            print(f"[handover][action-filter] ema={w} max_delta={md} "
                  f"({'ACTIVE' if (w > 0.0 or md > 0.0) else 'disabled'})", flush=True)
        if w <= 0.0 and md <= 0.0:
            return actions
        prev = getattr(self, "_action_filter_prev", None)
        if prev is None or prev.shape != actions.shape:
            self._action_filter_prev = actions.detach().clone()
            return actions
        # Envs that just reset must not inherit the previous episode's command.
        # Exempt exactly the first step of an episode (<=1 exempted two, which let the
        # filter engage a step later for reset envs than for the rest of the batch).
        fresh = (self.episode_length_buf < 1).view(-1, 1)
        out = actions
        if w > 0.0:
            out = (1.0 - w) * out + w * prev
        if md > 0.0:
            out = prev + (out - prev).clamp(min=-md, max=md)
        out = torch.where(fresh, actions, out)
        self._action_filter_prev = out.detach().clone()
        return out

    def _pre_physics_step_full_hand(self, actions: torch.Tensor) -> None:
        """Dexterous path: 22 dims per agent, flat [N, 44].

        Layout `[R_arm 0:6, R_hand 6:22, L_arm 22:28, L_hand 28:44]` is exactly
        `(N, 2, 22)` flattened, so it is MAT-compatible without reshaping. Ported from
        the 2026-06-17 `task_handover_tshape_22.py` reference.
        """
        if actions.ndim != 2 or int(actions.shape[-1]) != 44:
            raise ValueError(
                "handover_full_hand_action expects actions of shape [num_envs, 44] "
                f"(2 agents x 22), got {tuple(actions.shape)}"
            )

        actions = self._smooth_actions(actions)

        right_arm_raw = actions[:, 0:6]
        right_hand_raw = actions[:, 6:22]
        left_arm_raw = actions[:, 22:28]
        left_hand_raw = actions[:, 28:44]

        # Keep every scalar-grip consumer (reward's open/closed test, obs export,
        # logging) working unchanged by summarising the 16 finger commands as their
        # mean "how closed is this hand" value.
        self.last_grip_R.copy_(right_hand_raw.detach().mean(dim=-1))
        self.last_grip_L.copy_(left_hand_raw.detach().mean(dim=-1))

        right_arm_scaled = unscale_action_to_joint(
            right_arm_raw,
            self.right_dof_lower_limits[:6],
            self.right_dof_upper_limits[:6],
        )
        left_arm_scaled = unscale_action_to_joint(
            left_arm_raw,
            self.left_dof_lower_limits[:6],
            self.left_dof_upper_limits[:6],
        )
        right_hand_scaled = unscale_action_to_joint(
            right_hand_raw, self.hand_dof_lower_limits, self.hand_dof_upper_limits
        )
        left_hand_scaled = unscale_action_to_joint(
            left_hand_raw, self.hand_dof_lower_limits, self.hand_dof_upper_limits
        )

        self.right_actions[:, :6] = right_arm_scaled
        self.right_actions[:, 6:22] = right_hand_scaled
        self.left_actions[:, :6] = left_arm_scaled
        self.left_actions[:, 6:22] = left_hand_scaled

    def _grasp_plan(self):
        """Per-env grasp assignment, built once, matching the demo collector's.

        Same library, same seed, same `right_is_giver=False` as
        `generate_demo._get_dexterous_plan`, so env i is assigned the same pregrasp/grasp
        pair the demos recorded for env i. That correspondence is what lets a demo's grip
        channel reproduce its own finger targets through the synergy; it holds only when
        num_envs matches between collection and training.
        """
        plan = getattr(self, "_dex_grasp_plan", None)
        if plan is not None and plan.num_envs == self.num_envs:
            return plan
        from trl_utils.dexterous_demo import DexterousGraspPlan, default_cylinder_library

        plan = DexterousGraspPlan(
            num_envs=self.num_envs,
            library=default_cylinder_library(),
            device=self.device,
            seed=int(getattr(self, "_dex_demo_seed", 0)),
            right_is_giver=False,
        )
        self._dex_grasp_plan = plan
        print(f"[handover][synergy] {plan.describe()}", flush=True)
        return plan

    def _pre_physics_step_synergy(self, actions: torch.Tensor) -> None:
        """Synergy path: 7 dims per agent, flat [N, 14] = [R_arm6, R_grip, L_arm6, L_grip].

        Layout matches the legacy 14-D path exactly, so every scalar-grip consumer
        (`last_grip_*`, obs export, logging) keeps working; only the fan-out differs --
        the grip drives the library posture instead of one shared angle.
        """
        if actions.ndim != 2 or int(actions.shape[-1]) != 14:
            raise ValueError(
                "handover_synergy_action expects actions of shape [num_envs, 14] "
                f"(2 agents x 7), got {tuple(actions.shape)}"
            )
        from trl_utils.dexterous_demo import synergy_finger_targets

        right_arm_raw = actions[:, 0:6]
        right_grip_raw = actions[:, 6]
        left_arm_raw = actions[:, 7:13]
        left_grip_raw = actions[:, 13]

        self.last_grip_R.copy_(right_grip_raw.detach())
        self.last_grip_L.copy_(left_grip_raw.detach())

        self.right_actions[:, :6] = unscale_action_to_joint(
            right_arm_raw, self.right_dof_lower_limits[:6], self.right_dof_upper_limits[:6]
        )
        self.left_actions[:, :6] = unscale_action_to_joint(
            left_arm_raw, self.left_dof_lower_limits[:6], self.left_dof_upper_limits[:6]
        )

        plan = self._grasp_plan()
        r_map = torch.as_tensor(self.right_hand_slot_of_canon, device=self.device, dtype=torch.long)
        l_map = torch.as_tensor(self.left_hand_slot_of_canon, device=self.device, dtype=torch.long)
        self.right_actions[:, 6:22] = synergy_finger_targets(
            plan, right_grip_raw, "right", slot_of_canon=r_map
        )
        self.left_actions[:, 6:22] = synergy_finger_targets(
            plan, left_grip_raw, "left", slot_of_canon=l_map
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if bool(getattr(self.cfg, "handover_synergy_action", False)):
            self._pre_physics_step_synergy(actions)
            return
        if bool(getattr(self.cfg, "handover_full_hand_action", False)):
            self._pre_physics_step_full_hand(actions)
            return

        right_arm_raw = actions[:, :6]
        left_arm_raw = actions[:, 7:13]
        right_hand_raw = actions[:, 6]
        left_hand_raw = actions[:, 13]
        self.last_grip_R.copy_(right_hand_raw.detach())
        self.last_grip_L.copy_(left_hand_raw.detach())

        right_arm_scaled = unscale_action_to_joint(
            right_arm_raw,
            self.right_dof_lower_limits[:6],
            self.right_dof_upper_limits[:6],
        )
        left_arm_scaled = unscale_action_to_joint(
            left_arm_raw,
            self.left_dof_lower_limits[:6],
            self.left_dof_upper_limits[:6],
        )

        thumb_oppose_lo = float(getattr(self.cfg, "handover_thumb_oppose_lower_limit", 0.2))
        right_grip_val = unscale_action_to_hand(right_hand_raw, 0.0, 1.57).unsqueeze(-1)
        right_thumb_oppose_val = unscale_action_to_hand(
            right_hand_raw, thumb_oppose_lo, 1.57
        ).unsqueeze(-1)
        left_grip_val = unscale_action_to_hand(left_hand_raw, 0.0, 1.57).unsqueeze(-1)
        left_thumb_oppose_val = unscale_action_to_hand(
            left_hand_raw, thumb_oppose_lo, 1.57
        ).unsqueeze(-1)

        self.right_actions[:, :6] = right_arm_scaled
        self.left_actions[:, :6] = left_arm_scaled

        self.right_actions[:, self.right_grip_ids] = right_grip_val
        self.right_actions[:, self.right_thumb_oppose_ids] = right_thumb_oppose_val
        self.right_actions[:, self.right_none_ids] = 0.0
        self.left_actions[:, self.left_grip_ids] = left_grip_val
        self.left_actions[:, self.left_thumb_oppose_ids] = left_thumb_oppose_val
        self.left_actions[:, self.left_none_ids] = 0.0

    # # Action-space-22 case.
    # def _pre_physics_step(self, actions: torch.Tensor) -> None:
    #     right_arm_raw = actions[:, :6]
    #     right_hand_raw = actions[:, 6:22]
    #     left_arm_raw = actions[:, 22:28]
    #     left_hand_raw = actions[:, 28:]

    #     right_arm_scaled = unscale_action_to_joint(
    #         right_arm_raw,
    #         self.right_dof_lower_limits[:6],
    #         self.right_dof_upper_limits[:6],
    #     )
    #     right_hand_val = unscale_action_to_joint(
    #         right_hand_raw, self.hand_dof_lower_limits, self.hand_dof_upper_limits
    #     )
    #     left_arm_scaled = unscale_action_to_joint(
    #         left_arm_raw, self.left_dof_lower_limits[:6], self.left_dof_upper_limits[:6]
    #     )
    #     left_hand_val = unscale_action_to_joint(
    #         left_hand_raw, self.hand_dof_lower_limits, self.hand_dof_upper_limits
    #     )

    #     self.right_actions[:, :6] = right_arm_scaled
    #     self.right_actions[:, 6:22] = right_hand_val
    #     self.left_actions[:, :6] = left_arm_scaled
    #     self.left_actions[:, 6:22] = left_hand_val

    def _apply_action(self) -> None:
        right_actions = self.right_actions
        left_actions = self.left_actions

        right_arm_joint_ids = self.right_robot.actuators["ur3_arm"].joint_indices
        right_hand_joint_ids = self.right_robot.actuators["allegro_hand"].joint_indices

        self.right_robot.set_joint_position_target(
            right_actions[:, : len(right_arm_joint_ids)], joint_ids=right_arm_joint_ids
        )
        self.right_robot.set_joint_position_target(
            right_actions[:, len(right_arm_joint_ids) :], joint_ids=right_hand_joint_ids
        )

        # Left hand: same control path.
        left_arm_joint_ids = self.left_robot.actuators["ur3_arm"].joint_indices
        left_hand_joint_ids = self.left_robot.actuators["allegro_hand"].joint_indices

        self.left_robot.set_joint_position_target(
            left_actions[:, : len(left_arm_joint_ids)], joint_ids=left_arm_joint_ids
        )
        self.left_robot.set_joint_position_target(
            left_actions[:, len(left_arm_joint_ids) :], joint_ids=left_hand_joint_ids
        )

    def _assert_vision_camera_ready(self, camera: Optional[TiledCamera], camera_name: str) -> None:
        if camera is None:
            raise RuntimeError(
                f"{camera_name} is not initialized. Launch with --enable_cameras when use_vision=True."
            )
        is_ready = getattr(camera, "is_initialized", True)
        if callable(is_ready):
            try:
                is_ready = is_ready()
            except Exception:
                is_ready = True
        if isinstance(is_ready, bool) and (not is_ready):
            raise RuntimeError(f"{camera_name} exists but is not initialized yet.")

    def _extract_camera_rgb(
        self,
        camera: Optional[TiledCamera] = None,
        camera_name: str = "vision camera",
    ) -> torch.Tensor:
        if not bool(getattr(self.cfg, "use_vision", False)):
            raise RuntimeError("vision is disabled but _extract_camera_rgb() was called")
        if camera is None:
            camera = self.vision_camera
        self._assert_vision_camera_ready(camera, camera_name)
        outputs = getattr(getattr(camera, "data", None), "output", None)
        if outputs is None or "rgb" not in outputs:
            raise RuntimeError(f"{camera_name} output['rgb'] is unavailable")
        rgb = outputs["rgb"]
        if rgb is None:
            raise RuntimeError(f"{camera_name} returned rgb=None")
        if rgb.dim() != 4:
            raise RuntimeError(f"Expected rgb tensor rank 4, got {tuple(rgb.shape)}")
        if rgb.shape[-1] in (3, 4):
            rgb = rgb[..., :3].permute(0, 3, 1, 2).contiguous()
        elif rgb.shape[1] in (3, 4):
            rgb = rgb[:, :3].contiguous()
        else:
            raise RuntimeError(f"Unsupported rgb tensor layout: {tuple(rgb.shape)}")
        rgb = rgb.to(device=self.device, dtype=torch.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        return rgb.clamp(0.0, 1.0)

    def _extract_camera_depth_and_invalid_mask(
        self,
        *,
        camera: Optional[TiledCamera] = None,
        camera_name: str = "vision camera",
        allow_when_depth_disabled: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(getattr(self.cfg, "use_vision", False)):
            raise RuntimeError(
                "vision is disabled but _extract_camera_depth_and_invalid_mask() was called"
            )
        if (not allow_when_depth_disabled) and (
            not bool(getattr(self.cfg, "vision_use_depth", True))
        ):
            raise RuntimeError("vision_use_depth is False but depth extraction was requested")
        if camera is None:
            camera = self.vision_camera
        self._assert_vision_camera_ready(camera, camera_name)
        outputs = getattr(getattr(camera, "data", None), "output", None)
        if outputs is None:
            raise RuntimeError(f"{camera_name} outputs are unavailable")
        depth = outputs.get("distance_to_image_plane", outputs.get("depth", None))
        if depth is None:
            raise RuntimeError(f"{camera_name} depth output is unavailable")
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        elif depth.dim() == 4:
            if depth.shape[-1] == 1:
                depth = depth.permute(0, 3, 1, 2).contiguous()
            elif depth.shape[1] == 1:
                depth = depth.contiguous()
            else:
                raise RuntimeError(f"Unsupported depth tensor layout: {tuple(depth.shape)}")
        else:
            raise RuntimeError(f"Unsupported depth tensor rank: {tuple(depth.shape)}")
        depth = depth.to(device=self.device, dtype=torch.float32)
        near = float(getattr(self.cfg, "vision_depth_near", 0.05))
        far = float(getattr(self.cfg, "vision_depth_far", 2.50))
        if far <= near:
            far = near + 1.0
        invalid_far = depth > (far * 10.0)
        invalid_nonfinite = ~torch.isfinite(depth)
        invalid_mask = (invalid_far | invalid_nonfinite).to(dtype=torch.bool)
        depth = torch.nan_to_num(depth, nan=far, posinf=far, neginf=near)
        depth = depth.clamp(min=near, max=far)
        depth = (depth - near) / max(1e-6, far - near)
        return depth.clamp(0.0, 1.0), invalid_mask

    def _extract_camera_depth(
        self,
        camera: Optional[TiledCamera] = None,
        camera_name: str = "vision camera",
    ) -> torch.Tensor:
        if not bool(getattr(self.cfg, "use_vision", False)):
            raise RuntimeError("vision is disabled but _extract_camera_depth() was called")
        if not bool(getattr(self.cfg, "vision_use_depth", True)):
            raise RuntimeError("vision_use_depth is False but _extract_camera_depth() was called")
        depth, _ = self._extract_camera_depth_and_invalid_mask(
            camera=camera,
            camera_name=camera_name,
            allow_when_depth_disabled=False,
        )
        return depth

    def _apply_vision_augment(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not bool(getattr(self.cfg, "vision_augment", False)):
            return rgb, depth
        out_rgb = rgb
        out_depth = depth
        n = out_rgb.shape[0]
        if float(getattr(self.cfg, "vision_contrast_jitter", 0.0)) > 0.0:
            c = float(self.cfg.vision_contrast_jitter)
            contrast = 1.0 + (
                torch.rand((n, 1, 1, 1), device=self.device) * 2.0 - 1.0
            ) * c
            mean = out_rgb.mean(dim=(2, 3), keepdim=True)
            out_rgb = (out_rgb - mean) * contrast + mean
        if float(getattr(self.cfg, "vision_brightness_jitter", 0.0)) > 0.0:
            b = float(self.cfg.vision_brightness_jitter)
            bias = (torch.rand((n, 1, 1, 1), device=self.device) * 2.0 - 1.0) * b
            out_rgb = out_rgb + bias
        if float(getattr(self.cfg, "vision_rgb_noise_std", 0.0)) > 0.0:
            out_rgb = out_rgb + torch.randn_like(out_rgb) * float(
                self.cfg.vision_rgb_noise_std
            )
        out_rgb = out_rgb.clamp(0.0, 1.0)
        if out_depth is not None:
            scale_jitter = float(getattr(self.cfg, "vision_depth_scale_jitter", 0.0))
            if scale_jitter > 0.0:
                scale = 1.0 + (
                    torch.rand((n, 1, 1, 1), device=self.device) * 2.0 - 1.0
                ) * scale_jitter
                out_depth = out_depth * scale
            depth_noise_std = float(getattr(self.cfg, "vision_depth_noise_std", 0.0))
            if depth_noise_std > 0.0:
                out_depth = out_depth + torch.randn_like(out_depth) * depth_noise_std
            dropout_prob = float(getattr(self.cfg, "vision_depth_dropout_prob", 0.0))
            if dropout_prob > 0.0:
                keep = (torch.rand_like(out_depth) >= dropout_prob).to(out_depth.dtype)
                out_depth = out_depth * keep + (1.0 - keep)
            out_depth = out_depth.clamp(0.0, 1.0)
        return out_rgb.clamp(0.0, 1.0), (
            out_depth.clamp(0.0, 1.0) if out_depth is not None else None
        )

    def _get_raw_student_vision_rgbd(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not bool(getattr(self.cfg, "use_vision", False)):
            raise RuntimeError("vision is disabled but raw student vision was requested")
        use_depth = bool(getattr(self.cfg, "vision_use_depth", True))
        if self._vision_use_head_camera_stream() and (self.vision_camera_head is not None):
            rgb = self._extract_camera_rgb(
                camera=self.vision_camera_head,
                camera_name="head vision camera",
            )
            depth = (
                self._extract_camera_depth(
                    camera=self.vision_camera_head,
                    camera_name="head vision camera",
                )
                if use_depth
                else None
            )
        elif self._vision_use_front_camera_stream() and (self.vision_camera_front is not None):
            rgb = self._extract_camera_rgb(
                camera=self.vision_camera_front,
                camera_name="front vision camera",
            )
            depth = (
                self._extract_camera_depth(
                    camera=self.vision_camera_front,
                    camera_name="front vision camera",
                )
                if use_depth
                else None
            )
        elif self._vision_use_top_camera() and (self.vision_camera is not None):
            rgb = self._extract_camera_rgb(
                camera=self.vision_camera,
                camera_name="top vision camera",
            )
            depth = (
                self._extract_camera_depth(
                    camera=self.vision_camera,
                    camera_name="top vision camera",
                )
                if use_depth
                else None
            )
        else:
            raise RuntimeError("Raw student vision requested but no active camera stream is configured.")
        rgb, depth = self._apply_vision_augment(rgb, depth)
        size = (int(self.cfg.vision_resize_h), int(self.cfg.vision_resize_w))
        rgb = F.interpolate(rgb, size=size, mode="bilinear", align_corners=False)
        if depth is not None:
            depth = F.interpolate(depth, size=size, mode="bilinear", align_corners=False)
        return rgb, depth

    def _get_faithful_stereo_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        h = int(getattr(self.cfg, "vision_resize_h", 0))
        w = int(getattr(self.cfg, "vision_resize_w", 0))
        if h <= 0 or w <= 0:
            raise RuntimeError(
                f"[{self.cfg.task_name}][faithful] invalid resize for img_left/img_right: H={h}, W={w}"
            )
        img_left = None
        img_right = None
        if self._is_faithful_front_stereo_mode():
            if self.vision_camera_front is None or self.vision_camera_front_right is None:
                raise RuntimeError(
                    f"[{self.cfg.task_name}][faithful] front stereo requires both front-left and front-right cameras."
                )
            img_left = self._extract_camera_rgb(
                camera=self.vision_camera_front,
                camera_name="front-left vision camera",
            )
            img_right = self._extract_camera_rgb(
                camera=self.vision_camera_front_right,
                camera_name="front-right vision camera",
            )
        else:
            if self._vision_use_top_camera() and (self.vision_camera is not None):
                img_left = self._extract_camera_rgb(
                    camera=self.vision_camera,
                    camera_name="top vision camera",
                )
            if self._vision_use_front_camera_stream() and (self.vision_camera_front is not None):
                img_right = self._extract_camera_rgb(
                    camera=self.vision_camera_front,
                    camera_name="front vision camera",
                )
        if img_left is None or img_right is None:
            raise RuntimeError(
                f"[{self.cfg.task_name}][faithful] strict stereo contract requires both left and right image streams."
            )
        size = (h, w)
        if int(img_left.shape[-2]) != h or int(img_left.shape[-1]) != w:
            img_left = F.interpolate(img_left, size=size, mode="bilinear", align_corners=False)
        if int(img_right.shape[-2]) != h or int(img_right.shape[-1]) != w:
            img_right = F.interpolate(img_right, size=size, mode="bilinear", align_corners=False)
        if bool(getattr(self.cfg, "dextrah_right_img_flip", True)):
            img_right = torch.flip(img_right, dims=(2, 3))
        return img_left, img_right

    def _get_faithful_stereo_masks(
        self,
        img_left: torch.Tensor,
        img_right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._is_faithful_front_stereo_mode():
            _, left_mask = self._extract_camera_depth_and_invalid_mask(
                camera=self.vision_camera_front,
                camera_name="front-left vision camera",
                allow_when_depth_disabled=True,
            )
            _, right_mask = self._extract_camera_depth_and_invalid_mask(
                camera=self.vision_camera_front_right,
                camera_name="front-right vision camera",
                allow_when_depth_disabled=True,
            )
            target_hw = (int(img_left.shape[-2]), int(img_left.shape[-1]))
            if tuple(left_mask.shape[-2:]) != target_hw:
                left_mask = (
                    F.interpolate(left_mask.to(dtype=torch.float32), size=target_hw, mode="nearest")
                    > 0.5
                )
            if tuple(right_mask.shape[-2:]) != target_hw:
                right_mask = (
                    F.interpolate(right_mask.to(dtype=torch.float32), size=target_hw, mode="nearest")
                    > 0.5
                )
        else:
            if not self._faithful_mask_export_logged:
                print(
                    f"[{self.cfg.task_name}][faithful] depth-based stereo mask unavailable in this camera layout; exporting zero masks.",
                    flush=True,
                )
                self._faithful_mask_export_logged = True
            left_mask = torch.zeros(
                (int(img_left.shape[0]), 1, int(img_left.shape[-2]), int(img_left.shape[-1])),
                dtype=torch.bool,
                device=self.device,
            )
            right_mask = torch.zeros(
                (int(img_right.shape[0]), 1, int(img_right.shape[-2]), int(img_right.shape[-1])),
                dtype=torch.bool,
                device=self.device,
            )
        return left_mask, right_mask

    def _get_vision_obs(self) -> torch.Tensor:
        if not bool(getattr(self.cfg, "use_vision", False)):
            raise RuntimeError("vision is disabled but _get_vision_obs() was called")
        self._ensure_vision_buffers()
        use_depth = bool(getattr(self.cfg, "vision_use_depth", True))
        update_stride = max(1, int(getattr(self.cfg, "vision_update_stride", 1)))
        refresh_mask = ~self._vision_hist_valid[:, 0]
        if update_stride <= 1:
            refresh_mask = torch.ones_like(refresh_mask, dtype=torch.bool)
        else:
            refresh_mask = refresh_mask | (
                torch.remainder(self.episode_length_buf.long(), update_stride) == 0
            )
        refresh_any = bool(refresh_mask.any().item())
        if refresh_any:
            fused_views = []
            if self._vision_use_head_camera_stream() and (self.vision_camera_head is not None):
                rgb_head = self._extract_camera_rgb(
                    camera=self.vision_camera_head,
                    camera_name="head vision camera",
                )
                depth_head = (
                    self._extract_camera_depth(
                        camera=self.vision_camera_head,
                        camera_name="head vision camera",
                    )
                    if use_depth
                    else None
                )
                rgb_head, depth_head = self._apply_vision_augment(rgb_head, depth_head)
                rgb_head = F.interpolate(
                    rgb_head,
                    size=(int(self.cfg.vision_resize_h), int(self.cfg.vision_resize_w)),
                    mode="bilinear",
                    align_corners=False,
                )
                fused_head = rgb_head
                if depth_head is not None:
                    depth_head = F.interpolate(
                        depth_head,
                        size=(int(self.cfg.vision_resize_h), int(self.cfg.vision_resize_w)),
                        mode="bilinear",
                        align_corners=False,
                    )
                    fused_head = torch.cat((rgb_head, depth_head), dim=1)
                fused_views.append(fused_head)
            if self._vision_use_top_camera() and (self.vision_camera is not None):
                rgb_top = self._extract_camera_rgb(
                    camera=self.vision_camera,
                    camera_name="top vision camera",
                )
                depth_top = (
                    self._extract_camera_depth(
                        camera=self.vision_camera,
                        camera_name="top vision camera",
                    )
                    if use_depth
                    else None
                )
                rgb_top, depth_top = self._apply_vision_augment(rgb_top, depth_top)
                rgb_top = F.interpolate(
                    rgb_top,
                    size=(int(self.cfg.vision_resize_h), int(self.cfg.vision_resize_w)),
                    mode="bilinear",
                    align_corners=False,
                )
                fused_top = rgb_top
                if depth_top is not None:
                    depth_top = F.interpolate(
                        depth_top,
                        size=(int(self.cfg.vision_resize_h), int(self.cfg.vision_resize_w)),
                        mode="bilinear",
                        align_corners=False,
                    )
                    fused_top = torch.cat((rgb_top, depth_top), dim=1)
                fused_views.append(fused_top)
            if self._vision_use_front_camera_stream() and (self.vision_camera_front is not None):
                rgb_front = self._extract_camera_rgb(
                    camera=self.vision_camera_front,
                    camera_name="front vision camera",
                )
                depth_front = (
                    self._extract_camera_depth(
                        camera=self.vision_camera_front,
                        camera_name="front vision camera",
                    )
                    if use_depth
                    else None
                )
                rgb_front, depth_front = self._apply_vision_augment(rgb_front, depth_front)
                rgb_front = F.interpolate(
                    rgb_front,
                    size=(int(self.cfg.vision_resize_h), int(self.cfg.vision_resize_w)),
                    mode="bilinear",
                    align_corners=False,
                )
                fused_front = rgb_front
                if depth_front is not None:
                    depth_front = F.interpolate(
                        depth_front,
                        size=(int(self.cfg.vision_resize_h), int(self.cfg.vision_resize_w)),
                        mode="bilinear",
                        align_corners=False,
                    )
                    fused_front = torch.cat((rgb_front, depth_front), dim=1)
                fused_views.append(fused_front)
            if self._vision_use_front_right_camera_stream() and (
                self.vision_camera_front_right is not None
            ):
                rgb_front_right = self._extract_camera_rgb(
                    camera=self.vision_camera_front_right,
                    camera_name="front-right vision camera",
                )
                depth_front_right = (
                    self._extract_camera_depth(
                        camera=self.vision_camera_front_right,
                        camera_name="front-right vision camera",
                    )
                    if use_depth
                    else None
                )
                rgb_front_right, depth_front_right = self._apply_vision_augment(
                    rgb_front_right, depth_front_right
                )
                rgb_front_right = F.interpolate(
                    rgb_front_right,
                    size=(int(self.cfg.vision_resize_h), int(self.cfg.vision_resize_w)),
                    mode="bilinear",
                    align_corners=False,
                )
                fused_front_right = rgb_front_right
                if depth_front_right is not None:
                    depth_front_right = F.interpolate(
                        depth_front_right,
                        size=(int(self.cfg.vision_resize_h), int(self.cfg.vision_resize_w)),
                        mode="bilinear",
                        align_corners=False,
                    )
                    fused_front_right = torch.cat((rgb_front_right, depth_front_right), dim=1)
                fused_views.append(fused_front_right)
            if not fused_views:
                raise RuntimeError("No active vision camera stream is configured.")
            fused = torch.cat(fused_views, dim=1) if len(fused_views) > 1 else fused_views[0]
            base_feat_new = fused.reshape(self.num_envs, -1)
            if int(base_feat_new.shape[-1]) != int(self._vision_base_dim):
                raise RuntimeError(
                    f"vision base dim mismatch: got {int(base_feat_new.shape[-1])}, expected {int(self._vision_base_dim)}"
                )
            base_feat = (
                base_feat_new
                if bool(refresh_mask.all().item())
                else self._vision_hist[:, 0].clone()
            )
            if not bool(refresh_mask.all().item()):
                base_feat[refresh_mask] = base_feat_new[refresh_mask]
        else:
            base_feat = self._vision_hist[:, 0].clone()
        self._vision_hist = torch.roll(self._vision_hist, shifts=1, dims=1)
        self._vision_hist_valid = torch.roll(self._vision_hist_valid, shifts=1, dims=1)
        self._vision_hist[:, 0] = base_feat
        self._vision_hist_valid[:, 0] = True
        stack = max(1, int(getattr(self.cfg, "vision_frame_stack", 1)))
        hist_len = int(self._vision_hist.shape[1])
        env_index = torch.arange(self.num_envs, device=self.device)
        stacked = []
        for k in range(stack):
            idx = torch.clamp(self._vision_delay_steps + k, min=0, max=hist_len - 1)
            feat = self._vision_hist[env_index, idx]
            valid = self._vision_hist_valid[env_index, idx]
            feat = torch.where(valid.unsqueeze(-1), feat, self._vision_hist[:, 0])
            stacked.append(feat)
        vision = torch.cat(stacked, dim=-1)
        if int(vision.shape[-1]) != int(self.cfg.vision_obs_dim):
            raise RuntimeError(
                f"vision obs dim mismatch: got {int(vision.shape[-1])}, expected {int(self.cfg.vision_obs_dim)}"
            )
        return vision

    def get_camera_debug_images(self, resize_to_obs: bool = False) -> dict[str, torch.Tensor | None]:
        if not bool(getattr(self.cfg, "use_vision", False)):
            raise RuntimeError("Camera debug images requested while use_vision=False")
        use_depth = bool(getattr(self.cfg, "vision_use_depth", True))
        head_rgb = None
        head_depth = None
        if self._vision_use_head_camera_stream() and (self.vision_camera_head is not None):
            head_rgb = self._extract_camera_rgb(
                camera=self.vision_camera_head,
                camera_name="head vision camera",
            )
            head_depth = (
                self._extract_camera_depth(
                    camera=self.vision_camera_head,
                    camera_name="head vision camera",
                )
                if use_depth
                else None
            )
        rgb = None
        depth = None
        if self._vision_use_top_camera() and (self.vision_camera is not None):
            rgb = self._extract_camera_rgb(camera=self.vision_camera, camera_name="top vision camera")
            depth = (
                self._extract_camera_depth(camera=self.vision_camera, camera_name="top vision camera")
                if use_depth
                else None
            )
        front_rgb = None
        front_depth = None
        if self._vision_use_front_camera_stream() and (self.vision_camera_front is not None):
            front_rgb = self._extract_camera_rgb(
                camera=self.vision_camera_front,
                camera_name="front vision camera",
            )
            front_depth = (
                self._extract_camera_depth(
                    camera=self.vision_camera_front,
                    camera_name="front vision camera",
                )
                if use_depth
                else None
            )
        front_right_rgb = None
        front_right_depth = None
        if self._vision_use_front_right_camera_stream() and (self.vision_camera_front_right is not None):
            front_right_rgb = self._extract_camera_rgb(
                camera=self.vision_camera_front_right,
                camera_name="front-right vision camera",
            )
            front_right_depth = (
                self._extract_camera_depth(
                    camera=self.vision_camera_front_right,
                    camera_name="front-right vision camera",
                )
                if use_depth
                else None
            )
        if resize_to_obs:
            size = (int(self.cfg.vision_resize_h), int(self.cfg.vision_resize_w))
            if rgb is not None:
                rgb = F.interpolate(rgb, size=size, mode="bilinear", align_corners=False)
            if depth is not None:
                depth = F.interpolate(depth, size=size, mode="bilinear", align_corners=False)
            if head_rgb is not None:
                head_rgb = F.interpolate(head_rgb, size=size, mode="bilinear", align_corners=False)
            if head_depth is not None:
                head_depth = F.interpolate(head_depth, size=size, mode="bilinear", align_corners=False)
            if front_rgb is not None:
                front_rgb = F.interpolate(front_rgb, size=size, mode="bilinear", align_corners=False)
            if front_depth is not None:
                front_depth = F.interpolate(front_depth, size=size, mode="bilinear", align_corners=False)
            if front_right_rgb is not None:
                front_right_rgb = F.interpolate(front_right_rgb, size=size, mode="bilinear", align_corners=False)
            if front_right_depth is not None:
                front_right_depth = F.interpolate(front_right_depth, size=size, mode="bilinear", align_corners=False)
        return {
            "head_rgb": None if head_rgb is None else head_rgb.detach(),
            "head_depth": None if head_depth is None else head_depth.detach(),
            "rgb": None if rgb is None else rgb.detach(),
            "depth": None if depth is None else depth.detach(),
            "front_rgb": None if front_rgb is None else front_rgb.detach(),
            "front_depth": None if front_depth is None else front_depth.detach(),
            "front_right_rgb": None if front_right_rgb is None else front_right_rgb.detach(),
            "front_right_depth": None if front_right_depth is None else front_right_depth.detach(),
        }

    def _handover_obs_rand_enabled(self) -> bool:
        return bool(getattr(self.cfg, "handover_domain_rand", False)) and bool(
            getattr(self.cfg, "handover_rand_obs_noise", True)
        )

    def _sample_handover_obs_biases(self, env_ids: torch.Tensor) -> None:
        if not self._handover_obs_rand_enabled():
            self._dr_cube_pos_bias[env_ids] = 0.0
            self._dr_cube_rot_bias_deg[env_ids] = 0.0
            self._dr_right_pos_bias[env_ids] = 0.0
            self._dr_left_pos_bias[env_ids] = 0.0
            self._dr_right_joint_pos_bias[env_ids] = 0.0
            self._dr_left_joint_pos_bias[env_ids] = 0.0
            self._dr_right_joint_vel_bias[env_ids] = 0.0
            self._dr_left_joint_vel_bias[env_ids] = 0.0
            return

        def _sample(shape, radius: float) -> torch.Tensor:
            if float(radius) <= 0.0:
                return torch.zeros(shape, device=self.device)
            return (torch.rand(shape, device=self.device) * 2.0 - 1.0) * float(radius)

        n = int(env_ids.numel())
        self._dr_cube_pos_bias[env_ids] = _sample((n, 3), float(getattr(self.cfg, "handover_object_pos_bias_range", 0.005)))
        self._dr_cube_rot_bias_deg[env_ids] = _sample((n, 3), float(getattr(self.cfg, "handover_object_rot_bias_deg", 2.0)))
        self._dr_right_pos_bias[env_ids] = _sample((n, 3), float(getattr(self.cfg, "handover_robot_pos_bias_range", 0.003)))
        self._dr_left_pos_bias[env_ids] = _sample((n, 3), float(getattr(self.cfg, "handover_robot_pos_bias_range", 0.003)))
        self._dr_right_joint_pos_bias[env_ids] = _sample((n, 6), float(getattr(self.cfg, "handover_robot_pos_bias_range", 0.003)))
        self._dr_left_joint_pos_bias[env_ids] = _sample((n, 6), float(getattr(self.cfg, "handover_robot_pos_bias_range", 0.003)))
        self._dr_right_joint_vel_bias[env_ids] = _sample((n, 6), float(getattr(self.cfg, "handover_robot_vel_bias_range", 0.02)))
        self._dr_left_joint_vel_bias[env_ids] = _sample((n, 6), float(getattr(self.cfg, "handover_robot_vel_bias_range", 0.02)))

    def _add_handover_obs_noise(self, x: torch.Tensor, std: float) -> torch.Tensor:
        if self._handover_obs_rand_enabled() and float(std) > 0.0:
            return x + torch.randn_like(x) * float(std)
        return x

    def _apply_euler_deg_delta_wxyz(self, quats: torch.Tensor, euler_deg: torch.Tensor) -> torch.Tensor:
        angles = torch.deg2rad(euler_deg.to(device=quats.device, dtype=quats.dtype))
        half = angles * 0.5
        sr, sp, sy = torch.sin(half).unbind(dim=-1)
        cr, cp, cy = torch.cos(half).unbind(dim=-1)
        delta = torch.stack(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ),
            dim=-1,
        )
        return F.normalize(_quat_mul_wxyz(quats, delta), dim=-1)

    def _apply_measured_quat_noise(self, quat_wxyz: torch.Tensor) -> torch.Tensor:
        if not self._handover_obs_rand_enabled():
            return quat_wxyz
        out = quat_wxyz
        if float(getattr(self.cfg, "handover_object_rot_bias_deg", 0.0)) > 0.0:
            out = self._apply_euler_deg_delta_wxyz(out, self._dr_cube_rot_bias_deg)
        noise_deg = float(getattr(self.cfg, "handover_object_rot_noise_deg", 0.0))
        if noise_deg > 0.0:
            out = _jitter_quat_wxyz(out, noise_deg)
        return F.normalize(out, dim=-1)

    def _get_joint_obs_limits(self) -> Optional[dict[str, torch.Tensor]]:
        if self._joint_obs_limits_checked:
            return self._joint_obs_limits

        self._joint_obs_limits_checked = True
        configured_path = str(getattr(self.cfg, "joint_obs_limits_path", "") or "").strip()
        if not configured_path:
            return None
        path = Path(configured_path)
        if not path.is_absolute():
            path = Path(self.cfg.BASE_DIR) / path
        path = path.resolve()
        if not path.is_file():
            print(f"[handover_tshape][obs] joint normalization limits not found: {path}; using built-in limits")
            return None

        required = (
            "right_pos_low",
            "right_pos_high",
            "left_pos_low",
            "left_pos_high",
            "right_vel_low",
            "right_vel_high",
            "left_vel_low",
            "left_vel_high",
        )
        with np.load(path, allow_pickle=False) as payload:
            missing = [key for key in required if key not in payload]
            if missing:
                raise KeyError(f"{path}: missing joint observation limit keys: {missing}")
            limits = {
                key: torch.as_tensor(payload[key], device=self.device, dtype=torch.float32)
                for key in required
            }
        for key, value in limits.items():
            if tuple(value.shape) != (22,):
                raise ValueError(f"{path}: {key} must have shape (22,), got {tuple(value.shape)}")
        for side in ("right", "left"):
            if torch.any(limits[f"{side}_pos_high"] < limits[f"{side}_pos_low"]):
                raise ValueError(f"{path}: invalid {side} position limits")
            if torch.any(limits[f"{side}_vel_high"] <= limits[f"{side}_vel_low"]):
                raise ValueError(f"{path}: invalid {side} velocity limits")

        self._joint_obs_limits = limits
        print(f"[handover_tshape][obs] loaded qpos/qvel min-max limits: {path}")
        return limits

    @staticmethod
    def _minmax_normalize(value: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        midpoint = 0.5 * (low + high)
        half_range = 0.5 * (high - low).clamp(min=1e-6)
        return ((value - midpoint) / half_range).clamp(-1.0, 1.0)

    def _get_observations_state_mode(
        self,
        export_debug: bool = True,
        student_policy_mode: bool = False,
    ) -> dict:
        with torch.no_grad():
            env_offsets = self.env_origins

            right_ee_pos = self.right_robot.data.body_pos_w[:, 6, :3] - env_offsets
            left_ee_pos = self.left_robot.data.body_pos_w[:, 6, :3] - env_offsets
            cube_pos_true = self.cube.data.root_pos_w[:, :3] - env_offsets
            cube_quat_true = self.cube.data.root_quat_w[:, :4]
            cube_pos = cube_pos_true
            if self._handover_obs_rand_enabled():
                cube_pos = self._add_handover_obs_noise(
                    cube_pos + self._dr_cube_pos_bias,
                    float(getattr(self.cfg, "handover_object_pos_noise_std", 0.003)),
                )
                robot_pos_noise = float(getattr(self.cfg, "handover_robot_pos_noise_std", 0.003))
                right_ee_pos = self._add_handover_obs_noise(right_ee_pos + self._dr_right_pos_bias, robot_pos_noise)
                left_ee_pos = self._add_handover_obs_noise(left_ee_pos + self._dr_left_pos_bias, robot_pos_noise)
            self.progress_buf[:, 0] = self.episode_length_buf / float(
                self.cfg.episode_length - 1
            )
            phase_signal = self.progress_buf[:, 0] * 3 * 3.14159 * 2
            sin_embed = torch.sin(phase_signal).unsqueeze(-1)
            cos_embed = torch.cos(phase_signal).unsqueeze(-1)

            right_dof_pos = self.right_robot.data.joint_pos[:, :6]
            left_dof_pos = self.left_robot.data.joint_pos[:, :6]
            right_dof_vel = self.right_robot.data.joint_vel[:, :6]
            left_dof_vel = self.left_robot.data.joint_vel[:, :6]
            if self._handover_obs_rand_enabled():
                robot_pos_noise = float(getattr(self.cfg, "handover_robot_pos_noise_std", 0.003))
                robot_vel_noise = float(getattr(self.cfg, "handover_robot_vel_noise_std", 0.03))
                right_dof_pos = self._add_handover_obs_noise(
                    right_dof_pos + self._dr_right_joint_pos_bias,
                    robot_pos_noise,
                )
                left_dof_pos = self._add_handover_obs_noise(
                    left_dof_pos + self._dr_left_joint_pos_bias,
                    robot_pos_noise,
                )
                right_dof_vel = self._add_handover_obs_noise(
                    right_dof_vel + self._dr_right_joint_vel_bias,
                    robot_vel_noise,
                )
                left_dof_vel = self._add_handover_obs_noise(
                    left_dof_vel + self._dr_left_joint_vel_bias,
                    robot_vel_noise,
                )

            right_gripper_pos = self.right_robot.data.joint_pos[:, 7:8]
            left_gripper_pos = self.left_robot.data.joint_pos[:, 7:8]
            right_gripper_vel = self.right_robot.data.joint_vel[:, 7:8]
            left_gripper_vel = self.left_robot.data.joint_vel[:, 7:8]

            # Continuous 6D rotation representation: first two object-frame axes.
            cube_quat_obs = self._apply_measured_quat_noise(cube_quat_true)
            cube_rot_matrix = matrix_from_quat(cube_quat_obs)
            cube_rotation_6d = cube_rot_matrix[:, :, :2].transpose(1, 2).reshape(-1, 6)

            # Arm joints plus one measured grip joint per hand match the 7D action contract.
            right_arm_qpos_raw = right_dof_pos  # already noised arm 6 dim
            left_arm_qpos_raw = left_dof_pos
            if self._handover_obs_rand_enabled():
                hand_pos_noise = float(getattr(self.cfg, "handover_robot_pos_noise_std", 0.003))
                right_gripper_pos = self._add_handover_obs_noise(right_gripper_pos, hand_pos_noise)
                left_gripper_pos = self._add_handover_obs_noise(left_gripper_pos, hand_pos_noise)

            joint_obs_limits = self._get_joint_obs_limits()
            if joint_obs_limits is not None:
                r_pos_lo = joint_obs_limits["right_pos_low"]
                r_pos_hi = joint_obs_limits["right_pos_high"]
                l_pos_lo = joint_obs_limits["left_pos_low"]
                l_pos_hi = joint_obs_limits["left_pos_high"]
            else:
                r_pos_lo = torch.cat([self.right_dof_lower_limits[:6], self.hand_dof_lower_limits])
                r_pos_hi = torch.cat([self.right_dof_upper_limits[:6], self.hand_dof_upper_limits])
                l_pos_lo = torch.cat([self.left_dof_lower_limits[:6], self.hand_dof_lower_limits])
                l_pos_hi = torch.cat([self.left_dof_upper_limits[:6], self.hand_dof_upper_limits])

            right_arm_qpos_norm = self._minmax_normalize(right_arm_qpos_raw, r_pos_lo[:6], r_pos_hi[:6])
            left_arm_qpos_norm = self._minmax_normalize(left_arm_qpos_raw, l_pos_lo[:6], l_pos_hi[:6])
            right_gripper_pos_norm = self._minmax_normalize(right_gripper_pos, r_pos_lo[7:8], r_pos_hi[7:8])
            left_gripper_pos_norm = self._minmax_normalize(left_gripper_pos, l_pos_lo[7:8], l_pos_hi[7:8])

            # Match ColorCube state observations: one shared velocity scale for all arm joints.
            right_arm_qvel_raw = right_dof_vel  # already noised arm 6 dim
            left_arm_qvel_raw = left_dof_vel
            if self._handover_obs_rand_enabled():
                hand_vel_noise = float(getattr(self.cfg, "handover_robot_vel_noise_std", 0.03))
                right_gripper_vel = self._add_handover_obs_noise(right_gripper_vel, hand_vel_noise)
                left_gripper_vel = self._add_handover_obs_noise(left_gripper_vel, hand_vel_noise)

            if bool(getattr(self.cfg, "handover_arm_velocity_clip_to_obs_limits", False)):
                if joint_obs_limits is None:
                    raise RuntimeError(
                        "handover_arm_velocity_clip_to_obs_limits requires joint_obs_limits_path"
                    )
                right_arm_qvel_raw = torch.clamp(
                    right_arm_qvel_raw,
                    min=joint_obs_limits["right_vel_low"][:6],
                    max=joint_obs_limits["right_vel_high"][:6],
                )
                left_arm_qvel_raw = torch.clamp(
                    left_arm_qvel_raw,
                    min=joint_obs_limits["left_vel_low"][:6],
                    max=joint_obs_limits["left_vel_high"][:6],
                )

            vel_scale = float(getattr(self.cfg, "joint_vel_scale", 10.0))
            right_arm_qvel_norm = (right_arm_qvel_raw / vel_scale).clamp(-1.0, 1.0)
            left_arm_qvel_norm = (left_arm_qvel_raw / vel_scale).clamp(-1.0, 1.0)
            right_gripper_vel_norm = (right_gripper_vel / vel_scale).clamp(-1.0, 1.0)
            left_gripper_vel_norm = (left_gripper_vel / vel_scale).clamp(-1.0, 1.0)

            # Dexterous mode observes all 16 Allegro joints per hand instead of a single
            # grip scalar, so the policy can sense the hand it is commanding. Hand joints
            # live at indices 6:22 (arm occupies 0:6).
            if bool(getattr(self.cfg, "handover_full_hand_action", False)):
                right_hand_qpos = self.right_robot.data.joint_pos[:, 6:22]
                left_hand_qpos = self.left_robot.data.joint_pos[:, 6:22]
                if self._handover_obs_rand_enabled():
                    hand_pos_noise = float(
                        getattr(self.cfg, "handover_robot_pos_noise_std", 0.003)
                    )
                    right_hand_qpos = self._add_handover_obs_noise(right_hand_qpos, hand_pos_noise)
                    left_hand_qpos = self._add_handover_obs_noise(left_hand_qpos, hand_pos_noise)
                right_hand_obs = self._minmax_normalize(
                    right_hand_qpos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
                )                            # 16
                left_hand_obs = self._minmax_normalize(
                    left_hand_qpos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
                )                            # 16
            else:
                right_hand_obs = right_gripper_pos_norm   # 1
                left_hand_obs = left_gripper_pos_norm     # 1

            full_obs_parts = [
                right_ee_pos,                # 3
                left_ee_pos,                 # 3
                cube_pos,                    # 3
                self.progress_buf,           # 1
                sin_embed,                   # 1
                cos_embed,                   # 1
                cube_rotation_6d,            # 6
                right_arm_qpos_norm,         # 6
                left_arm_qpos_norm,          # 6
                right_hand_obs,              # 1 legacy / 16 dexterous
                left_hand_obs,               # 1 legacy / 16 dexterous
            ]
            if bool(getattr(self.cfg, "handover_teacher_arm_velocity_obs", True)):
                full_obs_parts.extend([right_arm_qvel_norm, left_arm_qvel_norm])

            # --- Relative / goal-conditioned features (handover_relative_obs) --------
            # The base observation gives only ABSOLUTE positions, and the task's
            # waypoints (handover point, home poses) appear nowhere in it -- they exist
            # only as threshold constants inside _get_rewards. The policy therefore has
            # to discover unobservable targets from binary stage bonuses, which is an
            # information deficit no amount of training or architecture can fix.
            #
            # These features expose, per end-effector, the vector TO each waypoint.
            # Standard practice in goal-conditioned manipulation RL (cf. the Fetch/Hand
            # gym envs, HER, and `delta_cube_target` already used in tasks/task_pickplace.py).
            # Constants are the same ones _get_rewards already uses, so the observation
            # and the reward agree on where the targets are.
            if bool(getattr(self.cfg, "handover_relative_obs", False)):
                o_handover = self.env_origins.new_tensor([0.0, 0.40, 1.23]).view(1, 3)
                r_handover = self.env_origins.new_tensor([0.05, 0.39, 1.33]).view(1, 3)
                r_home = self.env_origins.new_tensor([0.3520, 0.4038, 1.3374]).view(1, 3)
                l_home = self.env_origins.new_tensor([-0.3195, 0.4150, 1.3263]).view(1, 3)
                goal_local = self.goal_positions[:, :3] - env_offsets

                right_rel = torch.cat(
                    [
                        cube_pos - right_ee_pos,        # -> object
                        left_ee_pos - right_ee_pos,     # -> other hand
                        r_handover - right_ee_pos,      # -> handover point
                        goal_local - right_ee_pos,      # -> goal
                        r_home - right_ee_pos,          # -> home
                    ],
                    dim=-1,
                )                                        # 15
                left_rel = torch.cat(
                    [
                        cube_pos - left_ee_pos,
                        right_ee_pos - left_ee_pos,
                        o_handover - left_ee_pos,
                        goal_local - left_ee_pos,
                        l_home - left_ee_pos,
                    ],
                    dim=-1,
                )                                        # 15

                # End-effector orientation: the base obs has none, so the policy cannot
                # tell how the wrist is turned -- which matters for aligning a grasp.
                right_ee_quat = self.right_robot.data.body_quat_w[:, 6, :4]   # 4
                left_ee_quat = self.left_robot.data.body_quat_w[:, 6, :4]     # 4

                # First-order motion cues. Scaled by 20 (a ~5 cm/step motion saturates)
                # and clamped, so they stay in the same range as the rest of the vector.
                prev_valid = self._prev_obs_pos_valid.unsqueeze(-1)
                d_cube = torch.where(
                    prev_valid, cube_pos - self._prev_cube_local, torch.zeros_like(cube_pos))
                d_rh = torch.where(
                    prev_valid, right_ee_pos - self._prev_rh_local, torch.zeros_like(right_ee_pos))
                d_lh = torch.where(
                    prev_valid, left_ee_pos - self._prev_lh_local, torch.zeros_like(left_ee_pos))
                d_cube = (d_cube * 20.0).clamp(-1.0, 1.0)   # 3
                d_rh = (d_rh * 20.0).clamp(-1.0, 1.0)       # 3
                d_lh = (d_lh * 20.0).clamp(-1.0, 1.0)       # 3

                full_obs_parts.extend([
                    right_rel, left_rel,                    # 30
                    right_ee_quat, left_ee_quat,            # 8
                    d_cube, d_rh, d_lh,                     # 9
                ])                                          # +47 total

                # Advance the motion-delta history. Done here (not in _reset_idx) so the
                # deltas are always relative to the previous OBSERVED frame.
                self._prev_cube_local.copy_(cube_pos)
                self._prev_rh_local.copy_(right_ee_pos)
                self._prev_lh_local.copy_(left_ee_pos)
                self._prev_obs_pos_valid.fill_(True)

            full_obs = torch.cat(full_obs_parts, dim=-1)

            qpos = torch.cat(
                [
                    left_dof_pos,
                    left_gripper_pos_norm,
                    right_dof_pos,
                    right_gripper_pos_norm,
                ],
                dim=-1,
            )

            qvel = torch.cat(
                [
                    left_dof_vel,
                    left_gripper_vel_norm,
                    right_dof_vel,
                    right_gripper_vel_norm,
                ],
                dim=-1,
            )

            critic_obs = full_obs
            if student_policy_mode and bool(getattr(self.cfg, "use_vision", False)):
                right_q_obs = right_dof_pos
                left_q_obs = left_dof_pos
                right_qd_obs = right_arm_qvel_raw
                left_qd_obs = left_arm_qvel_raw

                if getattr(self.cfg, "normalize_obs", False) and (self.right_dof_lower_limits is not None):
                    r_low = self.right_dof_lower_limits[:6]
                    r_high = self.right_dof_upper_limits[:6]
                    l_low = self.left_dof_lower_limits[:6]
                    l_high = self.left_dof_upper_limits[:6]
                    r_mid = 0.5 * (r_low + r_high)
                    r_half = 0.5 * (r_high - r_low).clamp(min=1e-6)
                    l_mid = 0.5 * (l_low + l_high)
                    l_half = 0.5 * (l_high - l_low).clamp(min=1e-6)
                    right_q_obs = ((right_dof_pos - r_mid) / r_half).clamp(-1.0, 1.0)
                    left_q_obs = ((left_dof_pos - l_mid) / l_half).clamp(-1.0, 1.0)

                    vel_scale = float(getattr(self.cfg, "joint_vel_scale", 10.0))
                    right_qd_obs = (right_qd_obs / vel_scale).clamp(-1.0, 1.0)
                    left_qd_obs = (left_qd_obs / vel_scale).clamp(-1.0, 1.0)

                right_hand_joint_ids = self.right_robot.actuators["allegro_hand"].joint_indices
                left_hand_joint_ids = self.left_robot.actuators["allegro_hand"].joint_indices
                right_fq = self.right_robot.data.joint_pos[:, right_hand_joint_ids]
                left_fq = self.left_robot.data.joint_pos[:, left_hand_joint_ids]
                right_fq_obs = right_fq
                left_fq_obs = left_fq

                hand_low = getattr(self, "hand_dof_lower_limits", None)
                hand_high = getattr(self, "hand_dof_upper_limits", None)
                if getattr(self.cfg, "normalize_obs", False) and torch.is_tensor(hand_low) and torch.is_tensor(hand_high):
                    h_mid = 0.5 * (hand_low.view(1, -1) + hand_high.view(1, -1))
                    h_half = 0.5 * (hand_high.view(1, -1) - hand_low.view(1, -1)).clamp(min=1e-6)
                    right_fq_obs = ((right_fq - h_mid) / h_half).clamp(-1.0, 1.0)
                    left_fq_obs = ((left_fq - h_mid) / h_half).clamp(-1.0, 1.0)

                if not bool(getattr(self.cfg, "use_finger_obs", True)):
                    right_fq_obs = torch.zeros_like(right_fq_obs)
                    left_fq_obs = torch.zeros_like(left_fq_obs)

                right_grip_open = ((self.last_grip_R + 1.0) * 0.5).clamp(0.0, 1.0).unsqueeze(-1)
                left_grip_open = ((self.last_grip_L + 1.0) * 0.5).clamp(0.0, 1.0).unsqueeze(-1)
                if bool(getattr(self.cfg, "handover_hw_small_obs", False)):
                    right_hand_actor = torch.cat(
                        [right_q_obs, right_fq_obs, right_grip_open, right_qd_obs, right_ee_pos],
                        dim=-1,
                    )
                    left_hand_actor = torch.cat(
                        [left_q_obs, left_fq_obs, left_grip_open, left_qd_obs, left_ee_pos],
                        dim=-1,
                    )
                else:
                    student_obs_dim = int(getattr(self.cfg, "handover_student_obs_dim", 26))
                    if student_obs_dim == 14:
                        right_hand_actor = torch.cat([right_q_obs, right_grip_open], dim=-1)
                        left_hand_actor = torch.cat([left_q_obs, left_grip_open], dim=-1)
                    elif student_obs_dim == 26:
                        right_hand_actor = torch.cat(
                            [right_q_obs, right_qd_obs, right_grip_open], dim=-1
                        )
                        left_hand_actor = torch.cat(
                            [left_q_obs, left_qd_obs, left_grip_open], dim=-1
                        )
                    elif student_obs_dim == 46:
                        right_hand_actor = torch.cat(
                            [right_q_obs, right_fq_obs, right_grip_open], dim=-1
                        )
                        left_hand_actor = torch.cat(
                            [left_q_obs, left_fq_obs, left_grip_open], dim=-1
                        )
                    else:
                        raise RuntimeError(
                            f"unsupported handover_student_obs_dim={student_obs_dim}"
                        )
                policy_obs = torch.cat([right_hand_actor, left_hand_actor], dim=-1)

                cube_quat = self._apply_measured_quat_noise(self.cube.data.root_quat_w[:, :4])
                prev_valid = self._prev_obs_pos_valid.unsqueeze(-1)
                d_cube = torch.where(prev_valid, cube_pos - self._prev_cube_local, torch.zeros_like(cube_pos))
                d_rh = torch.where(prev_valid, right_ee_pos - self._prev_rh_local, torch.zeros_like(right_ee_pos))
                d_lh = torch.where(prev_valid, left_ee_pos - self._prev_lh_local, torch.zeros_like(left_ee_pos))
                d_cube = (d_cube * 20.0).clamp(-1.0, 1.0)
                d_rh = (d_rh * 20.0).clamp(-1.0, 1.0)
                d_lh = (d_lh * 20.0).clamp(-1.0, 1.0)

                goal_pos = getattr(self, "goal_positions", None)
                if torch.is_tensor(goal_pos) and int(goal_pos.shape[0]) == int(self.num_envs):
                    goal_local = goal_pos[:, :3] - env_offsets
                else:
                    goal_local = self.base_goal_pos.expand(self.num_envs, -1)

                supp1_local = self.supporter_1.data.root_pos_w[:, :3] - env_offsets
                supp2_local = self.supporter_2.data.root_pos_w[:, :3] - env_offsets
                init_cube = getattr(self, "initial_cube_pos", None)
                if torch.is_tensor(init_cube) and int(init_cube.shape[0]) == int(self.num_envs):
                    init_cube_local = init_cube[:, :3] - env_offsets
                else:
                    init_cube_local = cube_pos

                hold_ratio = (
                    self.hold_timer / max(float(getattr(self.cfg, "episode_length", 1)), 1.0)
                ).clamp(0.0, 1.0).unsqueeze(-1)
                success_flag = self.success_buf.float().unsqueeze(-1)

                shared = torch.cat(
                    [
                        right_ee_pos,
                        left_ee_pos,
                        cube_pos,
                        cube_quat,
                        self.progress_buf,
                        sin_embed,
                        cos_embed,
                        right_qd_obs,
                        left_qd_obs,
                        right_gripper_vel_norm,
                        left_gripper_vel_norm,
                        d_rh,
                        d_lh,
                        d_cube,
                        cube_pos - right_ee_pos,
                        cube_pos - left_ee_pos,
                        left_ee_pos - right_ee_pos,
                        goal_local,
                        goal_local - cube_pos,
                        goal_local - right_ee_pos,
                        goal_local - left_ee_pos,
                        supp1_local,
                        supp2_local,
                        cube_pos - supp1_local,
                        cube_pos - supp2_local,
                        self.subgoal.float(),
                        hold_ratio,
                        success_flag,
                        init_cube_local,
                        cube_pos - init_cube_local,
                    ],
                    dim=-1,
                )
                target_shared_dim = int(self.cfg.share_observation_space) - int(self.cfg.observation_space)
                if int(shared.shape[-1]) < target_shared_dim:
                    pad = torch.zeros(
                        (self.num_envs, target_shared_dim - int(shared.shape[-1])),
                        device=self.device,
                        dtype=shared.dtype,
                    )
                    shared = torch.cat([shared, pad], dim=-1)
                elif int(shared.shape[-1]) > target_shared_dim:
                    shared = shared[:, :target_shared_dim]
                critic_obs = torch.cat([right_hand_actor, left_hand_actor, shared], dim=-1)

                if not getattr(self, "_vision_student_obs_contract_checked", False):
                    if int(policy_obs.shape[-1]) != int(self.cfg.observation_space):
                        raise RuntimeError(
                            f"[{self.cfg.task_name}] vision policy obs dim mismatch: got {int(policy_obs.shape[-1])}, "
                            f"cfg.observation_space={int(self.cfg.observation_space)}"
                        )
                    if int(critic_obs.shape[-1]) != int(self.cfg.share_observation_space):
                        raise RuntimeError(
                            f"[{self.cfg.task_name}] vision critic obs dim mismatch: got {int(critic_obs.shape[-1])}, "
                            f"cfg.share_observation_space={int(self.cfg.share_observation_space)}"
                        )
                    print(
                        f"[{self.cfg.task_name}] ColorCube-like vision student obs policy={tuple(policy_obs.shape)} "
                        f"critic={tuple(critic_obs.shape)} num_hand_obs={int(getattr(self.cfg, 'num_hand_obs', 0))}",
                        flush=True,
                    )
                    self._vision_student_obs_contract_checked = True

                self._prev_cube_local.copy_(cube_pos)
                self._prev_rh_local.copy_(right_ee_pos)
                self._prev_lh_local.copy_(left_ee_pos)
                self._prev_obs_pos_valid.fill_(True)
            else:
                student_obs = torch.cat(
                    [
                        right_ee_pos,
                        left_ee_pos,
                        self.progress_buf,
                        sin_embed,
                        cos_embed,
                    ],
                    dim=-1,
                )
                policy_obs = student_obs if student_policy_mode else full_obs

            ee_pos = self.left_robot.data.body_pos_w[:, 6, :3] - env_offsets
            joint_angle = self.left_robot.data.joint_pos[:, :6]

            observations = {
                "policy": policy_obs,
                "critic": critic_obs,
                "proprio_observations": policy_obs,
                "privileged_observations": critic_obs,
                "action": qpos,
                "qpos": qpos,
                "qvel": qvel,
                "ee_pos": ee_pos,
                "joint_angle": joint_angle,
                "aux_info": {"object_pos": cube_pos_true, "object_quat": cube_quat_true},
            }
            if export_debug:
                observations["obj_pos"] = cube_pos_true

        return observations

    def _get_trlp4_student_proprio(self) -> torch.Tensor:
        env_offsets = self.env_origins
        right_ee_pos = self.right_robot.data.body_pos_w[:, 6, :3] - env_offsets
        left_ee_pos = self.left_robot.data.body_pos_w[:, 6, :3] - env_offsets
        return torch.cat(
            [
                self.right_robot.data.joint_pos,
                self.right_robot.data.joint_vel,
                self.left_robot.data.joint_pos,
                self.left_robot.data.joint_vel,
                right_ee_pos,
                left_ee_pos,
            ],
            dim=-1,
        )

    def _get_observations_vision_mode(self) -> dict:
        need_teacher_dual = bool(getattr(self.cfg, "distill_dual_obs", False))
        teacher_state = (
            self._get_observations_state_mode(
                export_debug=False,
                student_policy_mode=False,
            )
            if need_teacher_dual
            else None
        )
        if bool(getattr(self.cfg, "dextrah_trlp4_distill_mode", False)):
            student_proprio = self._get_trlp4_student_proprio()
            cube_pos = self.cube.data.root_pos_w[:, :3] - self.env_origins
            cube_quat = self.cube.data.root_quat_w[:, :4]
            out = {
                "policy": student_proprio,
                "critic": student_proprio,
                "aux_info": {"object_pos": cube_pos, "object_quat": cube_quat},
            }
        else:
            out = self._get_observations_state_mode(
                export_debug=False,
                student_policy_mode=True,
            )

        if self._use_raw_student_vision_mode():
            faithful_stereo_raw = self._is_dextrah_faithful_mode() and need_teacher_dual
            if faithful_stereo_raw:
                img_left, img_right = self._get_faithful_stereo_pair()
                out["img_left"] = img_left
                out["img_right"] = img_right
                out["vision_shared"] = True
                out["right_img_preflipped"] = bool(
                    getattr(self.cfg, "dextrah_right_img_flip", True)
                )
                if bool(getattr(self.cfg, "dextrah_export_stereo_masks", True)):
                    mask_left, mask_right = self._get_faithful_stereo_masks(img_left, img_right)
                    out["mask_left"] = mask_left
                    out["mask_right"] = mask_right
            else:
                vision_rgb, vision_depth = self._get_raw_student_vision_rgbd()
                out["vision_rgb"] = vision_rgb
                if vision_depth is not None:
                    out["vision_depth"] = vision_depth
                out["vision_shared"] = True
        else:
            vision_obs = self._get_vision_obs()
            out["vision_obs"] = vision_obs.unsqueeze(1).expand(-1, 2, -1)
            out["critic_vision_obs"] = out["vision_obs"]

        if (
            self._is_dextrah_faithful_mode()
            and bool(getattr(self.cfg, "distill_dual_obs", False))
            and (("img_left" not in out) or ("img_right" not in out))
        ):
            img_left, img_right = self._get_faithful_stereo_pair()
            out["img_left"] = img_left
            out["img_right"] = img_right
            out["vision_shared"] = True
            out["right_img_preflipped"] = bool(
                getattr(self.cfg, "dextrah_right_img_flip", True)
            )
            if bool(getattr(self.cfg, "dextrah_export_stereo_masks", True)):
                mask_left, mask_right = self._get_faithful_stereo_masks(img_left, img_right)
                out["mask_left"] = mask_left
                out["mask_right"] = mask_right

        if need_teacher_dual and isinstance(teacher_state, dict):
            teacher_policy_raw = teacher_state.get("teacher_policy", teacher_state.get("policy", None))
            teacher_critic_raw = teacher_state.get("teacher_critic", teacher_state.get("critic", None))
            if teacher_policy_raw is None or teacher_critic_raw is None:
                raise RuntimeError(
                    "[handover_tshape] distill_dual_obs=True requires teacher policy/critic tensors from state mode."
                )
            out["teacher_policy"] = teacher_policy_raw
            out["teacher_critic"] = teacher_critic_raw
            if bool(getattr(self.cfg, "dextrah_trlp4_distill_mode", False)):
                out["expert_policy"] = teacher_policy_raw

        if bool(getattr(self.cfg, "vision_zero_input", False)):
            for key in (
                "img_left",
                "img_right",
                "vision_rgb",
                "vision_depth",
                "vision_obs",
                "critic_vision_obs",
            ):
                value = out.get(key, None)
                if torch.is_tensor(value):
                    out[key] = torch.zeros_like(value)

        if bool(getattr(self.cfg, "strict_vision_only_policy", False)) and (
            not self._vision_guard_checked
        ):
            leaked = sorted(self._banned_obs_keys.intersection(out.keys()))
            if leaked and (not bool(getattr(self.cfg, "enable_privileged_debug_dict", False))):
                raise RuntimeError(
                    f"[{self.cfg.task_name}][policy] privileged debug keys leaked into obs_dict: {leaked}"
                )
            self._vision_guard_checked = True

        return out

    def _get_observations(self) -> dict:
        if bool(getattr(self.cfg, "use_vision", False)):
            return self._get_observations_vision_mode()
        return self._get_observations_state_mode()

    def get_joint_states(self, robot, joint_names):
        all_names = robot.data.joint_names
        indices = [all_names.index(name) for name in joint_names]
        pos = robot.data.joint_pos[:, indices]
        vel = robot.data.joint_vel[:, indices]
        return pos, vel

    def _get_rewards(self) -> torch.Tensor:
        env_offsets = self.env_origins

        rh_pos = self.right_robot.data.body_pos_w[:, 6, :3] - env_offsets
        rh_index = self.right_robot.data.body_pos_w[:, -4, :3] - env_offsets
        rh_thumb = self.right_robot.data.body_pos_w[:, -3, :3] - env_offsets
        rh_middle = self.right_robot.data.body_pos_w[:, -2, :3] - env_offsets
        rh_ring = self.right_robot.data.body_pos_w[:, -1, :3] - env_offsets

        lh_pos = self.left_robot.data.body_pos_w[:, 6, :3] - env_offsets
        lh_index = self.left_robot.data.body_pos_w[:, -4, :3] - env_offsets
        lh_thumb = self.left_robot.data.body_pos_w[:, -3, :3] - env_offsets
        lh_middle = self.left_robot.data.body_pos_w[:, -2, :3] - env_offsets
        lh_ring = self.left_robot.data.body_pos_w[:, -1, :3] - env_offsets

        cube_pos = self.cube.data.root_pos_w[:, :3] - env_offsets
        cube_quat = self.cube.data.root_quat_w
        if self.cfg.events:
            cube_init_pos = self.initial_cube_pos - env_offsets
        else:
            cube_init_pos = self.initial_cube_pos

        # Compute fingertip distances used to classify open/closed hands.
        dist_in_th = torch.norm(rh_index - rh_thumb, p=2, dim=-1)
        dist_mi_th = torch.norm(rh_middle - rh_thumb, p=2, dim=-1)
        dist_ri_th = torch.norm(rh_ring - rh_thumb, p=2, dim=-1)
        dist_rf = dist_in_th + dist_mi_th + dist_ri_th

        dist_in_th_lh = torch.norm(lh_index - lh_thumb, p=2, dim=-1)
        dist_mi_th_lh = torch.norm(lh_middle - lh_thumb, p=2, dim=-1)
        dist_ri_th_lh = torch.norm(lh_ring - lh_thumb, p=2, dim=-1)
        dist_lf = dist_in_th_lh + dist_mi_th_lh + dist_ri_th_lh

        reward = self.env_origins.new_zeros((self.num_envs,))
        threshold = 0.05
        _goal_th = float(getattr(self.cfg, "handover_goal_threshold", 0.08))
        collision_penalty = self.env_origins.new_zeros((self.num_envs,))
        arm_collision_penalty = self.env_origins.new_zeros((self.num_envs,))
        hand_collision_penalty = self.env_origins.new_zeros((self.num_envs,))
        arm_min_dist = torch.full((self.num_envs,), float("inf"), device=self.device)
        hand_min_dist = torch.full((self.num_envs,), float("inf"), device=self.device)
        if bool(getattr(self.cfg, "handover_robot_collision_penalty", False)):
            arm_th = float(getattr(self.cfg, "handover_robot_collision_arm_threshold", 0.03))
            hand_th = float(getattr(self.cfg, "handover_robot_collision_hand_threshold", 0.005))
            arm_coef = float(getattr(self.cfg, "handover_robot_collision_arm_coef", 0.01))
            hand_coef = float(getattr(self.cfg, "handover_robot_collision_hand_coef", 0.005))

            right_arm_pts = self.right_robot.data.body_pos_w[:, 1:7, :3] - env_offsets.unsqueeze(1)
            left_arm_pts = self.left_robot.data.body_pos_w[:, 1:7, :3] - env_offsets.unsqueeze(1)
            arm_min_dist = torch.cdist(right_arm_pts, left_arm_pts, p=2).amin(dim=2).amin(dim=1)

            right_hand_pts = torch.stack([rh_pos, rh_index, rh_thumb, rh_middle, rh_ring], dim=1)
            left_hand_pts = torch.stack([lh_pos, lh_index, lh_thumb, lh_middle, lh_ring], dim=1)
            hand_min_dist = torch.cdist(right_hand_pts, left_hand_pts, p=2).amin(dim=2).amin(dim=1)

            if arm_th > 0.0 and arm_coef > 0.0:
                arm_pen = (torch.clamp(arm_th - arm_min_dist, min=0.0) / arm_th) ** 2
                arm_collision_penalty = arm_coef * arm_pen
                collision_penalty = collision_penalty + arm_collision_penalty
            if hand_th > 0.0 and hand_coef > 0.0:
                hand_pen = (torch.clamp(hand_th - hand_min_dist, min=0.0) / hand_th) ** 2
                hand_collision_penalty = hand_coef * hand_pen
                collision_penalty = collision_penalty + hand_collision_penalty
        else:
            arm_th = float(getattr(self.cfg, "handover_robot_collision_arm_threshold", 0.03))
            hand_th = float(getattr(self.cfg, "handover_robot_collision_hand_threshold", 0.005))

        # Catalog-scaled (rod: L=0.30 -> +0.11 / -0.09, unchanged). +z of the mesh = receiver end.
        _axis_len = float(getattr(self.cfg, "handover_object_axis_len", 0.30))
        right_object_contact_local = self.env_origins.new_tensor(
            [0.0, 0.0, (0.11 / 0.30) * _axis_len]
        ).repeat(self.num_envs, 1)

        capture_step = int(getattr(self.cfg, "demo_handover_capture_step", 240))
        demo_handover_capture = (
            (self.episode_length_buf >= capture_step)
            & (~self.demo_handover_ref_valid)
        )
        if torch.any(demo_handover_capture):
            self.demo_handover_ref_pos[demo_handover_capture] = cube_pos[
                demo_handover_capture
            ].detach()
            self.demo_handover_ref_valid[demo_handover_capture] = True

        # Define task-specific target positions.
        # Make L_GRASP cube-aware by rotating the left grasp offset with the cube's initial yaw.
        # This matches the p_l_grasp computation in generate_demo.py.
        # Extract the rotation delta as initial_cube_quat * inv(q_cube_default), then apply it to the default offset.
        _q_cube_default = self.env_origins.new_tensor(
            [0.7071, 0.0, 0.7071, 0.0]
        ).repeat(self.num_envs, 1)
        if bool(getattr(self.cfg, "handover_grasp_waypoint_live", True)):
            # Live anchor (see cfg comment): L_GRASP is only consumed pre-grasp
            # (subgoals 0/1 and their shaping column), where the rod sits free on
            # the supporters, so the live pose is well-defined.
            _grasp_anchor_pos = cube_pos
            _grasp_anchor_quat = cube_quat
        else:
            _grasp_anchor_pos = cube_init_pos
            _grasp_anchor_quat = self.initial_cube_quat
        _delta_cube_quat = quat_mul(_grasp_anchor_quat, quat_inv(_q_cube_default))
        _grasp_offset_default = self.env_origins.new_tensor(
            [-(0.09 / 0.30) * _axis_len, -0.125, 0.068]
        ).repeat(self.num_envs, 1)
        _grasp_offset_world = quat_apply(_delta_cube_quat, _grasp_offset_default)
        L_GRASP = _grasp_anchor_pos + _grasp_offset_world
        # Keep reward targets aligned with trl_utils/generate_demo.py.
        # The right hand moves from home directly to the X/Z-aligned clearance
        # point, then approaches the final handover point along Y.
        O_HANDOVER_NOMINAL = self.env_origins.new_tensor([0.0, 0.40, 1.23]).repeat(
            self.num_envs, 1
        )
        O_HANDOVER = torch.where(
            self.demo_handover_ref_valid.unsqueeze(-1),
            self.demo_handover_ref_pos,
            O_HANDOVER_NOMINAL,
        )
        right_base_approach = self.env_origins.new_tensor([0.03, -0.11, 0.0]).repeat(
            self.num_envs, 1
        )
        R_HANDOVER = O_HANDOVER_NOMINAL + self.env_origins.new_tensor([0.05, -0.01, 0.1]).repeat(
            self.num_envs, 1
        )
        R_CLEARANCE = R_HANDOVER + self.env_origins.new_tensor([0.0, -0.05, 0.0]).repeat(
            self.num_envs, 1
        )
        object_receive_point = cube_pos + quat_apply(cube_quat, right_object_contact_local)
        fixed_receive_point = R_HANDOVER - right_base_approach
        O_TARGET = self.goal_positions - env_offsets
        L_INIT = self.env_origins.new_tensor([-0.3195, 0.4150, 1.3263])
        R_INIT = self.env_origins.new_tensor([0.3520, 0.4038, 1.3374])
        if bool(getattr(self.cfg, "handover_full_hand_action", False)):
            # Dexterous hands need a different open/closed test.
            #
            # The fingertip-to-thumb distance below works for a binary gripper, whose
            # joints are only ever 0.0 or 1.57. It breaks for a 16-DOF hand: a correct
            # PREGRASP opposes the thumb before closing, which collapses dist_rf below
            # the 0.35 threshold while the hand is still wide open. R_OPEN then never
            # fires and the right hand is pinned at subgoal 3 forever (observed: 0/25
            # demos across three grasp variants, while the legacy path got 20/25).
            #
            # Measure curl directly instead, from the flexion joints only -- excluding
            # abduction (0,4,8) and thumb opposition (12), since those position the hand
            # rather than close it. One threshold keeps OPEN/CLOSE complementary, so no
            # dead zone can stall a stage.
            # Slots resolved BY NAME: Isaac interleaves the fingers, so canonical joint
            # numbers are not slot indices (see _hand_slot_map).
            th = float(getattr(self.cfg, "handover_dex_flex_threshold", 0.80))
            r_flex = self.right_robot.data.joint_pos[:, 6:22][:, self.right_flex_slots].mean(dim=-1)
            l_flex = self.left_robot.data.joint_pos[:, 6:22][:, self.left_flex_slots].mean(dim=-1)
            R_OPEN = r_flex < th
            R_CLOSE = r_flex >= th
            L_OPEN = l_flex < th
            L_CLOSE = l_flex >= th
        else:
            R_OPEN = dist_rf > 0.35
            R_CLOSE = dist_rf <= 0.35
            L_OPEN = dist_lf > 0.5
            L_CLOSE = dist_lf <= 0.5

        # Subgoal 0: move the left hand to the object.
        dist = torch.norm(L_GRASP - lh_pos, p=2, dim=-1)
        condition = (self.subgoal[:, 0] == 0.0) & (self.subgoal[:, 1] == 0.0)
        rew = (dist < threshold) & L_OPEN
        reward += torch.where(condition, rew.float(), torch.zeros_like(rew))
        self.subgoal[:, 1] = torch.where(condition & rew, 1.0, self.subgoal[:, 1])

        # Subgoal 1: grasp the object with the left hand.
        dist = torch.norm(L_GRASP - lh_pos, p=2, dim=-1)
        condition = (self.subgoal[:, 0] == 0.0) & (self.subgoal[:, 1] == 1.0)
        rew = (dist < threshold) & L_CLOSE
        reward += torch.where(condition, rew.float(), torch.zeros_like(rew))
        self.subgoal[:, 1] = torch.where(condition & rew, 2.0, self.subgoal[:, 1])

        # Subgoal 2: move the left-held object receive point to the fixed right receive point.
        dist = torch.norm(fixed_receive_point - object_receive_point, p=2, dim=-1)
        condition = (self.subgoal[:, 0] == 0.0) & (self.subgoal[:, 1] == 2.0)
        # This gate was the ONLY 0.03 in the reward -- every other positional gate uses
        # `threshold` (0.05). Measured over 150 scripted-expert episodes, the envs that FAIL
        # this stage have a median closest approach of **0.0316 m** against the 0.030 m gate,
        # i.e. they miss by ~1.6 mm; 21-23 of every ~28 failures land within 5 cm and some
        # reach 0.0030 m. The success and failure distributions overlap heavily (success
        # median 0.0156, failure p10 0.0137), which is a threshold cutting through the middle
        # of one distribution, not two different behaviours. The handover IS being performed;
        # a 3 cm gate on a 5.4 cm rod is simply tighter than the system's repeatability, and
        # because the right arm is gated on this stage it silently killed ~42% of all envs.
        rew = (dist < float(getattr(self.cfg, "handover_receive_threshold", 0.05))) & L_CLOSE
        reward += torch.where(condition, rew.float(), torch.zeros_like(rew))
        self.subgoal[:, 1] = torch.where(condition & rew, 3.0, self.subgoal[:, 1])

        # Subgoal 3: move the right hand directly to the X/Z-aligned clearance point.
        dist = torch.norm(R_CLEARANCE - rh_pos, p=2, dim=-1)
        condition = (self.subgoal[:, 0] == 0.0) & (self.subgoal[:, 1] == 3.0)
        rew = (dist < threshold) & R_OPEN
        reward += torch.where(condition, rew.float(), torch.zeros_like(rew))
        self.subgoal[:, 0] = torch.where(condition & rew, 1.0, self.subgoal[:, 0])

        # Subgoal 4: approach the final handover target along Y while staying open.
        dist = torch.norm(R_HANDOVER - rh_pos, p=2, dim=-1)
        condition = (self.subgoal[:, 0] == 1.0) & (self.subgoal[:, 1] == 3.0)
        rew = (dist < threshold) & R_OPEN
        reward += torch.where(condition, rew.float(), torch.zeros_like(rew))
        self.subgoal[:, 0] = torch.where(condition & rew, 2.0, self.subgoal[:, 0])

        # Subgoal 5: grasp the object with the right hand.
        dist = torch.norm(R_HANDOVER - rh_pos, p=2, dim=-1)
        condition = (self.subgoal[:, 0] == 2.0) & (self.subgoal[:, 1] == 3.0)
        rew = (dist < threshold) & R_CLOSE
        reward += torch.where(condition, rew.float(), torch.zeros_like(rew))
        self.subgoal[:, 0] = torch.where(condition & rew, 3.0, self.subgoal[:, 0])

        # Subgoal 6: release the object from the left hand.
        condition = (self.subgoal[:, 0] >= 3.0) & (self.subgoal[:, 1] == 3.0)
        rew = R_CLOSE & L_OPEN
        reward += torch.where(condition, rew.float(), torch.zeros_like(rew))
        self.subgoal[:, 1] = torch.where(condition & rew, 4.0, self.subgoal[:, 1])

        # Subgoal 7: after the left hand releases, move the object in the right
        # hand to the final target while the left hand returns home.
        dist = torch.norm(O_TARGET - cube_pos, p=2, dim=-1)
        condition = (self.subgoal[:, 0] == 3.0) & (self.subgoal[:, 1] >= 4.0)
        rew = (dist < _goal_th) & R_CLOSE
        reward += torch.where(condition, rew.float(), torch.zeros_like(rew))
        self.subgoal[:, 0] = torch.where(condition & rew, 4.0, self.subgoal[:, 0])

        # Subgoal 8: hold the object at the right-hand target.
        dist = torch.norm(O_TARGET - cube_pos, p=2, dim=-1)
        condition = self.subgoal[:, 0] == 4.0
        is_holding = (dist < _goal_th) & R_CLOSE
        self.hold_timer = torch.where(
            condition & is_holding,
            self.hold_timer + 1.0,
            torch.zeros_like(self.hold_timer),
        )
        rew = self.hold_timer > 10
        reward += torch.where(condition, rew.float(), torch.zeros_like(rew))
        self.subgoal[:, 0] = torch.where(condition & rew, 5.0, self.subgoal[:, 0])

        # Subgoal 9: release the object from the right hand.
        condition = self.subgoal[:, 0] == 5.0
        rew = R_OPEN
        reward += torch.where(condition, rew.float(), torch.zeros_like(rew))
        self.subgoal[:, 0] = torch.where(condition & rew, 6.0, self.subgoal[:, 0])

        # Subgoal 10: move the left hand back to its initial position.
        dist = torch.norm(L_INIT - lh_pos, p=2, dim=-1)
        condition = self.subgoal[:, 1] == 4.0
        rew = (dist < threshold) & L_OPEN
        reward += torch.where(condition, rew.float(), torch.zeros_like(rew))
        self.subgoal[:, 1] = torch.where(condition & rew, 5.0, self.subgoal[:, 1])

        # Subgoal 11: move the right hand back home after releasing the object.
        dist = torch.norm(R_INIT - rh_pos, p=2, dim=-1)
        condition = self.subgoal[:, 0] == 6.0
        rew = (dist < threshold) & R_OPEN
        reward += torch.where(condition, rew.float(), torch.zeros_like(rew))
        self.subgoal[:, 0] = torch.where(condition & rew, 7.0, self.subgoal[:, 0])

        # print(self.subgoal, "\n")

        # Potential-based dense shaping over the stage cascade above. Computed
        # AFTER the cascade so phi reads the stage the env actually carries into
        # the next step, keeping phi a pure function of stored state (which is
        # what the policy-invariance proof requires).
        if bool(getattr(self.cfg, "handover_dense_shaping", False)):
            d_r_clearance = torch.norm(R_CLEARANCE - rh_pos, p=2, dim=-1)
            d_r_handover = torch.norm(R_HANDOVER - rh_pos, p=2, dim=-1)
            d_object_goal = torch.norm(O_TARGET - cube_pos, p=2, dim=-1)
            d_r_home = torch.norm(R_INIT - rh_pos, p=2, dim=-1)
            d_l_grasp = torch.norm(L_GRASP - lh_pos, p=2, dim=-1)
            d_l_receive = torch.norm(fixed_receive_point - object_receive_point, p=2, dim=-1)
            d_l_home = torch.norm(L_INIT - lh_pos, p=2, dim=-1)
            no_target = torch.zeros_like(d_object_goal)
            # One column per stage; stages with no spatial target (release, done)
            # contribute nothing, so the potential simply goes flat there.
            right_stage_dists = torch.stack(
                [d_r_clearance,     # 0 reach the clearance point
                 d_r_handover,      # 1 approach the handover point
                 d_r_handover,      # 2 close on the object
                 d_object_goal,     # 3 carry the object to the goal
                 d_object_goal,     # 4 hold it there
                 no_target,         # 5 release
                 d_r_home,          # 6 return home
                 no_target],        # 7 done
                dim=1,
            )
            left_stage_dists = torch.stack(
                [d_l_grasp,         # 0 reach the grasp pose
                 d_l_grasp,         # 1 close on the object
                 d_l_receive,       # 2 present it at the receive point
                 no_target,         # 3 wait for the right hand, then release
                 d_l_home,          # 4 return home
                 no_target],        # 5 done
                dim=1,
            )
            phi = stage_potential(
                [self.subgoal[:, 0], self.subgoal[:, 1]],
                [right_stage_dists, left_stage_dists],
                coef=float(getattr(self.cfg, "handover_dense_shaping_coef", 0.5)),
            )
            shaping = potential_shaping(
                phi, self._shaping_phi_prev, self._shaping_phi_valid,
                gamma=float(getattr(self.cfg, "handover_dense_shaping_gamma", 0.99)),
            )
            reward += shaping
            self._shaping_phi_prev = phi.detach()
            self._shaping_phi_valid[:] = True
            self.extras["dense_shaping"] = shaping
            self.extras["dense_shaping_phi"] = phi

        success_target = O_TARGET
        dist_goal = torch.norm(success_target - cube_pos, p=2, dim=-1)
        region_hold_steps = float(getattr(self.cfg, "success_region_hold_steps", 5.0))
        home_hold_steps = float(getattr(self.cfg, "success_home_hold_steps", 10.0))
        home_th = float(getattr(self.cfg, "success_home_th", 0.08))

        # Final success follows the original TRLD criterion: reach the final object target,
        # then return both hands home.
        goal_region_now = (self.subgoal[:, 0] >= 4.0) & (dist_goal < _goal_th)
        self.in_success_region[:] = goal_region_now
        self.success_region_timer = torch.where(
            goal_region_now,
            self.success_region_timer + 1.0,
            torch.zeros_like(self.success_region_timer),
        )
        target_held = self.success_region_timer >= region_hold_steps
        self.target_held[:] = target_held
        self.success_region_latch.logical_or_(target_held)

        d_home_r = torch.norm(R_INIT - rh_pos, p=2, dim=-1)
        d_home_l = torch.norm(L_INIT - lh_pos, p=2, dim=-1)
        hands_home_now = (
            self.success_region_latch
            & (d_home_r < home_th)
            & (d_home_l < home_th)
            & R_OPEN
            & L_OPEN
        )
        self.hands_home[:] = hands_home_now
        self.home_timer = torch.where(
            hands_home_now,
            self.home_timer + 1.0,
            torch.zeros_like(self.home_timer),
        )

        success_now = self.home_timer >= home_hold_steps
        new_strict_success = success_now & (~self.success_latched)
        reward += new_strict_success.float()
        self.success_latched.logical_or_(success_now)
        self.success_buf[:] = self.success_latched
        success = self.success_buf.clone()
        self.extras["success"] = success
        self.extras["in_success_region"] = self.in_success_region
        self.extras["target_held"] = self.target_held
        self.extras["success_region_latch"] = self.success_region_latch
        self.extras["success_region_timer"] = self.success_region_timer
        self.extras["home_timer"] = self.home_timer
        self.extras["hands_home"] = self.hands_home
        self.extras["demo_handover_ref_valid"] = self.demo_handover_ref_valid
        self.extras["robot_collision_penalty"] = collision_penalty
        self.extras["robot_arm_collision_penalty"] = arm_collision_penalty
        self.extras["robot_hand_collision_penalty"] = hand_collision_penalty
        self.extras["robot_arm_min_dist"] = arm_min_dist
        self.extras["robot_hand_min_dist"] = hand_min_dist
        reward -= collision_penalty
        self._update_debug_markers(
            target_local=success_target,
            right_home_local=R_INIT,
            left_home_local=L_INIT,
            object_receive_target_local=fixed_receive_point,
            stage_flags=torch.stack(
                [
                    self.subgoal[:, 1] >= 2.0,
                    self.subgoal[:, 1] >= 3.0,
                    self.subgoal[:, 0] >= 3.0,
                    self.success_region_latch,
                    self.success_latched,
                ],
                dim=1,
            ),
            success_radius=threshold,
        )

        LOG_EVERY = int(getattr(self.cfg, "simple_log_every", 30))
        # Use index 0 assuming episode_length_buf is synchronized across environments.
        current_step = int(self.episode_length_buf[0].item())

        if LOG_EVERY > 0 and (current_step % LOG_EVERY == 0):
            # 1. Basic statistics.
            total_envs = self.num_envs
            total_suc_count = int(self.success_buf.sum().item())
            mean_reward = float(reward.mean().item())
            
            # 2. Per-subgoal stage distribution histogram.
            # RH uses stages 0-6 because it includes one clearance waypoint.
            # LH still uses stages 0-5.
            rh_stages = self.subgoal[:, 0].long().clamp(0, 6)
            lh_stages = self.subgoal[:, 1].long().clamp(0, 5)
            
            rh_hist = torch.bincount(rh_stages, minlength=7)
            lh_hist = torch.bincount(lh_stages, minlength=6)
            
            # Format as "0:count|1:count|...".
            rh_str = "RH[" + "|".join([f"{k}:{int(rh_hist[k].item()):02d}" for k in range(7)]) + "]"
            lh_str = "LH[" + "|".join([f"{k}:{int(lh_hist[k].item()):02d}" for k in range(6)]) + "]"
            rh_ge = [
                int((self.subgoal[:, 0] >= float(k)).sum().item()) for k in range(1, 7)
            ]
            lh_ge = [
                int((self.subgoal[:, 1] >= float(k)).sum().item()) for k in range(1, 6)
            ]
            rh_ge_str = "RH>=" + "|".join([f"{k}:{rh_ge[k - 1]:02d}" for k in range(1, 7)])
            lh_ge_str = "LH>=" + "|".join([f"{k}:{lh_ge[k - 1]:02d}" for k in range(1, 6)])
            goal_n = int(goal_region_now.sum().item())
            target_hold_n = int(target_held.sum().item())
            home_n = int(hands_home_now.sum().item())
            latch_n = int(self.success_region_latch.sum().item())
            max_goal_t = float(self.success_region_timer.max().item())
            max_home_t = float(self.home_timer.max().item())
            d_goal_min = float(dist_goal.min().item())
            d_home_r_min = float(d_home_r.min().item())
            d_home_l_min = float(d_home_l.min().item())
            d_clear = torch.norm(R_CLEARANCE - rh_pos, p=2, dim=-1)
            d_final = torch.norm(R_HANDOVER - rh_pos, p=2, dim=-1)
            clear_now = (d_clear < threshold) & R_OPEN
            final_open_now = (d_final < threshold) & R_OPEN
            clear_n = int(clear_now.sum().item())
            final_open_n = int(final_open_now.sum().item())
            d_clear_min = float(d_clear.min().item())
            d_final_min = float(d_final.min().item())
            coll_pen_mean = float(collision_penalty.mean().item())
            arm_pen_mean = float(arm_collision_penalty.mean().item())
            hand_pen_mean = float(hand_collision_penalty.mean().item())
            arm_min = float(arm_min_dist.min().item())
            hand_min = float(hand_min_dist.min().item())

            print(
                f"[St:{current_step:03d}] "
                f"Rew={mean_reward:6.3f} "
                f"Succ={total_suc_count}/{total_envs} | "
                f"{rh_str} {lh_str} | "
                f"{rh_ge_str} {lh_ge_str} | "
                f"goal_now={goal_n:02d} target5={target_hold_n:02d} "
                f"latch={latch_n:02d} home_now={home_n:02d} "
                f"tGoalMax={max_goal_t:.0f} tHomeMax={max_home_t:.0f} "
                f"dMin(goal/Rhome/Lhome)={d_goal_min:.3f}/{d_home_r_min:.3f}/{d_home_l_min:.3f} "
                f"clear_now={clear_n:02d} final_open={final_open_n:02d} "
                f"dMin(clear/final)={d_clear_min:.3f}/{d_final_min:.3f} "
                f"collPen={coll_pen_mean:.4f} armPen={arm_pen_mean:.4f} "
                f"handPen={hand_pen_mean:.4f} "
                f"dMin(arm/hand)={arm_min:.3f}/{hand_min:.3f} "
                f"th(arm/hand)={arm_th:.3f}/{hand_th:.3f}"
                # Dexterous mode: report MEASURED mean flexion against the open/closed
                # threshold. Commanded targets say what we asked for; these say what the
                # fingers actually reached, which is what the stage gate tests. A hand
                # blocked by the rod plateaus below its commanded angle.
                + (
                    f" | flex(L/R)={float(l_flex.mean()):.3f}/{float(r_flex.mean()):.3f} "
                    f"max={float(l_flex.max()):.3f}/{float(r_flex.max()):.3f} "
                    f"th={float(getattr(self.cfg, 'handover_dex_flex_threshold', 0.80)):.2f} "
                    f"L_CLOSE={int(L_CLOSE.sum()):02d} R_OPEN={int(R_OPEN.sum()):02d}"
                    if bool(getattr(self.cfg, "handover_full_hand_action", False))
                    else ""
                )
            )
        # =========================================================================

        return reward
    

    def _ensure_debug_markers(self) -> None:
        if self._debug_markers_initialized:
            return
        self.debug_success_region_marker = VisualizationMarkers(self.cfg.debug_success_region_cfg)
        self.debug_home_right_marker = VisualizationMarkers(self.cfg.debug_home_right_cfg)
        self.debug_home_left_marker = VisualizationMarkers(self.cfg.debug_home_left_cfg)
        self.debug_object_receive_target_marker = VisualizationMarkers(
            self.cfg.debug_object_receive_target_cfg
        )
        self.debug_stage_ball_markers = [
            VisualizationMarkers(self.cfg.debug_stage_left_grasp_cfg),
            VisualizationMarkers(self.cfg.debug_stage_handover_cfg),
            VisualizationMarkers(self.cfg.debug_stage_right_grasp_cfg),
            VisualizationMarkers(self.cfg.debug_stage_target_hold_cfg),
            VisualizationMarkers(self.cfg.debug_stage_success_cfg),
        ]
        self._debug_markers_initialized = True

    def enable_debug_markers(self, enabled: bool = True, vis_every: int = 1) -> None:
        self._debug_markers_enabled = bool(enabled)
        if self._debug_markers_enabled:
            self._debug_vis_every = max(1, int(vis_every))
            self._ensure_debug_markers()
            if not self._debug_marker_enable_logged:
                print(
                    f"[handover_tshape][debug] markers enabled vis_every={self._debug_vis_every}"
                )
                self._debug_marker_enable_logged = True
        else:
            self._debug_vis_every = 10**9

    def _update_debug_markers(
        self,
        *,
        target_local: torch.Tensor,
        right_home_local: torch.Tensor,
        left_home_local: torch.Tensor,
        object_receive_target_local: torch.Tensor,
        stage_flags: torch.Tensor,
        success_radius: float,
    ) -> None:
        if (
            not self._debug_markers_enabled
            or self.num_envs <= 0
            or int(self.episode_length_buf[0].item()) % int(self._debug_vis_every) != 0
        ):
            return
        self._ensure_debug_markers()
        if not self._debug_marker_visualize_logged:
            print("[handover_tshape][debug] marker visualize active")
            self._debug_marker_visualize_logged = True

        target_world = target_local + self.env_origins
        region_scale = torch.full(
            (self.num_envs, 3),
            2.0 * float(success_radius),
            device=self.device,
            dtype=torch.float32,
        )
        self.debug_success_region_marker.visualize(
            translations=target_world,
            scales=region_scale,
        )

        right_home_world = self.env_origins + right_home_local.view(1, 3)
        left_home_world = self.env_origins + left_home_local.view(1, 3)
        self.debug_home_right_marker.visualize(translations=right_home_world)
        self.debug_home_left_marker.visualize(translations=left_home_world)
        self.debug_object_receive_target_marker.visualize(
            translations=object_receive_target_local + self.env_origins
        )

        for idx, marker in enumerate(self.debug_stage_ball_markers):
            trans = self.env_origins + self._debug_stage_ball_offsets[idx].view(1, 3)
            active = stage_flags[:, idx].to(dtype=torch.bool)
            trans = trans.clone()
            trans[~active, 2] = -100.0
            marker.visualize(translations=trans)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length
        # print("time out : ", time_out)

        cube_pos = self.cube.data.root_pos_w[:, :3]
        fall_condition = cube_pos[:, 2] < self.cfg.fall_dist
        done_success = (
            self.success_latched.clone()
            if hasattr(self, "success_latched")
            else torch.zeros_like(time_out, dtype=torch.bool)
        )
        terminated = fall_condition
        if bool(getattr(self.cfg, "terminate_on_success", False)):
            terminated = terminated | done_success
        self.extras["done_success"] = done_success
        self.extras["done_fallen"] = fall_condition
        self.extras["done_timeout"] = time_out
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        env_offsets = self.env_origins[env_ids_t]
        num_envs_to_reset = int(env_ids_t.numel())
        self.last_grip_R[env_ids_t] = 1.0
        self.last_grip_L[env_ids_t] = 1.0
        self._prev_obs_pos_valid[env_ids_t] = False
        # Drop the previous episode's potential; carrying it across a reset would
        # pay out a large spurious shaping reward on the first step.
        self._shaping_phi_valid[env_ids_t] = False
        self._shaping_phi_prev[env_ids_t] = 0.0
        self._sample_handover_obs_biases(env_ids_t)
        self._apply_light_randomization()

        # Reset robots.
        self.right_robot.reset(env_ids)
        self.left_robot.reset(env_ids)

        # right_joint_pos = self.env_origins.new_tensor(
        #     [0.0, -1.57, -1.57, -3.14, -1.57, -3.14]
        # ).unsqueeze(0)
        # left_joint_pos = self.env_origins.new_tensor(
        #     [0.0, -1.57, 1.57, 0.0, 1.57, 3.14]
        # ).unsqueeze(0)

        right_joint_pos = self.env_origins.new_tensor(
            [
                self.cfg.right_robot_cfg.init_state.joint_pos["ra_shoulder_pan_joint"],
                self.cfg.right_robot_cfg.init_state.joint_pos["ra_shoulder_lift_joint"],
                self.cfg.right_robot_cfg.init_state.joint_pos["ra_elbow_joint"],
                self.cfg.right_robot_cfg.init_state.joint_pos["ra_wrist_1_joint"],
                self.cfg.right_robot_cfg.init_state.joint_pos["ra_wrist_2_joint"],
                self.cfg.right_robot_cfg.init_state.joint_pos["ra_wrist_3_joint"],
            ]
        ).unsqueeze(0)

        left_joint_pos = self.env_origins.new_tensor(
            [
                self.cfg.left_robot_cfg.init_state.joint_pos["la_shoulder_pan_joint"],
                self.cfg.left_robot_cfg.init_state.joint_pos["la_shoulder_lift_joint"],
                self.cfg.left_robot_cfg.init_state.joint_pos["la_elbow_joint"],
                self.cfg.left_robot_cfg.init_state.joint_pos["la_wrist_1_joint"],
                self.cfg.left_robot_cfg.init_state.joint_pos["la_wrist_2_joint"],
                self.cfg.left_robot_cfg.init_state.joint_pos["la_wrist_3_joint"],
            ]
        ).unsqueeze(0)
        right_joint_pos = right_joint_pos.repeat(num_envs_to_reset, 1)
        left_joint_pos = left_joint_pos.repeat(num_envs_to_reset, 1)

        right_joint_vel = torch.zeros((num_envs_to_reset, int(right_joint_pos.shape[-1])), device=self.device)
        left_joint_vel = torch.zeros((num_envs_to_reset, int(left_joint_pos.shape[-1])), device=self.device)
        if (
            bool(getattr(self.cfg, "handover_domain_rand", False))
            and bool(getattr(self.cfg, "handover_rand_initial_joint_velocity", True))
        ):
            vel_range = float(getattr(self.cfg, "handover_initial_joint_velocity_range", 0.03))
            if vel_range > 0.0:
                right_joint_vel = (torch.rand_like(right_joint_vel) * 2.0 - 1.0) * vel_range
                left_joint_vel = (torch.rand_like(left_joint_vel) * 2.0 - 1.0) * vel_range

        right_arm_joint_ids = self.right_robot.actuators["ur3_arm"].joint_indices
        left_arm_joint_ids = self.left_robot.actuators["ur3_arm"].joint_indices

        self.right_robot.write_joint_state_to_sim(
            position=right_joint_pos,
            velocity=right_joint_vel,
            joint_ids=right_arm_joint_ids,
            env_ids=env_ids,
        )
        self.left_robot.write_joint_state_to_sim(
            position=left_joint_pos,
            velocity=left_joint_vel,
            joint_ids=left_arm_joint_ids,
            env_ids=env_ids,
        )

        _legacy_reset = bool(getattr(self.cfg, "handover_legacy_dirty_reset", False))
        _hand_flag = int(getattr(self.cfg, "handover_reset_hand_joints", -1))
        _want_hand_reset = (not _legacy_reset) if _hand_flag < 0 else bool(_hand_flag)
        _vel_flag = int(getattr(self.cfg, "handover_reset_object_velocity", -1))
        _want_vel_zero = (not _legacy_reset) if _vel_flag < 0 else bool(_vel_flag)
        self._want_obj_vel_zero = _want_vel_zero  # consumed by the supporter block below
        if _want_hand_reset:
            # RESET THE HANDS TOO. Historically only the 6 arm joints were written
            # back, so the 16 Allegro joints per hand kept whatever pose AND velocity
            # the previous episode ended with -- training episodes were never i.i.d.,
            # and policies could become dependent on the carried state.
            #
            # WHAT to write matters a lot. Measured 2026-08-28 (state probe + seed-7
            # dose-response, teacher madit_teacher_v1): the actuator position targets
            # initialise to ZERO, so during sim-startup settling the PD drives every
            # hand joint toward 0 (thumb opposition: default 1.396 -> ~0.12, still
            # moving) BEFORE the first _reset_idx. Historical episode-1 evals start
            # from that settling snapshot. Teacher success vs initial thumb-opp pose:
            # 0.117 (snapshot) 0.80 | 0.5 (hw home) 0.76 | 0.0 0.71 | 1.396 (URDF
            # default) 0.63. The hand-joint VELOCITY is irrelevant (zeroing it alone:
            # 0.82). Default mode 4 writes the HARDWARE home pose (fingers 0, thumb
            # opposition 0.5 = the deploy contract's floor): i.i.d. episode starts
            # that match what the real robot presents.
            # Diagnostic sub-modes via handover_reset_hand_joints: 1 = zeros pos+vel,
            # 2 = zeros position only, 3 = velocity only, 4 = hardware home pose.
            _hand_mode = _hand_flag if _hand_flag > 0 else 4
            for robot in (self.right_robot, self.left_robot):
                hand_ids = robot.actuators["allegro_hand"].joint_indices
                hand_zero = torch.zeros_like(
                    robot.data.default_joint_pos[env_ids_t][:, hand_ids]
                )
                if _hand_mode == 4:
                    hand_zero[:, 1] = 0.5  # thumb opposition (grip obs joint)
                if _hand_mode in (1, 2, 4):
                    robot.write_joint_position_to_sim(
                        hand_zero, joint_ids=hand_ids, env_ids=env_ids
                    )
                if _hand_mode in (1, 3, 4):
                    robot.write_joint_velocity_to_sim(
                        torch.zeros_like(hand_zero), joint_ids=hand_ids, env_ids=env_ids
                    )
                # Align the commanded targets so the written pose is stationary
                # (the next _apply_action overwrites them; this kills the one-step
                # PD yank toward stale targets in between).
                robot.set_joint_position_target(
                    hand_zero, joint_ids=hand_ids, env_ids=env_ids
                )

        for name, obj in self._rigid_objects.items():
            # Skip supporters here because they are repositioned relative to the cube below.
            if name in ["supporter_1", "supporter_2"]:
                continue

            # If custom object pose randomization is disabled, keep the event-managed cube reset.
            if (
                self.cfg.events
                and name == "cube"
                and not bool(getattr(self.cfg, "handover_randomize_object_pose", False))
            ):
                continue

            obj_cfg = self.cfg.rigid_objects_cfg[name]

            pos_local = self.env_origins.new_tensor(obj_cfg.init_state.pos).repeat(
                num_envs_to_reset, 1
            )
            rot_local = self.env_origins.new_tensor(obj_cfg.init_state.rot).repeat(
                num_envs_to_reset, 1
            )
            if name == "cube" and bool(getattr(self.cfg, "handover_randomize_object_pose", False)):
                xy_range_cfg = getattr(self.cfg, "handover_object_pos_xy_range", (0.08, 0.08))
                if isinstance(xy_range_cfg, (int, float)):
                    xy_low = self.env_origins.new_tensor([-float(xy_range_cfg), -float(xy_range_cfg)])
                    xy_high = self.env_origins.new_tensor([float(xy_range_cfg), float(xy_range_cfg)])
                else:
                    xy_values = tuple(float(v) for v in xy_range_cfg)
                    if len(xy_values) >= 4:
                        xy_low = self.env_origins.new_tensor([xy_values[0], xy_values[2]])
                        xy_high = self.env_origins.new_tensor([xy_values[1], xy_values[3]])
                    else:
                        x_half = abs(xy_values[0]) if len(xy_values) > 0 else 0.08
                        y_half = abs(xy_values[1]) if len(xy_values) > 1 else x_half
                        xy_low = self.env_origins.new_tensor([-x_half, -y_half])
                        xy_high = self.env_origins.new_tensor([x_half, y_half])
                if bool(getattr(self.cfg, "handover_object_pose_bin_balanced", False)):
                    x_rand = self._sample_bin_balanced_uniform(
                        num_envs_to_reset,
                        float(xy_low[0].item()),
                        float(xy_high[0].item()),
                        int(getattr(self.cfg, "handover_object_pos_x_bins", 1)),
                    )
                    y_rand = self._sample_bin_balanced_uniform(
                        num_envs_to_reset,
                        float(xy_low[1].item()),
                        float(xy_high[1].item()),
                        int(getattr(self.cfg, "handover_object_pos_y_bins", 1)),
                    )
                    pos_local[:, 0] += x_rand
                    pos_local[:, 1] += y_rand
                else:
                    pos_local[:, :2] += xy_low + torch.rand((num_envs_to_reset, 2), device=self.device) * (xy_high - xy_low)
                z_range = float(getattr(self.cfg, "handover_object_pos_z_range", 0.0))
                if z_range > 0.0:
                    pos_local[:, 2] += (torch.rand((num_envs_to_reset,), device=self.device) * 2.0 - 1.0) * z_range
                yaw_deg_range = getattr(self.cfg, "handover_object_yaw_deg_range", 0.0)
                if isinstance(yaw_deg_range, (tuple, list)):
                    yaw_enabled = any(abs(float(v)) > 0.0 for v in yaw_deg_range)
                else:
                    yaw_enabled = float(yaw_deg_range) > 0.0
                if yaw_enabled:
                    yaw_quats = self._sample_yaw_quats_wxyz(num_envs_to_reset, yaw_deg_range).to(rot_local.dtype)
                    rot_local = F.normalize(_quat_mul_wxyz(yaw_quats, rot_local), dim=-1)
            pose = torch.cat([pos_local + env_offsets, rot_local], dim=-1)
            obj.write_root_pose_to_sim(pose, env_ids)
            if getattr(self, "_want_obj_vel_zero", True):
                # Pose-only writes keep the body's momentum: a falling cube teleported
                # to its spawn kept falling. Kill the velocity too.
                obj.write_root_velocity_to_sim(
                    torch.zeros((num_envs_to_reset, 6), device=self.device), env_ids
                )

        # 3) Reset subgoals and hold timers.
        self.subgoal[env_ids] = torch.zeros(2, device=self.device)
        self.subgoal[env_ids] = torch.zeros(2, device=self.device)
        self.right_handover_ref_pos[env_ids] = 0.0
        self.right_handover_ref_quat[env_ids] = 0.0
        self.right_handover_ref_quat[env_ids, 0] = 1.0
        self.right_handover_ref_valid[env_ids] = False
        self.demo_handover_ref_pos[env_ids] = 0.0
        self.demo_handover_ref_valid[env_ids] = False
        self.hold_timer[env_ids] = 0
        self.in_success_region[env_ids] = False
        self.success_buf[env_ids] = False
        self.target_held[env_ids] = False
        self.hands_home[env_ids] = False
        self.success_region_latch[env_ids] = False
        self.success_region_timer[env_ids] = 0
        self.home_timer[env_ids] = 0
        self.success_latched[env_ids] = False

        # 4) Re-visualize the goal marker.
        if hasattr(self, "goal_marker"):
            self.goal_marker.visualize(translations=self.goal_positions)

        _clob = int(getattr(self.cfg, "handover_legacy_initial_pose_clobber", -1))
        _legacy_clobber = (
            bool(getattr(self.cfg, "handover_legacy_dirty_reset", False))
            if _clob < 0
            else bool(_clob)
        )
        if self.cfg.events and _legacy_clobber:
            # Legacy: clobbered EVERY env's initial pose whenever ANY env reset, so in
            # desynchronised training L_GRASP effectively tracked the live cube.
            self.initial_cube_pos = self.cube.data.root_pos_w.clone()
            self.initial_cube_quat = self.cube.data.root_quat_w.clone()
        else:
            self.initial_cube_pos[env_ids_t] = self.cube.data.root_pos_w[env_ids_t, :3]
            self.initial_cube_quat[env_ids_t] = self.cube.data.root_quat_w[env_ids_t, :4]

        # if not self.camera.is_initialized:
        #     self.camera.initialize(self.scene.physics_sim_view)

        if self.right_dof_lower_limits is None:
            # # right arm/hand
            # right_limits = self.right_robot.data.joint_pos_limits[0]  # [22, 2]
            # self.right_dof_lower_limits = right_limits[:, 0].detach().clone().to(self.device)
            # self.right_dof_upper_limits = right_limits[:, 1].detach().clone().to(self.device)
            # # left arm/hand
            # left_limits = self.left_robot.data.joint_pos_limits[0]  # [22, 2]
            # self.left_dof_lower_limits = left_limits[:, 0].detach().clone().to(self.device)
            # self.left_dof_upper_limits = left_limits[:, 1].detach().clone().to(self.device)

            # Use demo-based limits.
            # self.right_dof_lower_limits = torch.tensor(
            #     [-0.14, -1.44, 0.64, -4.78, -1.22, 3.91, 0.0], device=self.device
            # )
            # self.right_dof_upper_limits = torch.tensor(
            #     [0.51, 0.23, 2.43, -3.96, -0.4, 5.51, 1.57], device=self.device
            # )
            # self.left_dof_lower_limits = torch.tensor(
            #     [-0.01, -3.23, -4.61, 0.26, 0.31, 0.78, 0.0], device=self.device
            # )
            # self.left_dof_upper_limits = torch.tensor(
            #     [0.41, -1.27, -0.29, 2.76, 0.8, 2.57, 1.57], device=self.device
            # )
            # Broad action limits used by the stable 2026-06-22 teacher run.
            # Thumb opposition still uses handover_thumb_oppose_lower_limit in
            # _pre_physics_step; the seventh range is the scalar grip command.
            self.right_dof_lower_limits = torch.tensor(
                [-0.2549, -1.5700, -1.5700, -4.8029, -1.5700, -3.1400, 0.0000], device=self.device
            )
            self.right_dof_upper_limits = torch.tensor(
                [0.7220, 0.1361, 2.6645, -3.1400, -0.5306, 5.5076, 1.5700], device=self.device
            )
            self.left_dof_lower_limits = torch.tensor(
                [-0.4197, -5.7584, -5.1701, -2.4004, -1.5700, 0.0000, 0.0000], device=self.device
            )
            self.left_dof_upper_limits = torch.tensor(
                [0.8257, -0.8284, 5.0597, 2.8967, 1.3468, 5.3065, 1.5700], device=self.device
            )

            # === Original demo-based limits kept for reference ===
            # self.right_dof_lower_limits = torch.tensor(
            #     [-0.14, -1.44, 0.63, -4.78, -1.22, 3.91, 0.0], device=self.device
            # )
            # self.right_dof_upper_limits = torch.tensor(
            #     [0.51, 0.25, 2.43, -3.96, -0.4, 5.51, 1.57], device=self.device
            # )
            # self.left_dof_lower_limits = torch.tensor(
            #     [-0.04, -3.4, -7.09, 0.08, 0.09, 0.78, 0.0], device=self.device
            # )
            # self.left_dof_upper_limits = torch.tensor(
            #     [0.7, -0.07, -0.0, 4.07, 0.82, 2.59, 1.57], device=self.device
            # )

            self.hand_dof_lower_limits = torch.tensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                device=self.device,
            )
            self.hand_dof_upper_limits = torch.tensor(
                [
                    0.0,
                    1.57,
                    0.0,
                    0.0,
                    1.57,
                    1.57,
                    1.57,
                    1.57,
                    1.57,
                    1.57,
                    1.57,
                    1.57,
                    1.57,
                    1.57,
                    1.57,
                    1.57,
                ],
                device=self.device,
            )

        if bool(getattr(self.cfg, "use_vision", False)):
            self._ensure_vision_buffers()
            self._vision_hist[env_ids] = 0.0
            self._vision_hist_valid[env_ids] = False
            dmin = int(getattr(self.cfg, "vision_latency_steps_min", 0))
            dmax = int(getattr(self.cfg, "vision_latency_steps_max", 0))
            if dmax < dmin:
                dmax = dmin
            if dmax > 0 or dmin > 0:
                self._vision_delay_steps[env_ids] = torch.randint(
                    low=dmin,
                    high=dmax + 1,
                    size=(env_ids.numel(),),
                    device=self.device,
                )
            else:
                self._vision_delay_steps[env_ids] = 0
            try:
                if self.vision_camera is not None:
                    self.vision_camera.reset(env_ids=env_ids)
                if self.vision_camera_head is not None:
                    self.vision_camera_head.reset(env_ids=env_ids)
                if self.vision_camera_front is not None:
                    self.vision_camera_front.reset(env_ids=env_ids)
                if self.vision_camera_front_right is not None:
                    self.vision_camera_front_right.reset(env_ids=env_ids)
            except TypeError:
                try:
                    if self.vision_camera is not None:
                        self.vision_camera.reset(env_ids.tolist())
                    if self.vision_camera_head is not None:
                        self.vision_camera_head.reset(env_ids.tolist())
                    if self.vision_camera_front is not None:
                        self.vision_camera_front.reset(env_ids.tolist())
                    if self.vision_camera_front_right is not None:
                        self.vision_camera_front_right.reset(env_ids.tolist())
                except Exception:
                    pass
            except Exception:
                pass
            self._configure_vision_camera_views(torch.as_tensor(env_ids, device=self.device))

        # Supporter randomization/repositioning.
        new_cube_pos_w = self.cube.data.root_pos_w[env_ids]
        num_envs_to_reset = len(env_ids)

        # 2. Expand offset vectors to the number of environments for broadcasting.
        # (3,) -> (N, 3)
        supp1_offset_expanded = self.supporter_1_world_offset.repeat(
            num_envs_to_reset, 1
        )
        supp2_offset_expanded = self.supporter_2_world_offset.repeat(
            num_envs_to_reset, 1
        )

        # 3. Position computation: new cube position plus world offset.
        # Use simple addition without applying a rotation matrix/quaternion.
        new_supp1_pos_w = new_cube_pos_w + supp1_offset_expanded
        new_supp2_pos_w = new_cube_pos_w + supp2_offset_expanded

        # 4. Rotation setup: supporters do not follow cube rotation and stay upright with identity rotation.
        # (4,) -> (N, 4)
        default_rot_expanded = self.default_rot.repeat(num_envs_to_reset, 1)

        # 5. Combine position and default rotation.
        new_supp1_pose_w = torch.cat([new_supp1_pos_w, default_rot_expanded], dim=-1)
        new_supp2_pose_w = torch.cat([new_supp2_pos_w, default_rot_expanded], dim=-1)

        # 6. Apply poses to the simulation.
        self.supporter_1.write_root_pose_to_sim(new_supp1_pose_w, env_ids=env_ids)
        self.supporter_2.write_root_pose_to_sim(new_supp2_pose_w, env_ids=env_ids)
        if getattr(self, "_want_obj_vel_zero", True):
            _zero6 = torch.zeros((num_envs_to_reset, 6), device=self.device)
            self.supporter_1.write_root_velocity_to_sim(_zero6, env_ids=env_ids)
            self.supporter_2.write_root_velocity_to_sim(_zero6, env_ids=env_ids)

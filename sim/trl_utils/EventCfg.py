# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Sequence

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg

def randomize_goal_position_uniform(
    env: DirectRLEnv,
    env_ids: Sequence[int],
    pos_range: dict[str, tuple[float, float]],
):
    if not hasattr(env, "goal_positions"):
        print("Warning: there's no 'goal_positions' in env")
        return
    if not hasattr(env, "base_goal_pos") or env.base_goal_pos is None:
        print("Warning: there's no 'base_goal_pos' in env")
        return

    num_envs_to_reset = len(env_ids)
    device = env.device

    base_pos_tensor = env.base_goal_pos.repeat(num_envs_to_reset, 1)

    offset = torch.zeros_like(base_pos_tensor)
    if "x" in pos_range:
        min_val, max_val = pos_range["x"]
        offset[:, 0] = (max_val - min_val) * torch.rand(num_envs_to_reset, device=device) + min_val
    if "y" in pos_range:
        min_val, max_val = pos_range["y"]
        offset[:, 1] = (max_val - min_val) * torch.rand(num_envs_to_reset, device=device) + min_val
    if "z" in pos_range:
        min_val, max_val = pos_range["z"]
        offset[:, 2] = (max_val - min_val) * torch.rand(num_envs_to_reset, device=device) + min_val

    new_local_pos = base_pos_tensor + offset

    env.goal_positions[env_ids] = new_local_pos + env.env_origins[env_ids]

    # if hasattr(env, "goal_marker") and env.goal_marker is not None:
    #     env.goal_marker.visualize(translations=env.goal_positions)

@configclass
class EventCfg:
    """Configuration for randomization."""

    # # -- robot
    # robot_physics_material = EventTermCfg(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="reset",
    #     min_step_count_between_reset=720,
    #     params={
    #         "asset_cfg": SceneEntityCfg("right_hand"),
    #         "static_friction_range": (0.7, 1.3),
    #         "dynamic_friction_range": (1.0, 1.0),
    #         "restitution_range": (1.0, 1.0),
    #         "num_buckets": 250,
    #     },
    # )
    # robot_physics_material = EventTermCfg(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="reset",
    #     min_step_count_between_reset=720,
    #     params={
    #         "asset_cfg": SceneEntityCfg("left_hand"),
    #         "static_friction_range": (0.7, 1.3),
    #         "dynamic_friction_range": (1.0, 1.0),
    #         "restitution_range": (1.0, 1.0),
    #         "num_buckets": 250,
    #     },
    # )

    # robot_joint_stiffness_and_damping = EventTermCfg(
    #     func=mdp.randomize_actuator_gains,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
    #         "stiffness_distribution_params": (0.75, 1.5),
    #         "damping_distribution_params": (0.3, 3.0),
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )
    # robot_joint_stiffness_and_damping = EventTermCfg(
    #     func=mdp.randomize_actuator_gains,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("left_hand", joint_names=".*"),
    #         "stiffness_distribution_params": (0.75, 1.5),
    #         "damping_distribution_params": (0.3, 3.0),
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )

    # -- cube
    cube_position_randomization = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": { 
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            },
        }
    )

    goal_position_randomization = EventTermCfg(
        func=randomize_goal_position_uniform,  
        mode="reset",
        params={
            "pos_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
            },
        }
    )

    # object_physics_material = EventTermCfg(
    #     func=mdp.randomize_rigid_body_material,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("cube"),
    #         "static_friction_range": (0.7, 1.3),
    #         "dynamic_friction_range": (1.0, 1.0),
    #         "restitution_range": (1.0, 1.0),
    #         "num_buckets": 250,
    #     },
    # )

    # object_scale_mass = EventTermCfg(
    #     func=mdp.randomize_rigid_body_mass,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("cube"),
    #         "mass_distribution_params": (0.5, 1.5),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )

    # -- cup
    # cup_position_randomization = EventTermCfg(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("cup"),
    #         "pose_range": {
    #             "x": (-0.02, 0.02),
    #             "y": (-0.02, 0.02),
    #             "z": (0.0, 0.0),
    #         },
    #         "velocity_range": {
    #             "x": (0.0, 0.0),
    #             "y": (0.0, 0.0),
    #             "z": (0.0, 0.0),
    #         },
    #     }
    # )

    # object_physics_material = EventTermCfg(
    #     func=mdp.randomize_rigid_body_material,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("cup"),
    #         "static_friction_range": (0.7, 1.3),
    #         "dynamic_friction_range": (1.0, 1.0),
    #         "restitution_range": (1.0, 1.0),
    #         "num_buckets": 250,
    #     },
    # )

    # object_scale_mass = EventTermCfg(
    #     func=mdp.randomize_rigid_body_mass,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("cup"),
    #         "mass_distribution_params": (0.5, 1.5),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )

    # # -- scene
    # reset_gravity = EventTermCfg(
    #     func=mdp.randomize_physics_scene_gravity,
    #     mode="interval",
    #     is_global_time=True,
    #     interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
    #     params={
    #         "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
    #         "operation": "add",
    #         "distribution": "gaussian",
    #     },
    # )






    # robot_joint_pos_limits = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
    #         "lower_limit_distribution_params": (0.00, 0.01),
    #         "upper_limit_distribution_params": (0.00, 0.01),
    #         "operation": "add",
    #         "distribution": "gaussian",
    #     },
    # )
    # robot_tendon_properties = EventTerm(
    #     func=mdp.randomize_fixed_tendon_parameters,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("right_hand", fixed_tendon_names=".*"),
    #         "stiffness_distribution_params": (0.75, 1.5),
    #         "damping_distribution_params": (0.3, 3.0),
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )






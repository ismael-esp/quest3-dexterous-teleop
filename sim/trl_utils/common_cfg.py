# utils/common_cfg.py
# SPDX-License-Identifier: BSD-3-Clause

"""my custom task에서 공통된 assets들을 불러오는 코드"""

from __future__ import annotations
import os
import math
import torch
import numpy as np
try:
    from gymnasium import spaces
except Exception:  # pragma: no cover
    try:
        from gym import spaces  # type: ignore
    except Exception:  # pragma: no cover
        class _SimpleBox:
            def __init__(self, low, high, shape, dtype=np.float32):
                self.low = np.full(shape, low, dtype=dtype)
                self.high = np.full(shape, high, dtype=dtype)
                self.shape = tuple(shape)
                self.dtype = dtype
        class _SpacesFallback:
            Box = _SimpleBox
        spaces = _SpacesFallback()
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass


# -------- 공통 로봇/테이블 생성기 --------
def make_right_robot_cfg(base_dir: str) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/RightRobot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                base_dir,
                "assets/robot_urdf_fix/ra_ur3_allegro_isaac_/ra_ur3_allegro_isaac_.usd",
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.2425, 0.0, 0.73),
            rot=(0.7071, 0, 0, -0.7071),
            joint_pos={
                "ra_shoulder_pan_joint": 0.0,
                "ra_shoulder_lift_joint": -1.57,
                "ra_elbow_joint": -1.57,
                "ra_wrist_1_joint": -3.14,
                "ra_wrist_2_joint": -1.57,
                "ra_wrist_3_joint": 3.14,
                "ra_joint_12_0": 1.396,  # 1.55,
            },
        ),
        actuators={
            "ur3_arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "ra_shoulder_pan_joint",
                    "ra_shoulder_lift_joint",
                    "ra_elbow_joint",
                    "ra_wrist_1_joint",
                    "ra_wrist_2_joint",
                    "ra_wrist_3_joint",
                ],
                effort_limit_sim=120.0,
                velocity_limit_sim=2.0,
                stiffness=1100.0,
                damping=300.0,
                armature=0.02,
                viscous_friction=0.02,
                # effort_limit_sim=10.0,
                # velocity_limit_sim=3.0,
                # stiffness=100.0,
                # damping=10.0,
            ),
            "allegro_hand": ImplicitActuatorCfg(
                joint_names_expr=[
                    # index          thumb            middle          ring
                    "ra_joint_0_0",
                    "ra_joint_12_0",
                    "ra_joint_4_0",
                    "ra_joint_8_0",
                    "ra_joint_1_0",
                    "ra_joint_13_0",
                    "ra_joint_5_0",
                    "ra_joint_9_0",
                    "ra_joint_2_0",
                    "ra_joint_14_0",
                    "ra_joint_6_0",
                    "ra_joint_10_0",
                    "ra_joint_3_0",
                    "ra_joint_15_0",
                    "ra_joint_7_0",
                    "ra_joint_11_0",
                ],
                # effort_limit_sim=0.7,
                # velocity_limit_sim=7.0,
                # stiffness=10.0,
                # damping=0.5,
                # armature=0.001,
                # viscous_friction=0.05,
                effort_limit_sim=1.200,
                velocity_limit_sim=30.0,
                stiffness=1200.0,
                damping=120.0,
                armature=0.005,
                viscous_friction=0.01,
                # effort_limit_sim=1.200,
                # velocity_limit_sim=30,#6.28,
                # stiffness=100.0,
                # damping=0.1,
            ),
        },
    )


def make_left_robot_cfg(base_dir: str) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/LeftRobot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                base_dir,
                "assets/robot_urdf_fix/la_ur3e_allegro_isaac_/la_ur3e_allegro_isaac_.usd",
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.2425, 0.0, 0.73),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                "la_shoulder_pan_joint": 0.0,
                "la_shoulder_lift_joint": -1.57,
                "la_elbow_joint": 1.57,
                "la_wrist_1_joint": 0.0,
                "la_wrist_2_joint": 1.57,
                "la_wrist_3_joint": 3.14,
                "la_joint_12_0": 1.396,  # 1.55,
            },
        ),
        actuators={
            "ur3_arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "la_shoulder_pan_joint",
                    "la_shoulder_lift_joint",
                    "la_elbow_joint",
                    "la_wrist_1_joint",
                    "la_wrist_2_joint",
                    "la_wrist_3_joint",
                ],
                effort_limit_sim=120.0,
                velocity_limit_sim=2.0,
                stiffness=1100.0,
                damping=300.0,
                armature=0.02,
                viscous_friction=0.02,
                # effort_limit_sim=10.0,
                # velocity_limit_sim=3.0,
                # stiffness=100.0,
                # damping=10.0,
            ),
            "allegro_hand": ImplicitActuatorCfg(
                joint_names_expr=[
                    # index          thumb            middle          ring
                    "la_joint_0_0",
                    "la_joint_12_0",
                    "la_joint_4_0",
                    "la_joint_8_0",
                    "la_joint_1_0",
                    "la_joint_13_0",
                    "la_joint_5_0",
                    "la_joint_9_0",
                    "la_joint_2_0",
                    "la_joint_14_0",
                    "la_joint_6_0",
                    "la_joint_10_0",
                    "la_joint_3_0",
                    "la_joint_15_0",
                    "la_joint_7_0",
                    "la_joint_11_0",
                ],
                # effort_limit_sim=0.7,
                # velocity_limit_sim=7.0,
                # stiffness=10.0,
                # damping=0.5,
                # armature=0.001,
                # viscous_friction=0.05,
                effort_limit_sim=1.200,
                velocity_limit_sim=30.0,
                stiffness=1200.0,
                damping=120.0,
                armature=0.005,
                viscous_friction=0.01,
                # effort_limit_sim=1.200,
                # velocity_limit_sim=30,#6.28,
                # stiffness=100.0,
                # damping=0.1,
            ),
        },
    )

def make_tshape_right_robot_cfg(base_dir: str) -> ArticulationCfg:

    # =========================================================
    # [수정 1] 오른팔 회전 (월드 축 xyz 기준)
    # Z축 -270도(=90도) 회전 상태에서 X축 150도 회전
    # =========================================================
    r_angles_deg = (-90.0, 135.0)  # (Z 먼저, X 다음)
    r_rot = Rotation.from_euler("zy", r_angles_deg, degrees=True)  # 소문자=월드 고정축
    r_local = Rotation.from_euler("Z", 135.0, degrees=True)  # 대문자=로컬(바디)축
    r_rot = (
        r_rot * r_local
    )  # "월드 회전 후 -> 로컬 회전" (오른쪽이 먼저 적용되는게 아니라, 이 형태가 의도에
    r_quat_scipy = r_rot.as_quat()
    right_rot_quat = (
        r_quat_scipy[3],
        r_quat_scipy[0],
        r_quat_scipy[1],
        r_quat_scipy[2],
    )

    return ArticulationCfg(
        prim_path="/World/envs/env_.*/RightRobot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                base_dir,
                "assets/robot_urdf_fix/ra_ur3_allegro_isaac_/ra_ur3_allegro_isaac_.usd",
            ),
                activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.177, 0.0, 1.49),
            rot=right_rot_quat,
            joint_pos={
                "ra_shoulder_pan_joint": math.radians(27.33),
                "ra_shoulder_lift_joint": math.radians(-43.98),
                "ra_elbow_joint": math.radians(98.18),
                "ra_wrist_1_joint": math.radians(-232.88),
                "ra_wrist_2_joint": math.radians(-71.98),
                "ra_wrist_3_joint": math.radians(223.84),
                "ra_joint_12_0": 1.396,
                # "ra_shoulder_pan_joint": math.radians(26.73),
                # "ra_shoulder_lift_joint": math.radians(-71.29),
                # "ra_elbow_joint": math.radians(140.68),
                # "ra_wrist_1_joint": math.radians(-232.63),
                # "ra_wrist_2_joint": math.radians(-45.48),
                # "ra_wrist_3_joint": math.radians(223.71),
                # "ra_joint_12_0": 1.396,
            },
        ),
        actuators={
            "ur3_arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "ra_shoulder_pan_joint",
                    "ra_shoulder_lift_joint",
                    "ra_elbow_joint",
                    "ra_wrist_1_joint",
                    "ra_wrist_2_joint",
                    "ra_wrist_3_joint",
                ],
                effort_limit_sim=120.0,
                velocity_limit_sim=2.0,
                stiffness=1100.0,
                damping=300.0,
                armature=0.02,
                viscous_friction=0.02,
            ),
            "allegro_hand": ImplicitActuatorCfg(
                joint_names_expr=[
                    "ra_joint_0_0",
                    "ra_joint_12_0",
                    "ra_joint_4_0",
                    "ra_joint_8_0",
                    "ra_joint_1_0",
                    "ra_joint_13_0",
                    "ra_joint_5_0",
                    "ra_joint_9_0",
                    "ra_joint_2_0",
                    "ra_joint_14_0",
                    "ra_joint_6_0",
                    "ra_joint_10_0",
                    "ra_joint_3_0",
                    "ra_joint_15_0",
                    "ra_joint_7_0",
                    "ra_joint_11_0",
                ],
                effort_limit_sim=1.200,
                velocity_limit_sim=30.0,
                stiffness=1200.0,
                damping=120.0,
                armature=0.005,
                viscous_friction=0.01,
            ),
        },
    )


def make_tshape_left_robot_cfg(base_dir: str) -> ArticulationCfg:

    l_angles_deg = (90.0, -135.0)
    l_rot = Rotation.from_euler("zy", l_angles_deg, degrees=True)
    l_local = Rotation.from_euler("Z", -135.0, degrees=True)
    l_rot = l_rot * l_local
    l_quat_scipy = l_rot.as_quat()
    left_rot_quat = (l_quat_scipy[3], l_quat_scipy[0], l_quat_scipy[1], l_quat_scipy[2])

    return ArticulationCfg(
        prim_path="/World/envs/env_.*/LeftRobot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                base_dir,
                "assets/robot_urdf_fix/la_ur3e_allegro_isaac_/la_ur3e_allegro_isaac_.usd",
            ),
                activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.177, 0.0, 1.49),
            rot=left_rot_quat,
            joint_pos={
                "la_shoulder_pan_joint": math.radians(-18.75),
                "la_shoulder_lift_joint": math.radians(-138.64),
                "la_elbow_joint": math.radians(-99.59),
                "la_wrist_1_joint": math.radians(57.97),
                "la_wrist_2_joint": math.radians(64.88),
                "la_wrist_3_joint": math.radians(137.67),
                "la_joint_12_0": 1.396,
                # "la_shoulder_pan_joint": math.radians(-28.54),
                # "la_shoulder_lift_joint": math.radians(-123.63),
                # "la_elbow_joint": math.radians(-137.41),
                # "la_wrist_1_joint": math.radians(73.44),
                # "la_wrist_2_joint": math.radians(81.66),
                # "la_wrist_3_joint": math.radians(138.93),
                # "la_joint_12_0": 1.396,
            },
        ),
        actuators={
            "ur3_arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "la_shoulder_pan_joint",
                    "la_shoulder_lift_joint",
                    "la_elbow_joint",
                    "la_wrist_1_joint",
                    "la_wrist_2_joint",
                    "la_wrist_3_joint",
                ],
                effort_limit_sim=120.0,
                velocity_limit_sim=2.0,
                stiffness=1100.0,
                damping=300.0,
                armature=0.02,
                viscous_friction=0.02,
            ),
            "allegro_hand": ImplicitActuatorCfg(
                joint_names_expr=[
                    "la_joint_0_0",
                    "la_joint_12_0",
                    "la_joint_4_0",
                    "la_joint_8_0",
                    "la_joint_1_0",
                    "la_joint_13_0",
                    "la_joint_5_0",
                    "la_joint_9_0",
                    "la_joint_2_0",
                    "la_joint_14_0",
                    "la_joint_6_0",
                    "la_joint_10_0",
                    "la_joint_3_0",
                    "la_joint_15_0",
                    "la_joint_7_0",
                    "la_joint_11_0",
                ],
                effort_limit_sim=1.200,
                velocity_limit_sim=30.0,
                stiffness=1200.0,
                damping=120.0,
                armature=0.005,
                viscous_friction=0.01,
            ),
        },
    )


def make_tshape_cfg(base_dir: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/tshape",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                base_dir,
                "assets/tshape.usd",
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.147, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )



def make_colorcube_table_cfg(base_dir: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                base_dir,
                "assets/motion_desk.usd",
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.4, -0.07),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

def make_handover_table_cfg(base_dir: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                base_dir,
                "assets/motion_desk.usd",
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.03),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

def make_table_cfg(base_dir: str) -> RigidObjectCfg:
    return make_colorcube_table_cfg(base_dir)

def make_task_table_cfg(base_dir: str, task_name: str) -> RigidObjectCfg:
    if str(task_name or "").strip().lower() in {"handover", "handover_tshape"}:
        return make_handover_table_cfg(base_dir)
    return make_colorcube_table_cfg(base_dir)

def make_btable_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(1.8, 0.90, 0.93),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,            # 고정물체로
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset =0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.0,
                dynamic_friction=0.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.0), 
                metallic=0.1
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.55, 0.465), 
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

def make_table_silver_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/table_1",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.74, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # 고정물체로
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.9, 0.9),
                metallic=0.6,
                roughness=0.3,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.375),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_table_black_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/table_2",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.57, 0.008),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # 고정물체로
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.03, 0.03, 0.03),
                metallic=0.1,
                roughness=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.435, 0.754),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_front_wall_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/front_wall",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 0.04, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.278, 0.282, 0.298),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 1.0, 0.75),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_right_wall_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/right_wall",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 2.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.278, 0.282, 0.298),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.0, 0.0, 0.75),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_left_wall_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/left_wall",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 2.0, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.278, 0.282, 0.298),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-1.0, 0.0, 0.75),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_back_wall_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/back_wall",
        spawn=sim_utils.CuboidCfg(
            size=(3.5, 0.04, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.278, 0.282, 0.298),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.75, -1.0, 0.75),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# -------- 공통 시뮬레이션/씬 + 로봇/테이블을 포함한 Base EnvCfg --------
@configclass
class BaseDualArmTableEnvCfg(DirectRLEnvCfg):
    """모든 태스크가 공유하는 베이스 설정.
    - 로봇, 테이블, 물리/씬 기본값
    - 태스크별로 episode_length, action/observation 크기만 오버라이드
    """

    # env
    decimation: int = 2
    episode_length: int = 0
    episode_length_s: float = 0.0
    action_space: int = 0
    observation_space: int = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,  # decimation 초기값, __post_init__에서 동기화
        physics_material=RigidBodyMaterialCfg(
            static_friction=2.0, dynamic_friction=2.0
        ),
        physx=PhysxCfg(bounce_threshold_velocity=0.2),
    )

    BASE_DIR: str = ""
    right_robot_cfg: ArticulationCfg | None = None
    left_robot_cfg: ArticulationCfg | None = None
    table_cfg: RigidObjectCfg | None = None
    table_silver_cfg: RigidObjectCfg = make_table_silver_cfg()
    table_black_cfg: RigidObjectCfg = make_table_black_cfg()

    front_wall_cfg: RigidObjectCfg | None = None
    right_wall_cfg: RigidObjectCfg | None = None
    left_wall_cfg: RigidObjectCfg | None = None
    back_wall_cfg: RigidObjectCfg | None = None

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=50, env_spacing=3, replicate_physics=True
    )

    def __post_init__(self):
        self.right_robot_cfg = make_right_robot_cfg(self.BASE_DIR)
        self.left_robot_cfg = make_left_robot_cfg(self.BASE_DIR)
        self.table_cfg = make_task_table_cfg(self.BASE_DIR, getattr(self, "task_name", ""))

        self.episode_length = int(self.episode_length)
        self.episode_length_s = float(self.episode_length) * (1.0 / 60.0)
        self.sim.render_interval = self.decimation  # 시뮬 설정과 동기화

        self.action_space = int(self.action_space)
        self.observation_space = int(self.observation_space)


@configclass
class BaseTshapeEnvCfg(DirectRLEnvCfg):
    """모든 태스크가 공유하는 베이스 설정.
    - 로봇, 테이블, 물리/씬 기본값
    - 태스크별로 episode_length, action/observation 크기만 오버라이드
    """

    # env
    decimation: int = 2
    episode_length: int = 0
    episode_length_s: float = 0.0
    action_space: int = 0
    observation_space: int = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=2,  # decimation 초기값, __post_init__에서 동기화
        physics_material=RigidBodyMaterialCfg(
            static_friction=2.0, dynamic_friction=2.0
        ),
        physx=PhysxCfg(bounce_threshold_velocity=0.2),
    )

    BASE_DIR: str = ""
    right_robot_cfg: ArticulationCfg | None = None
    left_robot_cfg: ArticulationCfg | None = None
    table_cfg: RigidObjectCfg | None = None
    tshape_cfg: RigidObjectCfg | None = None

    front_wall_cfg: RigidObjectCfg | None = None
    right_wall_cfg: RigidObjectCfg | None = None
    left_wall_cfg: RigidObjectCfg | None = None
    back_wall_cfg: RigidObjectCfg | None = None

    articulations_cfg: dict[str, ArticulationCfg] | None = None
    rigid_objects_cfg: dict[str, RigidObjectCfg] | None = None
    markers_cfg: dict[str, VisualizationMarkersCfg] | None = None

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=50, env_spacing=3, replicate_physics=True
    )

    def __post_init__(self):
        # 0) dict 초기화
        self.articulations_cfg = {}
        self.rigid_objects_cfg = {}
        self.markers_cfg = {}

        # 1) 로봇 cfg 생성
        self.right_robot_cfg = make_tshape_right_robot_cfg(self.BASE_DIR)
        self.left_robot_cfg = make_tshape_left_robot_cfg(self.BASE_DIR)
        self.table_cfg = make_task_table_cfg(self.BASE_DIR, getattr(self, "task_name", ""))
        # self.table_cfg = make_btable_cfg()
        self.tshape_cfg = make_tshape_cfg(self.BASE_DIR)

        # 2) dict에 등록
        self.articulations_cfg["right_robot"] = self.right_robot_cfg
        self.articulations_cfg["left_robot"] = self.left_robot_cfg

        # table / walls 등록
        self.rigid_objects_cfg["table"] = self.table_cfg
        self.rigid_objects_cfg["tshape"] = self.tshape_cfg
        if self.front_wall_cfg:
            self.rigid_objects_cfg["front_wall"] = self.front_wall_cfg
        if self.right_wall_cfg:
            self.rigid_objects_cfg["right_wall"] = self.right_wall_cfg
        if self.left_wall_cfg:
            self.rigid_objects_cfg["left_wall"] = self.left_wall_cfg
        if self.back_wall_cfg:
            self.rigid_objects_cfg["back_wall"] = self.back_wall_cfg

        # 3) 나머지 공통 세팅
        self.episode_length = int(self.episode_length)
        self.episode_length_s = float(self.episode_length) * (1.0 / 60.0)
        self.sim.render_interval = self.decimation

        self.action_space = int(self.action_space)
        self.observation_space = int(self.observation_space)



# -------- [신규] 통합 CommonVecEnv Class --------
class CommonVecEnv:
    def __init__(
        self,
        env: DirectRLEnv,
        device: torch.device = torch.device("cuda:0"),
        clip_observation=5.0,
        clip_actions=1.0,
    ):
        self.env = env
        self.device = torch.device(device) if isinstance(device, str) else device
        self.clip_obs = clip_observation
        self.clip_actions = clip_actions
        self.vec_debug_prints = bool(getattr(env.cfg, "vec_debug_prints", False))
        self._step_dbg_counter = 0
        self._obs_shape_checked = not self.vec_debug_prints
        self._critic_shape_checked = not self.vec_debug_prints

        self.num_envs = env.num_envs
        self.minimal_step_info = bool(getattr(env.cfg, "minimal_step_info", True))
        self.num_agents = 2  # Dual Arm: 2 Agents 고정
        self.num_hand_obs = int(getattr(env.cfg, "num_hand_obs", 0))
        self._raw_num_observations = int(env.cfg.observation_space)
        self._raw_num_share_observations = int(
            getattr(env.cfg, "share_observation_space", self._raw_num_observations)
        )
        self.num_observations = self._per_agent_obs_dim(
            self._raw_num_observations, "observation_space"
        )
        self.num_share_observations = self._per_agent_obs_dim(
            self._raw_num_share_observations, "share_observation_space"
        )
        self.num_actions = env.cfg.action_space
        self.use_vision = bool(getattr(env.cfg, "use_vision", False))
        self.vision_obs_dim = int(getattr(env.cfg, "vision_obs_dim", 0)) if self.use_vision else 0
        self.strict_vision_only_policy = bool(getattr(env.cfg, "strict_vision_only_policy", False))
        self._vision_shape_checked = False

        # Spaces 정의
        self.obs_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observations,))
            for _ in range(self.num_agents)
        ]
        self.share_observation_space = [
            spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_share_observations,))
            for _ in range(self.num_agents)
        ]
        self.act_space = [
            spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
            for _ in range(self.num_agents)
        ]
        if self.vec_debug_prints:
            print(
                f"[CommonVecEnv] cfg={type(env.cfg).__name__} num_envs={self.num_envs} "
                f"num_agents={self.num_agents} num_hand_obs={self.num_hand_obs} "
                f"obs_raw={self._raw_num_observations} obs_per_agent={self.num_observations} "
                f"share_raw={self._raw_num_share_observations} share_per_agent={self.num_share_observations} "
                f"action_dim={self.num_actions} use_vision={self.use_vision} vision_obs_dim={self.vision_obs_dim}"
            )

    def _per_agent_obs_dim(self, total_dim: int, field_name: str) -> int:
        total_dim = int(total_dim)
        if self.num_hand_obs <= 0:
            return total_dim
        split_width = 2 * self.num_hand_obs
        if total_dim < split_width:
            raise ValueError(
                f"{field_name}({total_dim}) is smaller than 2*num_hand_obs({split_width})."
            )
        return self.num_hand_obs + (total_dim - split_width)

    def _split_agent_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if int(obs.dim()) != 2:
            raise ValueError(f"obs for split should be 2D [num_envs, dim], got {tuple(obs.shape)}")
        obs = torch.clamp(obs, -self.clip_obs, self.clip_obs).to(self.device)
        if self.num_hand_obs > 0:
            split_width = 2 * self.num_hand_obs
            if int(obs.shape[-1]) < split_width:
                raise ValueError(
                    f"obs dim({int(obs.shape[-1])}) is smaller than 2*num_hand_obs({split_width})."
                )
            shared = obs[:, split_width:]
            out = obs.new_empty((obs.shape[0], self.num_agents, self.num_hand_obs + shared.shape[1]))
            out[:, 0, : self.num_hand_obs] = obs[:, : self.num_hand_obs]
            out[:, 1, : self.num_hand_obs] = obs[:, self.num_hand_obs:split_width]
            out[:, :, self.num_hand_obs :] = shared.unsqueeze(1)
            return out
        return obs.unsqueeze(1).expand(-1, self.num_agents, -1)

    def _normalize_optional_agent_obs(self, obs_dict: dict, key: str) -> None:
        if key not in obs_dict:
            return
        x = obs_dict[key]
        if torch.is_tensor(x):
            x = x.to(device=self.device, dtype=torch.float32)
        else:
            x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        if x.dim() == 2:
            if int(x.shape[0]) != int(self.num_envs):
                raise RuntimeError(
                    f"[CommonVecEnv] {key} env dim mismatch: got {int(x.shape[0])}, expected {int(self.num_envs)}"
                )
            # Dual-observation distill feeds checkpoint-native env-wide teacher state
            # that is broadcast to both agents. It must be checked BEFORE the student
            # per-agent split: the teacher dim can exceed 2*num_hand_obs (e.g. 38/44)
            # yet must NOT be split into the student per-agent contract.
            if key in ("teacher_policy", "teacher_critic"):
                x = x.unsqueeze(1).expand(-1, self.num_agents, -1)
            # Raw env-wide observation -> split into per-agent view.
            elif int(x.shape[-1]) >= int(2 * self.num_hand_obs):
                x = self._split_agent_obs(x)
            elif int(x.shape[-1]) in (int(self.num_observations), int(self.num_share_observations)):
                x = x.unsqueeze(1).expand(-1, self.num_agents, -1)
            else:
                raise RuntimeError(
                    f"[CommonVecEnv] unsupported {key} feature dim {int(x.shape[-1])} "
                    f"(num_obs={int(self.num_observations)}, num_share_obs={int(self.num_share_observations)})."
                )
        elif x.dim() == 3:
            if int(x.shape[0]) != int(self.num_envs):
                raise RuntimeError(
                    f"[CommonVecEnv] {key} env dim mismatch: got {int(x.shape[0])}, expected {int(self.num_envs)}"
                )
            if int(x.shape[1]) == 1:
                x = x.expand(-1, self.num_agents, -1)
            if int(x.shape[1]) != int(self.num_agents):
                raise RuntimeError(
                    f"[CommonVecEnv] {key} agent dim mismatch: got {int(x.shape[1])}, expected {int(self.num_agents)}"
                )
        else:
            raise RuntimeError(f"[CommonVecEnv] {key} must be rank-2/3, got {tuple(x.shape)}")
        obs_dict[key] = x

    def _assert_no_privileged_obs_dict(self, obs_dict: dict):
        if not self.strict_vision_only_policy:
            return
        leaked = []
        for key in ("obj_pos", "cube_pos", "cube_quat", "cubes_pos", "boxes_pos", "pad_pos", "privileged_obs"):
            if key in obs_dict:
                leaked.append(key)
        if leaked:
            raise RuntimeError(
                f"[CommonVecEnv] privileged keys leaked into rollout path while strict_vision_only_policy=True: {leaked}"
            )

    def _validate_vision_obs(self, obs_dict: dict) -> dict:
        if not self.use_vision:
            return obs_dict
        raw_student_mode = bool(getattr(getattr(self.env, "cfg", None), "vision_student_raw_mode", False))
        if raw_student_mode:
            img_left = obs_dict.get("img_left", None)
            img_right = obs_dict.get("img_right", None)
            if (img_left is not None) or (img_right is not None):
                if img_left is None or img_right is None:
                    raise RuntimeError(
                        "[CommonVecEnv] stereo raw student mode requires both obs_dict['img_left'] and obs_dict['img_right']."
                    )
                if torch.is_tensor(img_left):
                    img_left = img_left.to(device=self.device, dtype=torch.float32)
                else:
                    img_left = torch.as_tensor(img_left, device=self.device, dtype=torch.float32)
                if torch.is_tensor(img_right):
                    img_right = img_right.to(device=self.device, dtype=torch.float32)
                else:
                    img_right = torch.as_tensor(img_right, device=self.device, dtype=torch.float32)
                if img_left.dim() != 4 or img_right.dim() != 4:
                    raise RuntimeError(
                        f"[CommonVecEnv] img_left/img_right must be rank-4 [env,3,H,W], got left={tuple(img_left.shape)} right={tuple(img_right.shape)}"
                    )
                if int(img_left.shape[0]) != int(self.num_envs) or int(img_right.shape[0]) != int(self.num_envs):
                    raise RuntimeError(
                        f"[CommonVecEnv] stereo env dim mismatch: left={int(img_left.shape[0])} right={int(img_right.shape[0])} expected={int(self.num_envs)}"
                    )
                if int(img_left.shape[1]) != 3 or int(img_right.shape[1]) != 3:
                    raise RuntimeError(
                        f"[CommonVecEnv] img_left/img_right channel mismatch: left={int(img_left.shape[1])} right={int(img_right.shape[1])}, expected 3/3"
                    )
                obs_dict["img_left"] = img_left
                obs_dict["img_right"] = img_right
                if "mask_left" in obs_dict:
                    mask_left = obs_dict["mask_left"]
                    obs_dict["mask_left"] = (
                        mask_left.to(device=self.device)
                        if torch.is_tensor(mask_left)
                        else torch.as_tensor(mask_left, device=self.device)
                    )
                if "mask_right" in obs_dict:
                    mask_right = obs_dict["mask_right"]
                    obs_dict["mask_right"] = (
                        mask_right.to(device=self.device)
                        if torch.is_tensor(mask_right)
                        else torch.as_tensor(mask_right, device=self.device)
                    )
                obs_dict["vision_shared"] = bool(obs_dict.get("vision_shared", True))
                if not self._vision_shape_checked:
                    print(
                        f"[CommonVecEnv][vision][raw-stereo] left={tuple(img_left.shape)} right={tuple(img_right.shape)} "
                        f"shared={bool(obs_dict['vision_shared'])}"
                    )
                    self._vision_shape_checked = True
                return obs_dict
            vision_rgb = obs_dict.get("vision_rgb", None)
            if vision_rgb is None:
                raise RuntimeError(
                    "[CommonVecEnv] raw student vision mode requires obs_dict['vision_rgb']."
                )
            if torch.is_tensor(vision_rgb):
                vision_rgb = vision_rgb.to(device=self.device, dtype=torch.float32)
            else:
                vision_rgb = torch.as_tensor(vision_rgb, device=self.device, dtype=torch.float32)
            if vision_rgb.dim() != 4:
                raise RuntimeError(
                    f"[CommonVecEnv] vision_rgb must be rank-4 [env,3,H,W], got {tuple(vision_rgb.shape)}"
                )
            if int(vision_rgb.shape[0]) != int(self.num_envs):
                raise RuntimeError(
                    f"[CommonVecEnv] vision_rgb env dim mismatch: got {int(vision_rgb.shape[0])}, expected {int(self.num_envs)}"
                )
            if int(vision_rgb.shape[1]) != 3:
                raise RuntimeError(
                    f"[CommonVecEnv] vision_rgb channel mismatch: got {int(vision_rgb.shape[1])}, expected 3"
                )

            use_depth = bool(getattr(getattr(self.env, "cfg", None), "vision_use_depth", True))
            vision_depth = obs_dict.get("vision_depth", None)
            if use_depth and vision_depth is None:
                raise RuntimeError(
                    "[CommonVecEnv] raw student vision mode requires obs_dict['vision_depth'] when vision_use_depth=True."
                )
            if vision_depth is not None:
                if torch.is_tensor(vision_depth):
                    vision_depth = vision_depth.to(device=self.device, dtype=torch.float32)
                else:
                    vision_depth = torch.as_tensor(vision_depth, device=self.device, dtype=torch.float32)
                if vision_depth.dim() != 4:
                    raise RuntimeError(
                        f"[CommonVecEnv] vision_depth must be rank-4 [env,1,H,W], got {tuple(vision_depth.shape)}"
                    )
                if int(vision_depth.shape[0]) != int(self.num_envs):
                    raise RuntimeError(
                        f"[CommonVecEnv] vision_depth env dim mismatch: got {int(vision_depth.shape[0])}, expected {int(self.num_envs)}"
                    )
                if int(vision_depth.shape[1]) != 1:
                    raise RuntimeError(
                        f"[CommonVecEnv] vision_depth channel mismatch: got {int(vision_depth.shape[1])}, expected 1"
                    )
                obs_dict["vision_depth"] = vision_depth

            obs_dict["vision_rgb"] = vision_rgb
            obs_dict["vision_shared"] = bool(obs_dict.get("vision_shared", True))
            if not self._vision_shape_checked:
                print(
                    f"[CommonVecEnv][vision][raw] rgb={tuple(vision_rgb.shape)} "
                    f"depth={None if vision_depth is None else tuple(vision_depth.shape)} "
                    f"shared={bool(obs_dict['vision_shared'])}"
                )
                self._vision_shape_checked = True
            return obs_dict
        if "vision_obs" not in obs_dict:
            raise RuntimeError(
                "[CommonVecEnv] use_vision=True but obs_dict['vision_obs'] is missing. "
                "Check camera sensor -> encoder -> policy pipeline."
            )
        vision_obs = obs_dict["vision_obs"]
        if torch.is_tensor(vision_obs):
            vision_obs = vision_obs.to(device=self.device, dtype=torch.float32)
        else:
            vision_obs = torch.as_tensor(vision_obs, device=self.device, dtype=torch.float32)
        if vision_obs.dim() == 2:
            vision_obs = vision_obs.unsqueeze(1).expand(-1, self.num_agents, -1)
        if vision_obs.dim() != 3:
            raise RuntimeError(f"[CommonVecEnv] vision_obs must be rank-3 [env,agent,dim], got {tuple(vision_obs.shape)}")
        if int(vision_obs.shape[0]) != int(self.num_envs):
            raise RuntimeError(
                f"[CommonVecEnv] vision_obs env dim mismatch: got {int(vision_obs.shape[0])}, expected {int(self.num_envs)}"
            )
        if int(vision_obs.shape[1]) == 1:
            vision_obs = vision_obs.expand(-1, self.num_agents, -1)
        if int(vision_obs.shape[1]) != int(self.num_agents):
            raise RuntimeError(
                f"[CommonVecEnv] vision_obs agent dim mismatch: got {int(vision_obs.shape[1])}, expected {int(self.num_agents)}"
            )
        if self.vision_obs_dim > 0 and int(vision_obs.shape[-1]) != int(self.vision_obs_dim):
            raise RuntimeError(
                f"[CommonVecEnv] vision_obs feature dim mismatch: got {int(vision_obs.shape[-1])}, expected {int(self.vision_obs_dim)}"
            )
        obs_dict["vision_obs"] = vision_obs
        if "critic_vision_obs" not in obs_dict:
            obs_dict["critic_vision_obs"] = vision_obs
        if not self._vision_shape_checked:
            print(f"[CommonVecEnv] vision_obs={tuple(vision_obs.shape)}")
            self._vision_shape_checked = True
        return obs_dict

    def step(self, actions: torch.Tensor, step: int = 0):
        # actions shape: [2, num_envs, action_dim] (Agent, Env, Action)
        if int(actions.dim()) != 3:
            raise ValueError(
                f"actions should be 3D [agents, num_envs, act_dim], got {tuple(actions.shape)}"
            )
        if int(actions.shape[0]) != int(self.num_agents):
            raise ValueError(f"Expected {int(self.num_agents)} agents, got {int(actions.shape[0])}")
        if int(actions.shape[1]) != int(self.num_envs):
            raise ValueError(
                f"Expected actions dim1=num_envs({int(self.num_envs)}), got {int(actions.shape[1])}"
            )
        if int(actions.shape[2]) != int(self.num_actions):
            raise ValueError(
                f"Expected action dim {int(self.num_actions)}, got {int(actions.shape[2])}"
            )

        # Action 병합: Agent 0과 Agent 1의 액션을 수평으로 연결하여 [num_envs, 2*action_dim] 형태로 변환
        a_hand_actions = actions.permute(1, 0, 2).reshape(self.num_envs, self.num_agents * self.num_actions)
        if self.vec_debug_prints and self._step_dbg_counter < 2:
            print(
                f"[CommonVecEnv] step={int(self._step_dbg_counter)} action_concat={tuple(a_hand_actions.shape)} "
                f"action_space={tuple(self.act_space[0].shape)}"
            )

        # Environment Step
        obs_dict, rewards, terminated, truncated, extras = self.env.step(a_hand_actions)
        self._assert_no_privileged_obs_dict(obs_dict)
        obs_dict = self._validate_vision_obs(obs_dict)
        self._normalize_optional_agent_obs(obs_dict, "student_policy")
        self._normalize_optional_agent_obs(obs_dict, "teacher_policy")
        self._normalize_optional_agent_obs(obs_dict, "teacher_critic")

        # Observation 처리
        obs = obs_dict["policy"]  # shape: [num_envs, raw_obs_dim]
        if int(obs.dim()) != 2:
            raise ValueError(f"policy obs should be 2D [num_envs, dim], got {tuple(obs.shape)}")
        if int(obs.shape[0]) != int(self.num_envs):
            raise ValueError(
                f"policy obs dim0 should be num_envs({int(self.num_envs)}), got {int(obs.shape[0])}"
            )
        if int(obs.shape[1]) != int(self._raw_num_observations):
            raise RuntimeError(
                f"policy obs dim1 mismatch: got {int(obs.shape[1])}, cfg.observation_space={int(self._raw_num_observations)}"
            )
        obs_all = self._split_agent_obs(obs)
        if int(obs_all.shape[-1]) != int(self.num_observations):
            raise RuntimeError(
                f"policy obs split dim mismatch: got {int(obs_all.shape[-1])}, "
                f"expected {int(self.num_observations)}"
            )

        critic_obs = obs_dict.get("critic", obs)
        if int(critic_obs.dim()) != 2:
            raise ValueError(f"critic obs should be 2D [num_envs, dim], got {tuple(critic_obs.shape)}")
        if int(critic_obs.shape[0]) != int(self.num_envs):
            raise ValueError(
                f"critic obs dim0 should be num_envs({int(self.num_envs)}), got {int(critic_obs.shape[0])}"
            )
        if int(critic_obs.shape[1]) != int(self._raw_num_share_observations):
            raise RuntimeError(
                f"critic obs dim1 mismatch: got {int(critic_obs.shape[1])}, cfg.share_observation_space={int(self._raw_num_share_observations)}"
            )
        state_all = self._split_agent_obs(critic_obs)
        if int(state_all.shape[-1]) != int(self.num_share_observations):
            raise RuntimeError(
                f"critic obs split dim mismatch: got {int(state_all.shape[-1])}, "
                f"expected {int(self.num_share_observations)}"
            )
        if self.vec_debug_prints and not self._obs_shape_checked:
            print(
                f"[CommonVecEnv] policy(obs)={tuple(obs.shape)} -> {tuple(obs_all.shape)} "
                f"critic(obs)={tuple(critic_obs.shape)} -> {tuple(state_all.shape)}"
            )
            self._obs_shape_checked = True
        if self.vec_debug_prints and not self._critic_shape_checked:
            print(
                f"[CommonVecEnv] reward_shape={tuple(rewards.shape)} terminated={tuple(terminated.shape)} "
                f"truncated={tuple(truncated.shape)} extras={list(extras.keys())}"
            )
            self._critic_shape_checked = True

        # Rewards 처리: [num_envs, 1] -> [num_envs, 2, 1]
        rewards = rewards.unsqueeze(-1).to(self.device)
        reward_all = rewards.unsqueeze(1).expand(-1, self.num_agents, -1)

        # Dones 처리: [num_envs] -> [num_envs, 2]
        done = (terminated | truncated).to(self.device)
        done_all = done.unsqueeze(-1).expand(-1, self.num_agents)
        self._step_dbg_counter += 1

        infos = []
        has_success_info = "success" in extras
        has_done_success_info = "done_success" in extras
        has_done_fallen_info = "done_fallen" in extras
        has_done_timeout_info = "done_timeout" in extras
        has_terminal_obs_info = "terminal_observation" in extras
        has_stage_level_info = "stage_level" in extras
        reward_info_keys = (
            "r_traj",
            "r_grasp",
            "r_handover",
            "r_transfer",
            "r_place",
            "r_release",
            "r_regress",
            "r_both",
            "r_success",
            "r_wrist34_penalty",
            "wrist34_pair_dist",
            "wrist34_pair_min_dist",
        )
        has_reward_info = {k: (k in extras) for k in reward_info_keys}
        flag_info_keys = (
            "in_success_region",
            "target_held",
            "success_region_latch",
            "hands_home",
        )
        has_flag_info = {k: (k in extras) for k in flag_info_keys}
        if self.minimal_step_info:
            infos = [None] * self.num_envs
            done_cpu = done.detach().to("cpu")
            done_ids = torch.nonzero(done_cpu, as_tuple=False).flatten().tolist()
            for i in done_ids:
                info = {}
                if has_success_info:
                    success_i = bool(extras["success"][i].item())
                    if has_done_success_info:
                        success_i = success_i or bool(extras["done_success"][i].item())
                    info["success"] = success_i
                if has_done_success_info:
                    info["done_success"] = bool(extras["done_success"][i].item())
                if has_done_fallen_info:
                    info["done_fallen"] = bool(extras["done_fallen"][i].item())
                if has_done_timeout_info:
                    info["done_timeout"] = bool(extras["done_timeout"][i].item())
                if has_terminal_obs_info:
                    term_obs_i = extras["terminal_observation"][i]
                    if torch.is_tensor(term_obs_i):
                        term_obs_i = term_obs_i.detach().cpu().numpy()
                    info["terminal_observation"] = np.asarray(term_obs_i, dtype=np.float32)
                if has_stage_level_info:
                    info["stage_level"] = int(extras["stage_level"][i].item())
                for k, enabled in has_reward_info.items():
                    if not enabled:
                        continue
                    v = extras[k][i]
                    if torch.is_tensor(v):
                        v = v.item()
                    info[k] = float(v)
                for k, enabled in has_flag_info.items():
                    if not enabled:
                        continue
                    v = extras[k][i]
                    if torch.is_tensor(v):
                        v = v.item()
                    info[k] = bool(v)
                infos[i] = info
            return obs_all, obs_all, state_all, reward_all, done_all, infos, obs_dict
        for i in range(self.num_envs):
            info = {}

            if has_success_info:
                success_i = bool(extras["success"][i].item())
                if has_done_success_info:
                    success_i = success_i or bool(extras["done_success"][i].item())
                info["success"] = success_i
            if has_done_success_info:
                info["done_success"] = bool(extras["done_success"][i].item())
            if has_done_fallen_info:
                info["done_fallen"] = bool(extras["done_fallen"][i].item())
            if has_done_timeout_info:
                info["done_timeout"] = bool(extras["done_timeout"][i].item())
            if has_terminal_obs_info:
                term_obs_i = extras["terminal_observation"][i]
                if torch.is_tensor(term_obs_i):
                    term_obs_i = term_obs_i.detach().cpu().numpy()
                info["terminal_observation"] = np.asarray(term_obs_i, dtype=np.float32)
            if has_stage_level_info:
                info["stage_level"] = int(extras["stage_level"][i].item())
            for k, enabled in has_reward_info.items():
                if not enabled:
                    continue
                v = extras[k][i]
                if torch.is_tensor(v):
                    v = v.item()
                info[k] = float(v)
            # 필요하다면 time_out 정보 등도 추가 가능

            for k, enabled in has_flag_info.items():
                if not enabled:
                    continue
                v = extras[k][i]
                if torch.is_tensor(v):
                    v = v.item()
                info[k] = bool(v)
            infos.append(info)

        return obs_all, obs_all, state_all, reward_all, done_all, infos, obs_dict

    def reset(self):
        # Environment Reset
        # DirectRLEnv.reset()은 (observations, extras)를 반환하지만,
        # self.env.observations에 접근하여 dict를 가져오는 방식이 안전합니다.
        if self.vec_debug_prints:
            print("[CommonVecEnv][reset] env.reset start", flush=True)
        obs_dict, _ = self.env.reset()
        if self.vec_debug_prints:
            print(f"[CommonVecEnv][reset] env.reset done keys={list(obs_dict.keys())}", flush=True)
        self._assert_no_privileged_obs_dict(obs_dict)
        if self.vec_debug_prints:
            print("[CommonVecEnv][reset] validate_vision start", flush=True)
        obs_dict = self._validate_vision_obs(obs_dict)
        if self.vec_debug_prints:
            print("[CommonVecEnv][reset] validate_vision done", flush=True)
        self._normalize_optional_agent_obs(obs_dict, "student_policy")
        self._normalize_optional_agent_obs(obs_dict, "teacher_policy")
        self._normalize_optional_agent_obs(obs_dict, "teacher_critic")
        if self.vec_debug_prints:
            print("[CommonVecEnv][reset] optional agent obs normalized", flush=True)

        obs = obs_dict["policy"]
        if self.vec_debug_prints:
            print(f"[CommonVecEnv][reset] split policy start shape={tuple(obs.shape)}", flush=True)
        obs_all = self._split_agent_obs(obs)
        if self.vec_debug_prints:
            print(f"[CommonVecEnv][reset] split policy done shape={tuple(obs_all.shape)}", flush=True)
        if int(obs_all.shape[-1]) != int(self.num_observations):
            raise RuntimeError(
                f"policy obs split dim mismatch on reset: got {int(obs_all.shape[-1])}, "
                f"expected {int(self.num_observations)}"
            )

        critic_obs = obs_dict.get("critic", obs)
        if self.vec_debug_prints:
            print(f"[CommonVecEnv][reset] split critic start shape={tuple(critic_obs.shape)}", flush=True)
        state_all = self._split_agent_obs(critic_obs)
        if self.vec_debug_prints:
            print(f"[CommonVecEnv][reset] split critic done shape={tuple(state_all.shape)}", flush=True)
        if int(state_all.shape[-1]) != int(self.num_share_observations):
            raise RuntimeError(
                f"critic obs split dim mismatch on reset: got {int(state_all.shape[-1])}, "
                f"expected {int(self.num_share_observations)}"
            )

        # obs_dict도 함께 반환 (기존 코드 호환성 유지)
        return obs_all, state_all, obs_dict

    def get_number_of_agents(self):
        return self.num_agents

    def get_env_info(self):
        return {
            "state_shape": self.share_observation_space,
            "obs_shape": self.num_observations,
            "n_actions": self.num_actions,
            "n_agents": self.num_agents,
        }

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

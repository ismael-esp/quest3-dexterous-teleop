# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import torch

import isaaclab.utils.math as math_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg


@dataclass
class GripperMapping:
    """
    Allegro Hand 그리퍼(파지) 값을 퍼지 joint들에 뿌리기 위한 매핑 테이블.
    value_joints: 그리퍼 값(닫힘 각도)을 복사해서 넣을 조인트 이름들
    zero_joints : 항상 0으로 둘 조인트 이름들
    close_angle : '닫힘'일 때 넣을 라디안 값 (예: 1.57)
    open_angle  : '열림'일 때 넣을 라디안 값 (예: 0.0)
    """
    value_joints: Tuple[str, ...]
    zero_joints: Tuple[str, ...]
    close_angle: float = 1.57
    open_angle: float = 0.0


class DualArmTeleop:
    """
    Dual-arm(오른손/왼손) 텔레오퍼레이션 보조 클래스.
    - EE 포즈/자코비안(body frame) 계산
    - Diff-IK로 관절 목표 각도 계산
    - Allegro Hand 그리퍼 매핑 적용
    """

    def __init__(
        self,
        right_robot,
        left_robot,
        device: torch.device,
        num_envs: int,
        ik_cfg: Optional[DifferentialIKControllerCfg] = None,
        right_ee_body_name: str = "ra_wrist_3_link",
        left_ee_body_name: str = "la_wrist_3_link",
    ):
        self.right_robot = right_robot
        self.left_robot = left_robot
        self.device = device
        self.num_envs = num_envs

        # IK 컨트롤러 설정
        if ik_cfg is None:
            ik_cfg = DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=True)
        self.ik_R = DifferentialIKController(cfg=ik_cfg, num_envs=num_envs, device=device)
        self.ik_L = DifferentialIKController(cfg=ik_cfg, num_envs=num_envs, device=device)

        # UR3 팔 조인트 인덱스
        self._joint_ids_R = self.right_robot.actuators["ur3_arm"].joint_indices
        self._joint_ids_L = self.left_robot.actuators["ur3_arm"].joint_indices

        # EE body index
        body_ids_R, _ = self.right_robot.find_bodies(right_ee_body_name)
        body_ids_L, _ = self.left_robot.find_bodies(left_ee_body_name)
        self._body_idx_R = body_ids_R[0]
        self._body_idx_L = body_ids_L[0]

        # fixed base 여부에 따라 자코비안 대상 인덱스 보정
        if self.right_robot.is_fixed_base:
            self._jacobi_body_idx_R = self._body_idx_R - 1
            self._jacobi_joint_ids_R = self._joint_ids_R
        else:
            self._jacobi_body_idx_R = self._body_idx_R
            self._jacobi_joint_ids_R = [i + 6 for i in self._joint_ids_R]

        if self.left_robot.is_fixed_base:
            self._jacobi_body_idx_L = self._body_idx_L - 1
            self._jacobi_joint_ids_L = self._joint_ids_L
        else:
            self._jacobi_body_idx_L = self._body_idx_L
            self._jacobi_joint_ids_L = [i + 6 for i in self._joint_ids_L]

        # 링크 오프셋(필요 시 사용). 기본은 None.
        self._offset_pos = None  # torch.tensor([0.,0.,0.], device=device)
        self._offset_rot = None  # torch.tensor([1.,0.,0.,0.], device=device)

    # ---------------------------
    # 기본 유틸
    # ---------------------------
    @staticmethod
    def build_joint_name_to_idx(robot) -> Dict[str, int]:
        """Articulation의 joint_names -> index 매핑."""
        return {name: i for i, name in enumerate(robot.joint_names)}

    @staticmethod
    def _split_actions(actions: torch.Tensor):
        """
        텔레옵 action을 분리:
        [0:6]   오른손 dpose(6), [6]   오른손 grip
        [7:13]  왼손 dpose(6),   [13]  왼손 grip
        """
        dpose_R = actions[:, 0:6]
        grip_R = actions[:, 6:7]
        dpose_L = actions[:, 7:13]
        grip_L = actions[:, 13:14]
        return dpose_R, grip_R, dpose_L, grip_L

    # ---------------------------
    # Kinematics (pose/jacobian)
    # ---------------------------
    def _compute_frame_pose_body(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 오른손
        ee_pos_w_R = self.right_robot.data.body_pos_w[:, self._body_idx_R]
        ee_quat_w_R = self.right_robot.data.body_quat_w[:, self._body_idx_R]
        root_pos_w_R = self.right_robot.data.root_pos_w
        root_quat_w_R = self.right_robot.data.root_quat_w
        ee_pos_b_R, ee_quat_b_R = math_utils.subtract_frame_transforms(
            root_pos_w_R, root_quat_w_R, ee_pos_w_R, ee_quat_w_R
        )

        # 왼손
        ee_pos_w_L = self.left_robot.data.body_pos_w[:, self._body_idx_L]
        ee_quat_w_L = self.left_robot.data.body_quat_w[:, self._body_idx_L]
        root_pos_w_L = self.left_robot.data.root_pos_w
        root_quat_w_L = self.left_robot.data.root_quat_w
        ee_pos_b_L, ee_quat_b_L = math_utils.subtract_frame_transforms(
            root_pos_w_L, root_quat_w_L, ee_pos_w_L, ee_quat_w_L
        )

        # 필요 시 오프셋 적용
        if (self._offset_pos is not None) and (self._offset_rot is not None):
            Rpos, Rquat = math_utils.combine_frame_transforms(
                ee_pos_b_R, ee_quat_b_R, self._offset_pos, self._offset_rot
            )
            Lpos, Lquat = math_utils.combine_frame_transforms(
                ee_pos_b_L, ee_quat_b_L, self._offset_pos, self._offset_rot
            )
            return Rpos, Rquat, Lpos, Lquat

        return ee_pos_b_R, ee_quat_b_R, ee_pos_b_L, ee_quat_b_L

    def _compute_frame_pose_world(self):
        ee_pos_w_R = self.right_robot.data.body_pos_w[:, self._body_idx_R]
        ee_quat_w_R = self.right_robot.data.body_quat_w[:, self._body_idx_R]
        ee_pos_w_L = self.left_robot.data.body_pos_w[:, self._body_idx_L]
        ee_quat_w_L = self.left_robot.data.body_quat_w[:, self._body_idx_L]

        return ee_pos_w_R, ee_quat_w_R, ee_pos_w_L, ee_quat_w_L

    def _compute_frame_jacobian_body(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # 오른손 jacobian (world)
        jac_w_R = self.right_robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx_R, :, self._jacobi_joint_ids_R]
        base_rot_R = self.right_robot.data.root_quat_w
        Rinv_R = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot_R))
        jac_b_R = jac_w_R.clone()
        jac_b_R[:, :3, :] = torch.bmm(Rinv_R, jac_b_R[:, :3, :])
        jac_b_R[:, 3:, :] = torch.bmm(Rinv_R, jac_b_R[:, 3:, :])

        # 왼손 jacobian (world)
        jac_w_L = self.left_robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx_L, :, self._jacobi_joint_ids_L
        ]
        base_rot_L = self.left_robot.data.root_quat_w
        Rinv_L = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot_L))
        jac_b_L = jac_w_L.clone()
        jac_b_L[:, :3, :] = torch.bmm(Rinv_L, jac_b_L[:, :3, :])
        jac_b_L[:, 3:, :] = torch.bmm(Rinv_L, jac_b_L[:, 3:, :])

        # 필요 시 링크 오프셋 반영
        if (self._offset_pos is not None) and (self._offset_rot is not None):
            # v_link = v_ee - r_x * w_ee
            jac_b_R[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jac_b_R[:, 3:, :])
            jac_b_R[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jac_b_R[:, 3:, :])

            jac_b_L[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jac_b_L[:, 3:, :])
            jac_b_L[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jac_b_L[:, 3:, :])

        return jac_b_R, jac_b_L

    def _compute_frame_jacobian_world(self):
        jac_w_R = self.right_robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx_R, :, self._jacobi_joint_ids_R
        ]
        jac_w_L = self.left_robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx_L, :, self._jacobi_joint_ids_L
        ]
        return jac_w_R, jac_w_L

    # ---------------------------
    # Public API
    # ---------------------------
    def step(
        self,
        actions: torch.Tensor,
        right_actions_buf: torch.Tensor,
        left_actions_buf: torch.Tensor,
        right_joint_name_to_idx: Dict[str, int],
        left_joint_name_to_idx: Dict[str, int],
        right_gripper_map: GripperMapping,
        left_gripper_map: GripperMapping,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        텔레옵 step:
        - dpose 명령으로 Diff-IK → UR3 관절 목표 각도 계산
        - 그리퍼 스칼라 값을 Allegro joint들에 복사
        - action 버퍼(right_actions_buf / left_actions_buf)를 업데이트
        - demos(오른/왼 7-DOF 프레임: 6(UR3) + 1(grip)) 반환
        """
        dpose_R, grip_R, dpose_L, grip_L = self._split_actions(actions)

        # 현재 EE 포즈(body frame)
        # ee_pos_R, ee_quat_R, ee_pos_L, ee_quat_L = self._compute_frame_pose_body()
        ee_pos_R, ee_quat_R, ee_pos_L, ee_quat_L = self._compute_frame_pose_world()

        # Diff-IK 설정
        self.ik_R.set_command(dpose_R, ee_pos_R, ee_quat_R)
        self.ik_L.set_command(dpose_L, ee_pos_L, ee_quat_L)

        # 현재 관절각 (UR3)
        joint_pos_R = self.right_robot.data.joint_pos[:, self._joint_ids_R]
        joint_pos_L = self.left_robot.data.joint_pos[:, self._joint_ids_L]

        # 자코비안 계산 가능 조건 확인(쿼터니언 노름)
        if (ee_quat_R.norm() != 0) and (ee_quat_L.norm() != 0):
            # jac_R, jac_L = self._compute_frame_jacobian_body()
            jac_R, jac_L = self._compute_frame_jacobian_world()

            q_des_R = self.ik_R.compute(ee_pos_R, ee_quat_R, jac_R, joint_pos_R)
            q_des_L = self.ik_L.compute(ee_pos_L, ee_quat_L, jac_L, joint_pos_L)
        else:
            q_des_R = joint_pos_R.clone()
            q_des_L = joint_pos_L.clone()

        # 버퍼에 UR3 목표 관절각 채우기 (앞 6개)
        right_actions_buf[:, :6] = q_des_R
        left_actions_buf[:, :6] = q_des_L

        # 그리퍼 스칼라 → Allegro Hand 조인트로 복사
        self._apply_gripper_mapping(
            buf=right_actions_buf,
            joint_name_to_idx=right_joint_name_to_idx,
            mapping=right_gripper_map,
            grip_scalar=grip_R,
        )
        self._apply_gripper_mapping(
            buf=left_actions_buf,
            joint_name_to_idx=left_joint_name_to_idx,
            mapping=left_gripper_map,
            grip_scalar=grip_L,
        )

        # demos: [num_envs, 7] (6 x UR3 + 1 x grip)
        right_demo = torch.zeros((self.num_envs, 7), device=self.device)
        left_demo = torch.zeros((self.num_envs, 7), device=self.device)
        right_demo[:, :6] = q_des_R
        # Linear, not a ==1.0 snap -- see _apply_gripper_mapping for why the
        # discontinuity broke behaviour cloning at the closing step.
        right_demo[:, 6] = ((1.0 - grip_R.squeeze(-1)) * 0.5).clamp(0.0, 1.0) * 1.57
        left_demo[:, :6] = q_des_L
        left_demo[:, 6] = ((1.0 - grip_L.squeeze(-1)) * 0.5).clamp(0.0, 1.0) * 1.57

        return right_actions_buf, left_actions_buf, right_demo, left_demo


    # ---------------------------
    # 내부: 그리퍼 매핑
    # ---------------------------
    def _apply_gripper_mapping(
        self,
        buf: torch.Tensor,
        joint_name_to_idx: Dict[str, int],
        mapping: GripperMapping,
        grip_scalar: torch.Tensor,
    ):
        """
        grip_scalar +1 → open_angle, -1 → close_angle, LINEAR in between.
        해당 값을 value_joints 에 복사하고, zero_joints 는 0으로 둔다.

        This was `torch.where(grip_scalar == 1.0, open, close)`, i.e. any value other
        than exactly +1.0 snapped to fully closed. That made the hand-close a ONE-STEP
        discontinuity in the recorded demo actions, and a BC policy cannot represent it:
        measured action RMSE on the demos' own states was 0.008-0.012 (2-3% of |a|) at
        every timestep EXCEPT the closing step, where it was **0.376 (81%)** -- the model
        averages the two modes, so the hand ends up neither open nor properly closed and
        the grasp fails. The linear map is exactly equivalent at +-1, so every caller that
        only ever passes +-1 is unaffected; it just lets a ramped grip survive.
        """
        # grip_scalar: [B,1] → [B,1] → 스칼라 angle로 변환
        angle_open = mapping.open_angle
        angle_close = mapping.close_angle
        closure = ((1.0 - grip_scalar) * 0.5).clamp(0.0, 1.0)
        angle = angle_open + closure * (angle_close - angle_open)
        angle = angle.view(-1, 1)  # [B,1]

        if mapping.value_joints:
            ids = [joint_name_to_idx[n] for n in mapping.value_joints]
            buf[:, ids] = angle  # 브로드캐스팅

        if mapping.zero_joints:
            ids = [joint_name_to_idx[n] for n in mapping.zero_joints]
            buf[:, ids] = 0.0


# ---------------------------
# 편의 유틸 (매핑 프리셋)
# ---------------------------

def make_default_right_gripper_mapping() -> GripperMapping:
    """
    Default right Allegro mapping used by the handover task scalar-grip path.
    """
    right_value = (
        "ra_joint_1_0", "ra_joint_13_0", "ra_joint_5_0", "ra_joint_9_0",
        "ra_joint_3_0", "ra_joint_15_0", "ra_joint_7_0", "ra_joint_11_0", "ra_joint_12_0",
        "ra_joint_2_0", "ra_joint_14_0", "ra_joint_6_0", "ra_joint_10_0",
    )
    right_zero = (
        "ra_joint_0_0", "ra_joint_4_0", "ra_joint_8_0",
    )
    return GripperMapping(value_joints=right_value, zero_joints=right_zero, close_angle=1.57, open_angle=0.0)


def make_default_left_gripper_mapping() -> GripperMapping:
    """
    Default left Allegro mapping used by the handover task scalar-grip path.
    """
    left_value = (
        "la_joint_1_0", "la_joint_13_0", "la_joint_5_0", "la_joint_9_0",
        "la_joint_3_0", "la_joint_15_0", "la_joint_7_0", "la_joint_11_0", "la_joint_12_0",
        "la_joint_2_0", "la_joint_14_0", "la_joint_6_0", "la_joint_10_0",
    )
    left_zero = (
        "la_joint_0_0", "la_joint_4_0", "la_joint_8_0",
    )
    return GripperMapping(value_joints=left_value, zero_joints=left_zero, close_angle=1.57, open_angle=0.0)

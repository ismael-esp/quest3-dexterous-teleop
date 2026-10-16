"""Coordinate frames for VR teleop (pure numpy).

WebXR reference space: y-up, -z forward (where the headset looked at session start), +x right.
Isaac world: z-up. WEBXR_TO_WORLD maps VR axes -> world axes so that the operator faces -y
(the cell's front is +y, so standing at +y and looking toward -y). A yaw about world z and a
translation (`--vr_origin`) are applied on top.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# world = WEBXR_TO_WORLD @ vr :  x_w = -x_vr, y_w = z_vr, z_w = y_vr   (det = +1)
WEBXR_TO_WORLD = np.array([[-1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0],
                           [0.0, 1.0, 0.0]])


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def T_world_vr(origin_xyz, yaw_deg: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot_z(np.deg2rad(yaw_deg)) @ WEBXR_TO_WORLD
    T[:3, 3] = np.asarray(origin_xyz, dtype=np.float64)
    return T


def transform_pose(T_ab: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    return T_ab @ T_b


def embody_T_world_vr(T_vr_head: np.ndarray, eye_pos_world, eye_yaw_deg: float = 90.0) -> np.ndarray:
    """VR->world transform that puts the operator's eyes at `eye_pos_world` (e.g. the robot's head)
    with the head's horizontal look direction along world heading `eye_yaw_deg` (90 = +y, toward the
    table). VR up stays world up; only a yaw + translation are chosen, so pitch/roll and everything
    relative to the head (hands!) are preserved 1:1. `T_vr_head` is in the WebXR camera convention
    (looks -z, up +y). If the head looks straight up/down, the head's up axis is used for heading."""
    T_vr_head = np.asarray(T_vr_head, dtype=np.float64)
    R0 = WEBXR_TO_WORLD
    f_w = R0 @ (-T_vr_head[:3, 2])                 # look direction in world (before yaw)
    if np.hypot(f_w[0], f_w[1]) < 0.2:             # looking (nearly) straight up/down
        u_w = R0 @ T_vr_head[:3, 1]
        f_w = u_w if T_vr_head[1, 2] > 0 else -u_w   # looking down: head-up points forward; looking up: backward
    h0 = np.arctan2(f_w[1], f_w[0])
    yaw = np.deg2rad(eye_yaw_deg) - h0
    R = rot_z(yaw) @ R0
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(eye_pos_world, dtype=np.float64) - R @ T_vr_head[:3, 3]
    return T


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w,x,y,z), w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def quat_wxyz_to_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_from_T(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.array(T[:3, 3], dtype=np.float64), mat_to_quat_wxyz(T[:3, :3])


class AnchoredMapper:
    """Clutch: on engage, bind the current hand pose to the current EE pose; afterwards
    p_tgt = p_e0 + s*C(p_h - p_h0),  R_tgt = C (R_h R_h0^T) C^T R_e0  where C is an optional
    view-correction rotation (identity by default). C re-expresses the operator's world-frame hand
    delta as if they were standing in the cell's canonical direction, so the mapping stays intuitive
    when teleoperating from the front or the side (world-frame 1:1 feels inverted from the front)."""

    def __init__(self, pos_scale: float = 1.0):
        self.pos_scale = float(pos_scale)
        self.engaged = False
        self._p_h0: Optional[np.ndarray] = None
        self._R_h0: Optional[np.ndarray] = None
        self._p_e0: Optional[np.ndarray] = None
        self._R_e0: Optional[np.ndarray] = None
        self._C: np.ndarray = np.eye(3)

    def engage(self, p_h, R_h, p_e, R_e, R_view: Optional[np.ndarray] = None) -> None:
        self._p_h0 = np.array(p_h, dtype=np.float64); self._R_h0 = np.array(R_h, dtype=np.float64)
        self._p_e0 = np.array(p_e, dtype=np.float64); self._R_e0 = np.array(R_e, dtype=np.float64)
        self._C = np.eye(3) if R_view is None else np.array(R_view, dtype=np.float64)
        self.engaged = True

    def disengage(self) -> None:
        self.engaged = False

    def target(self, p_h, R_h) -> Tuple[np.ndarray, np.ndarray]:
        if not self.engaged:
            raise RuntimeError("AnchoredMapper.target() called while disengaged")
        p_h = np.asarray(p_h, dtype=np.float64); R_h = np.asarray(R_h, dtype=np.float64)
        p_tgt = self._p_e0 + self.pos_scale * (self._C @ (p_h - self._p_h0))
        R_tgt = (self._C @ (R_h @ self._R_h0.T) @ self._C.T) @ self._R_e0
        return p_tgt, R_tgt


def head_forward(R: np.ndarray, forward_axis: str = "-z") -> np.ndarray:
    """Facing direction of a head rotation matrix. forward_axis picks the convention; both the
    WebXR bridge and Kit XR's /user/head virtual-world pose use the camera convention "-z"
    (confirmed live: with "+z" the view-relative engage mapping inverted for an operator facing
    the robots). "+z" stays available for pose sources that look along +z.
    Looking straight up/down falls back to the head's up axis, tilted the way the head is."""
    R = np.asarray(R, dtype=np.float64)[:3, :3]
    f = R[:, 2] if forward_axis == "+z" else -R[:, 2]
    if np.hypot(f[0], f[1]) < 0.2:                 # looking straight up/down: fall back to head up axis
        f = R[:, 1] if f[2] < 0 else -R[:, 1]
    return f


def view_correction_from_head(T_head: np.ndarray, cell_forward_yaw_deg: float = 90.0,
                              snap_deg: float = 90.0, forward_axis: str = "-z") -> np.ndarray:
    """Rotation C about world z mapping the operator's current facing onto the cell's forward
    direction (default +y = toward the table), snapped to the nearest `snap_deg` so small head turns
    at engage time do not skew the mapping. Standing behind the robots facing the table -> identity;
    standing in front facing the robots -> 180 deg (push still moves the arm toward the table)."""
    f = head_forward(T_head, forward_axis)
    yaw_head = np.arctan2(f[1], f[0])
    corr = np.deg2rad(cell_forward_yaw_deg) - yaw_head
    if snap_deg > 0:
        step = np.deg2rad(snap_deg)
        corr = round(corr / step) * step
    return rot_z(corr)


def view_nudge_delta(key: str, T_head=None, fallback_yaw_deg: float = 90.0,
                     step: float = 0.05, forward_axis: str = "-z") -> np.ndarray:
    """World-frame translation for one view-nudge key while view-hold is active.
    "UP"/"DOWN" move along the operator's horizontal facing, "RIGHT"/"LEFT" strafe,
    "o"/"l" move straight up/down. Facing comes from the head pose (head_forward with the
    given convention); without one, from fallback_yaw_deg."""
    if key in ("o", "l"):
        return np.array([0.0, 0.0, step if key == "o" else -step])
    if T_head is not None:
        f = head_forward(T_head, forward_axis)
        yaw = np.arctan2(f[1], f[0])
    else:
        yaw = np.deg2rad(fallback_yaw_deg)
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    right = np.array([fwd[1], -fwd[0], 0.0])
    return {"UP": fwd, "DOWN": -fwd, "RIGHT": right, "LEFT": -right}[key] * step


def recenter_camera_pose(T_head, eye_world, eye_yaw_deg: float = 90.0,
                         forward_axis: str = "-z") -> np.ndarray:
    """Target camera pose (4x4 world) that puts the operator's eyes at eye_world facing
    eye_yaw_deg horizontally: the current head orientation yawed about world z until its facing
    matches, so pitch/roll (how the operator actually holds their head) are preserved and the
    horizon stays level. Feed to XRCore.schedule_set_camera (transposed into Gf row-vector form)."""
    T_head = np.asarray(T_head, dtype=np.float64)
    f = head_forward(T_head[:3, :3], forward_axis)
    dyaw = np.deg2rad(eye_yaw_deg) - np.arctan2(f[1], f[0])
    M = np.eye(4)
    M[:3, :3] = rot_z(dyaw) @ T_head[:3, :3]
    M[:3, 3] = np.asarray(eye_world, dtype=np.float64)
    return M


def look_at_camera_pose(pos, target, up=(0.0, 0.0, 1.0)):
    """Static camera pose (position, quat wxyz) in the opengl/USD convention (looks -z, +y up)
    placed at `pos` looking at `target`, roll chosen so the camera's up leans toward `up`."""
    pos = np.asarray(pos, dtype=np.float64)
    z = pos - np.asarray(target, dtype=np.float64)
    z = z / np.linalg.norm(z)
    x = np.cross(np.asarray(up, dtype=np.float64), z)
    n = np.linalg.norm(x)
    if n < 1e-6:                                   # looking straight up/down: pick any horizontal right
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = x / n
    y = np.cross(z, x)
    return pos, mat_to_quat_wxyz(np.column_stack([x, y, z]))

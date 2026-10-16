import numpy as np
import pytest

from isaac_teleop.frames import (
    WEBXR_TO_WORLD, rot_z, T_world_vr, transform_pose, pose_from_T,
    mat_to_quat_wxyz, quat_wxyz_to_mat, AnchoredMapper,
)


def test_webxr_to_world_axes():
    B = WEBXR_TO_WORLD
    assert np.isclose(np.linalg.det(B), 1.0)                       # proper rotation
    np.testing.assert_allclose(B @ [0, 1, 0], [0, 0, 1])           # VR up -> world +z
    np.testing.assert_allclose(B @ [0, 0, -1], [0, -1, 0])         # VR forward -> world -y
    np.testing.assert_allclose(B @ [1, 0, 0], [-1, 0, 0])          # VR right -> world -x


def test_T_world_vr_origin_and_yaw():
    T = T_world_vr((0.0, 1.2, 0.0), yaw_deg=0.0)
    head_vr = np.eye(4); head_vr[:3, 3] = [0, 1.6, 0]
    head_w = transform_pose(T, head_vr)
    np.testing.assert_allclose(head_w[:3, 3], [0, 1.2, 1.6], atol=1e-12)
    # yaw 90 deg (ccw from above): forward -y -> +x
    T90 = T_world_vr((0, 0, 0), yaw_deg=90.0)
    fwd = T90[:3, :3] @ np.array([0, 0, -1.0])
    np.testing.assert_allclose(fwd, [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(rot_z(np.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_quat_roundtrip_and_pose_from_T():
    R = rot_z(0.7) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    q = mat_to_quat_wxyz(R)
    assert q[0] >= 0                          # canonical hemisphere
    np.testing.assert_allclose(quat_wxyz_to_mat(q), R, atol=1e-12)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [1, 2, 3]
    p, qq = pose_from_T(T)
    np.testing.assert_allclose(p, [1, 2, 3]); np.testing.assert_allclose(qq, q)


def test_anchored_mapper_identity_at_engage_and_translation():
    m = AnchoredMapper(pos_scale=1.0)
    assert not m.engaged
    with pytest.raises(RuntimeError):
        m.target(np.zeros(3), np.eye(3))
    p_h0 = np.array([0.1, 1.0, -0.3]); R_h0 = rot_z(0.3)
    p_e0 = np.array([0.4, 0.5, 1.2]);  R_e0 = rot_z(-1.0)
    m.engage(p_h0, R_h0, p_e0, R_e0)
    p, R = m.target(p_h0, R_h0)
    np.testing.assert_allclose(p, p_e0, atol=1e-12); np.testing.assert_allclose(R, R_e0, atol=1e-12)
    p, R = m.target(p_h0 + [0.05, 0, 0.02], R_h0)
    np.testing.assert_allclose(p, p_e0 + [0.05, 0, 0.02], atol=1e-12)


def test_anchored_mapper_rotation_delta_and_scale():
    m = AnchoredMapper(pos_scale=0.5)
    R_h0 = np.eye(3); R_e0 = rot_z(0.2)
    m.engage(np.zeros(3), R_h0, np.zeros(3), R_e0)
    dR = rot_z(0.4)
    p, R = m.target(np.array([0.2, 0, 0]), dR @ R_h0)
    np.testing.assert_allclose(R, dR @ R_e0, atol=1e-12)     # left-multiplied world delta
    np.testing.assert_allclose(p, [0.1, 0, 0], atol=1e-12)   # scaled
    m.disengage()
    assert not m.engaged


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_embody_T_world_vr_puts_head_at_eye_pos_facing_eye_yaw():
    from isaac_teleop.frames import embody_T_world_vr
    # operator's head somewhere in the VR room, turned 30 deg left of the session-start forward and
    # pitched down 20 deg (camera convention: looks -z, up +y)
    head_vr = np.eye(4)
    head_vr[:3, :3] = _rot_y(np.deg2rad(30)) @ np.array([[1, 0, 0], [0, np.cos(-.35), -np.sin(-.35)], [0, np.sin(-.35), np.cos(-.35)]])
    head_vr[:3, 3] = [0.7, 1.72, -1.1]
    eye = np.array([0.0, -0.42, 1.34])
    T = embody_T_world_vr(head_vr, eye, eye_yaw_deg=90.0)      # face world +y
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)
    head_w = transform_pose(T, head_vr)
    np.testing.assert_allclose(head_w[:3, 3], eye, atol=1e-9)              # eyes at the robot head
    fwd = head_w[:3, :3] @ [0, 0, -1.0]                                    # horizontal heading = +y
    assert np.isclose(np.arctan2(fwd[1], fwd[0]), np.pi / 2, atol=1e-9)
    assert fwd[2] < -0.3                                                   # pitch (looking down) preserved
    np.testing.assert_allclose(T[:3, :3] @ [0, 1, 0], [0, 0, 1], atol=1e-12)  # VR up stays world up
    # a hand 30 cm to the operator's right ends up at +x of the head (right robot arm side)
    hand_vr = np.eye(4); hand_vr[:3, 3] = head_vr[:3, 3] + head_vr[:3, :3] @ [0.3, -0.5, -0.4]
    hand_w = transform_pose(T, hand_vr)
    d = hand_w[:3, 3] - head_w[:3, 3]
    assert d[0] > 0.2 and d[1] > 0.15 and d[2] < -0.3      # 0.4 fwd pitched 20 deg down -> ~0.20 in +y


def test_embody_T_world_vr_looking_straight_down_uses_up_axis_heading():
    from isaac_teleop.frames import embody_T_world_vr
    head_vr = np.eye(4)
    head_vr[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])   # looks straight down (-y), up = -z (VR forward)
    T = embody_T_world_vr(head_vr, [0, 0, 1.5], eye_yaw_deg=90.0)
    up_w = transform_pose(T, head_vr)[:3, :3] @ [0, 1, 0]                # head "up" (= body forward) -> +y
    assert np.isclose(np.arctan2(up_w[1], up_w[0]), np.pi / 2, atol=1e-9)


def test_view_correction_and_mapper_conjugation():
    from isaac_teleop.frames import view_correction_from_head, AnchoredMapper
    def head_facing(yaw):
        T = np.eye(4)
        # camera looks along -z_local; build R with -z pointing (cos yaw, sin yaw, 0), up = +z world -> y_local
        f = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        z = -f; y = np.array([0.0, 0.0, 1.0]); x = np.cross(y, z)
        T[:3, :3] = np.stack([x, y, z], axis=1)
        return T
    # behind the robots, facing the table (+y): identity
    C = view_correction_from_head(head_facing(np.pi / 2))
    np.testing.assert_allclose(C, np.eye(3), atol=1e-12)
    # in front, facing the robots (-y): 180 deg about z
    C = view_correction_from_head(head_facing(-np.pi / 2))
    np.testing.assert_allclose(C @ [1, 0, 0], [-1, 0, 0], atol=1e-12)
    # 25 deg head turn snaps away (stays identity)
    C = view_correction_from_head(head_facing(np.pi / 2 + np.deg2rad(25)))
    np.testing.assert_allclose(C, np.eye(3), atol=1e-12)
    # mapper: from the front, pushing away from me (world -y) must move the EE toward the table (+y)
    m = AnchoredMapper()
    C180 = view_correction_from_head(head_facing(-np.pi / 2))
    m.engage(np.zeros(3), np.eye(3), np.array([0.3, 0.4, 1.0]), np.eye(3), R_view=C180)
    p, R = m.target(np.array([0.0, -0.2, 0.0]), np.eye(3))
    np.testing.assert_allclose(p, [0.3, 0.6, 1.0], atol=1e-12)
    # rotation delta conjugation: I roll about MY forward (-y) by a; robot rolls about ITS forward (+y) by a
    a = 0.4
    c, s_ = np.cos(a), np.sin(a)
    R_h = np.array([[c, 0, -s_], [0, 1, 0], [s_, 0, c]])   # rotation about -y ... about y axis by -a
    _, R_t = m.target(np.zeros(3), R_h)
    np.testing.assert_allclose(R_t, np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]), atol=1e-12)


def _head_T(yaw):
    """Camera-convention head pose looking along horizontal yaw (x=right, y=up, z=backward)."""
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, 0] = [s, -c, 0]
    T[:3, 1] = [0, 0, 1]
    T[:3, 2] = [-c, -s, 0]
    return T


def test_view_nudge_delta_view_relative():
    from isaac_teleop.frames import view_nudge_delta
    T = _head_T(np.pi / 2)                                   # facing +y (toward the table)
    np.testing.assert_allclose(view_nudge_delta("UP", T, step=0.05), [0, 0.05, 0], atol=1e-12)
    np.testing.assert_allclose(view_nudge_delta("DOWN", T, step=0.05), [0, -0.05, 0], atol=1e-12)
    np.testing.assert_allclose(view_nudge_delta("RIGHT", T, step=0.05), [0.05, 0, 0], atol=1e-12)
    np.testing.assert_allclose(view_nudge_delta("LEFT", T, step=0.05), [-0.05, 0, 0], atol=1e-12)
    T = _head_T(np.pi)                                       # facing -x: forward nudge is -x
    np.testing.assert_allclose(view_nudge_delta("UP", T, step=0.1), [-0.1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(view_nudge_delta("RIGHT", T, step=0.1), [0, 0.1, 0], atol=1e-12)


def test_view_nudge_delta_vertical_and_fallback():
    from isaac_teleop.frames import view_nudge_delta
    np.testing.assert_allclose(view_nudge_delta("o", None, step=0.05), [0, 0, 0.05], atol=1e-12)
    np.testing.assert_allclose(view_nudge_delta("l", None, step=0.05), [0, 0, -0.05], atol=1e-12)
    # no head pose: fall back to the cell forward yaw (90 deg = +y)
    np.testing.assert_allclose(view_nudge_delta("UP", None, fallback_yaw_deg=90.0, step=0.05),
                               [0, 0.05, 0], atol=1e-12)


def _head_T_zfwd(yaw):
    """+z-forward head pose looking along horizontal yaw (x=left, y=up, z=forward). Both real
    pipelines use "-z" (see head_forward); this exercises the alternate convention parameter."""
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, 0] = [-s, c, 0]
    T[:3, 1] = [0, 0, 1]
    T[:3, 2] = [c, s, 0]
    return T


def test_head_forward_both_conventions():
    from isaac_teleop.frames import head_forward
    f = head_forward(_head_T(np.pi / 2)[:3, :3], forward_axis="-z")
    np.testing.assert_allclose(f, [0, 1, 0], atol=1e-12)
    f = head_forward(_head_T_zfwd(np.pi / 2)[:3, :3], forward_axis="+z")
    np.testing.assert_allclose(f, [0, 1, 0], atol=1e-12)
    # same pose read with the wrong convention faces the opposite way
    f = head_forward(_head_T_zfwd(np.pi / 2)[:3, :3], forward_axis="-z")
    np.testing.assert_allclose(f, [0, -1, 0], atol=1e-12)


def test_view_correction_plus_z_convention():
    from isaac_teleop.frames import view_correction_from_head
    # operator faces the table (+y): correction must be identity, not a 180 flip
    C = view_correction_from_head(_head_T_zfwd(np.pi / 2), forward_axis="+z")
    np.testing.assert_allclose(C, np.eye(3), atol=1e-12)
    # facing -x: snapped correction rotates -x onto +y (90 deg about z... -x -> +y is -90)
    C = view_correction_from_head(_head_T_zfwd(np.pi), forward_axis="+z")
    np.testing.assert_allclose(C @ np.array([-1.0, 0, 0]), [0, 1, 0], atol=1e-12)


def test_view_nudge_delta_plus_z_convention():
    from isaac_teleop.frames import view_nudge_delta
    d = view_nudge_delta("UP", _head_T_zfwd(np.pi / 2), step=0.05, forward_axis="+z")
    np.testing.assert_allclose(d, [0, 0.05, 0], atol=1e-12)
    d = view_nudge_delta("RIGHT", _head_T_zfwd(np.pi / 2), step=0.05, forward_axis="+z")
    np.testing.assert_allclose(d, [0.05, 0, 0], atol=1e-12)


def test_recenter_camera_pose_places_eyes_facing_table():
    from isaac_teleop.frames import head_forward, recenter_camera_pose
    rng = np.random.default_rng(7)
    eye = np.array([0.0, 0.05, 1.65])
    for _ in range(50):
        yaw = rng.uniform(-np.pi, np.pi)
        pitch = rng.uniform(-0.6, 0.6)
        cp, sp = np.cos(pitch), np.sin(pitch)
        T = _head_T(yaw)                     # -z-forward head at some yaw
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])  # local pitch
        T[:3, :3] = T[:3, :3] @ Rx
        T[:3, 3] = rng.uniform(-2, 2, 3)
        M = recenter_camera_pose(T, eye, eye_yaw_deg=90.0, forward_axis="-z")
        np.testing.assert_allclose(M[:3, 3], eye, atol=1e-12)
        f = head_forward(M[:3, :3], "-z")
        assert abs(np.degrees(np.arctan2(f[1], f[0])) - 90.0) < 1e-9   # faces the table
        f0 = head_forward(T[:3, :3], "-z")
        np.testing.assert_allclose(f[2], f0[2], atol=1e-12)            # pitch preserved
        np.testing.assert_allclose(M[:3, :3] @ M[:3, :3].T, np.eye(3), atol=1e-12)


def test_look_at_camera_pose():
    from isaac_teleop.frames import look_at_camera_pose, quat_wxyz_to_mat
    pos, tgt = np.array([0.0, 1.9, 1.5]), np.array([0.0, 0.25, 1.1])
    p, q = look_at_camera_pose(pos, tgt)
    np.testing.assert_allclose(p, pos)
    R = quat_wxyz_to_mat(q)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    d = tgt - pos
    np.testing.assert_allclose(-R[:, 2], d / np.linalg.norm(d), atol=1e-12)   # -z looks at target
    assert R[2, 1] > 0                                                        # camera up has world-up component

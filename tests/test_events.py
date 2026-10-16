import numpy as np

from quest_bridge.events import parse_hand_value, parse_head_value, BODY_HEAD_INDEX


def _flat_hand(wrist_xyz):
    mats = np.stack([np.eye(4) for _ in range(25)])
    mats[0, :3, 3] = wrist_xyz
    return np.concatenate([m.flatten(order="F") for m in mats]).tolist()


def test_parse_hand_value_both_hands_and_states():
    value = {
        "left": _flat_hand([-0.2, 1.0, -0.4]),
        "right": _flat_hand([0.2, 1.0, -0.4]),
        "leftState": {"pinch": True, "squeeze": False, "pinchValue": 0.8, "squeezeValue": 0.0,
                      "tap": False, "tapValue": 0.0},
        "rightState": {"pinch": False, "squeeze": True, "pinchValue": 0.1, "squeezeValue": 0.7},
    }
    left, right, ls, rs = parse_hand_value(value)
    assert left.shape == (25, 4, 4) and right.shape == (25, 4, 4)
    np.testing.assert_allclose(left[0, :3, 3], [-0.2, 1.0, -0.4])
    assert ls == {"pinch": True, "squeeze": False, "pinch_value": 0.8, "squeeze_value": 0.0}
    assert rs["squeeze"] is True and rs["squeeze_value"] == 0.7


def test_parse_hand_value_missing_or_bad_hand():
    left, right, ls, rs = parse_hand_value({"right": [0.0] * 399})
    assert left is None and right is None
    assert ls == {"pinch": False, "squeeze": False, "pinch_value": 0.0, "squeeze_value": 0.0}


def test_parse_head_from_camera_move():
    T = np.eye(4); T[:3, 3] = [0, 1.6, 0]
    flat = T.flatten(order="F").tolist()
    np.testing.assert_allclose(parse_head_value("CAMERA_MOVE", {"camera": {"matrix": flat}}), T)
    np.testing.assert_allclose(parse_head_value("CAMERA_MOVE", {"matrix": flat}), T)
    assert parse_head_value("CAMERA_MOVE", {"camera": {}}) is None


def test_parse_head_from_body_move():
    assert BODY_HEAD_INDEX == 6
    n_joints = 10
    body = np.zeros(16 * n_joints)
    T = np.eye(4); T[:3, 3] = [0.1, 1.5, -0.2]
    body[16 * BODY_HEAD_INDEX: 16 * BODY_HEAD_INDEX + 16] = T.flatten(order="F")
    np.testing.assert_allclose(parse_head_value("BODY_MOVE", {"body": body.tolist()}, head_axes="camera"), T)
    # degenerate (all-zero) head matrix -> None
    assert parse_head_value("BODY_MOVE", {"body": [0.0] * (16 * n_joints)}) is None
    assert parse_head_value("OTHER", {}) is None


def test_parse_hand_value_untracked_hand_is_msgpackr_undefined():
    """The vuer client (msgpackr) encodes a JS `undefined` hand as msgpack ext type 0 / b'\\x00'
    (wire bytes d4 00 00); msgpack-python yields ExtType(0, b'\\x00'). Seen live on the Quest 3
    (2026-08-19). The tracked hand must still be parsed and the untracked one must be None."""
    import msgpack
    payload = msgpack.unpackb(
        msgpack.packb({"left": msgpack.ExtType(0, b"\x00"), "right": _flat_hand([0.2, 1.0, -0.4]),
                       "leftState": {}, "rightState": {"pinch": False, "pinchValue": 0.3}}),
        raw=False)
    assert isinstance(payload["left"], msgpack.ExtType)
    left, right, ls, rs = parse_hand_value(payload)
    assert left is None
    assert right is not None and right.shape == (25, 4, 4)
    assert rs["pinch_value"] == 0.3
    # a state field that is itself `undefined` must not raise either
    _, _, ls, _ = parse_hand_value({"leftState": {"pinch": msgpack.ExtType(0, b"\x00"),
                                                  "pinchValue": msgpack.ExtType(0, b"\x00")}})
    assert ls == {"pinch": False, "squeeze": False, "pinch_value": 0.0, "squeeze_value": 0.0}


def test_parse_head_body_move_undefined_is_none():
    import msgpack
    assert parse_head_value("BODY_MOVE", {"body": msgpack.ExtType(0, b"\x00")}) is None
    assert parse_head_value("CAMERA_MOVE", {"matrix": msgpack.ExtType(0, b"\x00")}) is None


def test_body_head_axes_converted_to_camera_convention():
    """Body-tracking 'head' joint (robotis: +Y forward, +Z right, +X down) -> camera convention
    (-Z forward, +Y up, +X right). Identity head-joint frame => camera looks along VR +y ... i.e.
    the conversion is a fixed right-handed rotation, and an upright head maps to up=+y."""
    from quest_bridge.events import head_axes_matrix, HEAD_AXES
    for name in HEAD_AXES:
        M = head_axes_matrix(name)
        assert np.isclose(np.linalg.det(M), 1.0)
    np.testing.assert_allclose(head_axes_matrix("camera"), np.eye(3))
    Mb = head_axes_matrix("body")
    # a head joint whose -x is world up and +y is world forward (-z_vr) is an upright, forward-looking head:
    # R_joint columns = (x_j, y_j, z_j) in VR coords: x_j = -up = (0,-1,0), y_j = fwd = (0,0,-1), z_j = right = (1,0,0)
    R_joint = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    R_cam = R_joint @ Mb
    np.testing.assert_allclose(R_cam, np.eye(3), atol=1e-12)
    n_joints = 10
    body = np.zeros(16 * n_joints); T = np.eye(4); T[:3, :3] = R_joint; T[:3, 3] = [0, 1.6, 0]
    body[16 * BODY_HEAD_INDEX: 16 * BODY_HEAD_INDEX + 16] = T.flatten(order="F")
    H = parse_head_value("BODY_MOVE", {"body": body.tolist()}, head_axes="body")
    np.testing.assert_allclose(H[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(H[:3, 3], [0, 1.6, 0])

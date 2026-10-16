import numpy as np
import pytest

from quest_bridge.messages import (
    XR_JOINT_NAMES, XR_WRIST, XR_THUMB_TIP, XR_INDEX_TIP,
    VRFrame, StereoFrame, empty_hand_state,
    matrix_from_webxr, hand_from_webxr,
)


def test_xr_joint_table():
    assert len(XR_JOINT_NAMES) == 25
    assert XR_JOINT_NAMES[XR_WRIST] == "wrist"
    assert XR_JOINT_NAMES[XR_THUMB_TIP] == "thumb-tip"
    assert XR_JOINT_NAMES[XR_INDEX_TIP] == "index-finger-tip"
    assert XR_JOINT_NAMES[24] == "pinky-finger-tip"


def test_matrix_from_webxr_is_column_major():
    # column-major flat: columns written top-down. Translation lives in a12,a13,a14.
    flat = [1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0,   7, 8, 9, 1]
    m = matrix_from_webxr(flat)
    assert m.shape == (4, 4)
    np.testing.assert_allclose(m[:3, 3], [7, 8, 9])
    np.testing.assert_allclose(m[3], [0, 0, 0, 1])
    # a rotation stored column-major must come back as its row-major self
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    T = np.eye(4); T[:3, :3] = rot
    flat2 = T.flatten(order="F").tolist()
    np.testing.assert_allclose(matrix_from_webxr(flat2), T)


def test_hand_from_webxr_shape_and_wrist():
    mats = np.stack([np.eye(4) for _ in range(25)])
    mats[0, :3, 3] = [0.1, 0.2, 0.3]
    flat = np.concatenate([m.flatten(order="F") for m in mats]).tolist()
    assert len(flat) == 400
    hand = hand_from_webxr(flat)
    assert hand.shape == (25, 4, 4)
    np.testing.assert_allclose(hand[0, :3, 3], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        hand_from_webxr(flat[:-1])


def test_vrframe_roundtrip():
    f = VRFrame(
        t_wall=12.5, seq=3,
        head=np.eye(4),
        left=None,
        right=np.random.rand(25, 4, 4),
        left_state=empty_hand_state(),
        right_state={"pinch": True, "squeeze": False, "pinch_value": 0.9, "squeeze_value": 0.1},
    )
    g = VRFrame.from_bytes(f.to_bytes())
    assert g.t_wall == pytest.approx(12.5) and g.seq == 3
    np.testing.assert_allclose(g.head, np.eye(4))
    assert g.left is None
    np.testing.assert_allclose(g.right, f.right, rtol=1e-6)
    assert g.right_state["pinch"] is True
    assert g.right_state["pinch_value"] == pytest.approx(0.9, abs=1e-6)


def test_stereoframe_roundtrip():
    s = StereoFrame(seq=1, t_wall=2.0, left_jpeg=b"\xff\xd8L", right_jpeg=b"\xff\xd8R", hud="L:ENGAGED")
    t = StereoFrame.from_bytes(s.to_bytes())
    assert t.left_jpeg == b"\xff\xd8L" and t.right_jpeg == b"\xff\xd8R" and t.hud == "L:ENGAGED"
    assert t.seq == 1 and t.t_wall == pytest.approx(2.0)

import numpy as np
import pytest

from isaac_teleop.openxr_source import XR25_JOINT_NAMES, OpenXRHandsSource, joints_to_xr25, pose7_to_T
from isaac_teleop.xr_hand import XR_TO_MP21, xr25_to_mp21


class TT:  # mimics OpenXRDevice.TrackingTarget enum keys
    class HAND_LEFT:  name = "HAND_LEFT"
    class HAND_RIGHT: name = "HAND_RIGHT"
    class HEAD:       name = "HEAD"


def _joints(offset=0.0):
    d = {"palm": np.array([0, 0, 0, 1, 0, 0, 0.0])}
    for i, n in enumerate(XR25_JOINT_NAMES):
        d[n] = np.array([0.1 * i + offset, 1.0, -0.4, 1, 0, 0, 0])
    return d


def test_xr25_order_matches_webxr_and_mp21():
    # OpenXR-minus-palm must line up with the WebXR order xr_hand assumes: 25 joints,
    # wrist first, 4 thumb joints, then 5 per finger; MP21 drops the 4 finger metacarpals.
    assert len(XR25_JOINT_NAMES) == 25 and XR25_JOINT_NAMES[0] == "wrist"
    assert [XR25_JOINT_NAMES[i] for i in (5, 10, 15, 20)] == [
        "index_metacarpal", "middle_metacarpal", "ring_metacarpal", "little_metacarpal"]
    mp21_names = [XR25_JOINT_NAMES[i] for i in XR_TO_MP21]
    assert "metacarpal" not in " ".join(mp21_names[5:])          # only the thumb keeps its metacarpal
    assert mp21_names[1:5] == ["thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip"]
    x = joints_to_xr25(_joints())
    assert x.shape == (25, 4, 4)
    kp = xr25_to_mp21(x)
    assert kp.shape == (21, 3)
    np.testing.assert_allclose(kp[0], [0.0, 1.0, -0.4])


def test_untracked_hand_is_none_and_staleness():
    src = OpenXRHandsSource(stale_after=0.5)
    zero = {n: np.zeros(7) for n in ["palm"] + XR25_JOINT_NAMES}
    src.update({TT.HAND_LEFT: zero, TT.HAND_RIGHT: _joints(), TT.HEAD: np.array([0, 0.05, 1.6, 1, 0, 0, 0])}, t=0.0)
    assert src.hand("left") is None
    assert src.hand("right") is not None and src.fresh("right", 0.1)
    np.testing.assert_allclose(src.head()[:3, 3], [0, 0.05, 1.6])
    # frozen pose -> stale after 0.5 s, fresh again when it moves
    src.update({TT.HAND_RIGHT: _joints()}, t=0.4)
    assert src.fresh("right", 0.4)
    src.update({TT.HAND_RIGHT: _joints()}, t=1.0)
    assert not src.fresh("right", 1.0)
    src.update({TT.HAND_RIGHT: _joints(offset=0.01)}, t=1.2)
    assert src.fresh("right", 1.2)


def test_pose7_roundtrip():
    T = pose7_to_T([1, 2, 3, np.cos(0.3), 0, 0, np.sin(0.3)])
    np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)


def test_head_live_requires_changing_pose():
    """The CloudXR runtime feeds a bit-static default head pose when no headset is connected;
    a worn headset jitters every frame. head_live must reject the static pose."""
    src = OpenXRHandsSource()
    rng = np.random.default_rng(0)
    static = np.array([0.455, 0.595, 1.193, 1, 0, 0, 0])
    for i in range(30):                                  # static default: never live
        src.update({"head": static}, t=i * 0.02)
        assert not src.head_live(i * 0.02)
    t0 = 30 * 0.02
    for i in range(30):                                  # real tracking: jitters every sample
        p = static.copy(); p[:3] += rng.normal(0, 1e-3, 3) * (i + 1)
        src.update({"head": p}, t=t0 + i * 0.02)
    assert src.head_live(t0 + 29 * 0.02)
    for i in range(60):                                  # freezes again (disconnect): liveness decays
        src.update({"head": p}, t=t0 + 0.6 + i * 0.02)
    assert not src.head_live(t0 + 0.6 + 59 * 0.02)


def test_head_live_false_without_head():
    src = OpenXRHandsSource()
    assert not src.head_live(0.0)

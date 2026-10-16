import numpy as np
import pytest

from isaac_teleop.xr_hand import (
    XR_TO_MP21, xr25_to_mp21, estimate_frame_from_hand_points, mp21_to_operator_wrist,
    AllegroDexPilot, canonical_to_slots, CANON_FLEX, CANON_JOINT_NAMES,
)


def synth_mp21(close: float, side: str = "right") -> np.ndarray:
    """A crude 21-keypoint hand in metres, wrist at origin, palm in the x-y plane, fingers along +y.
    close in [0,1]: 0 = flat open hand, 1 = fist (tips curled back toward the palm, below it)."""
    sx = 1.0 if side == "right" else -1.0
    kp = np.zeros((21, 3))
    finger_x = {"index": 0.03, "middle": 0.01, "ring": -0.01, "pinky": -0.03}
    base = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    for name, b in base.items():
        x = sx * finger_x[name]
        kp[b] = [x, 0.09, 0.0]                              # MCP
        for k, frac in enumerate([0.33, 0.66, 1.0], start=1):
            l = 0.08 * frac
            # rotate the finger down/back by close*150deg about the MCP x-axis
            ang = close * np.deg2rad(150.0)
            kp[b + k] = [x, 0.09 + l * np.cos(ang), -l * np.sin(ang)]
    # thumb: CMC(1) near the wrist on the thumb side, then out sideways; curls in when closing
    kp[1] = [sx * 0.03, 0.02, 0.0]
    for k, frac in enumerate([0.33, 0.66, 1.0], start=2):
        l = 0.07 * frac
        ang = close * np.deg2rad(90.0)
        kp[k] = [sx * (0.03 + l * np.cos(ang)), 0.02 + l * 0.6, -l * np.sin(ang) * 0.6]
    return kp


def synth_xr25_from_mp21(mp21: np.ndarray) -> np.ndarray:
    """Place MP21 positions into a 25-joint XR array (identity rotations, metacarpals interpolated)."""
    xr = np.tile(np.eye(4), (25, 1, 1))
    for mp_i, xr_i in enumerate(XR_TO_MP21):
        xr[xr_i, :3, 3] = mp21[mp_i]
    # metacarpals (XR 5,10,15,20) halfway between wrist and MCP
    for meta, mcp in [(5, 5), (10, 9), (15, 13), (20, 17)]:
        xr[meta, :3, 3] = 0.5 * (mp21[0] + mp21[mcp])
    return xr


def test_xr_to_mp21_index_map():
    assert len(XR_TO_MP21) == 21
    assert XR_TO_MP21[0] == 0 and XR_TO_MP21[1:5] == [1, 2, 3, 4]
    assert XR_TO_MP21[5:9] == [6, 7, 8, 9]          # index skips XR metacarpal 5
    assert XR_TO_MP21[17:21] == [21, 22, 23, 24]    # pinky
    mp = synth_mp21(0.0)
    xr = synth_xr25_from_mp21(mp)
    np.testing.assert_allclose(xr25_to_mp21(xr), mp)


def test_estimate_frame_is_orthonormal_and_rotation_invariant():
    kp = synth_mp21(0.2)
    F = estimate_frame_from_hand_points(kp - kp[0:1])
    np.testing.assert_allclose(F.T @ F, np.eye(3), atol=1e-9)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    a = mp21_to_operator_wrist(kp, "right")
    b = mp21_to_operator_wrist(kp @ R.T + np.array([1.0, 2.0, 3.0]), "right")
    np.testing.assert_allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("side", ["right", "left"])
def test_dexpilot_open_vs_fist(side):
    rt = AllegroDexPilot(side)
    assert rt.joint_names_canonical == CANON_JOINT_NAMES
    open_q = None
    for _ in range(15):                     # let the sequential optimiser settle
        open_q = rt.retarget(mp21_to_operator_wrist(synth_mp21(0.0, side), side))
    fist_q = None
    rt.reset()
    for _ in range(15):
        fist_q = rt.retarget(mp21_to_operator_wrist(synth_mp21(1.0, side), side))
    assert open_q.shape == (16,) and fist_q.shape == (16,)
    assert np.all(np.isfinite(open_q)) and np.all(np.isfinite(fist_q))
    flex = list(CANON_FLEX)
    assert fist_q[flex].mean() > open_q[flex].mean() + 0.3


def test_canonical_to_slots_permutation():
    # Isaac breadth-first slot order for the 16 hand joints (canonical index in each slot):
    slot_canon = [0, 12, 4, 8, 1, 13, 5, 9, 2, 14, 6, 10, 3, 15, 7, 11]
    slot_of_canon = [slot_canon.index(k) for k in range(16)]
    q = np.arange(16, dtype=float)
    out = canonical_to_slots(q, slot_of_canon)
    assert out.tolist() == [float(c) for c in slot_canon]


def _synth_mp21_per_finger(curl: dict, side: str) -> np.ndarray:
    """Like fake_vr._synth_mp21_local but with a per-finger curl {index,middle,ring,pinky,thumb: 0..1}."""
    sx = 1.0 if side == "right" else -1.0
    kp = np.zeros((21, 3))
    finger_x = {"index": -0.03, "middle": -0.01, "ring": 0.01, "pinky": 0.03}
    base = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    for name, b in base.items():
        x = sx * finger_x[name]; kp[b] = [x, 0.0, -0.09]
        ang = curl.get(name, 0.0) * np.deg2rad(150.0)
        for k, frac in enumerate([0.33, 0.66, 1.0], start=1):
            l = 0.08 * frac; kp[b + k] = [x, -l * np.sin(ang), -0.09 - l * np.cos(ang)]
    kp[1] = [-sx * 0.03, 0.0, -0.02]
    ang = curl.get("thumb", 0.0) * np.deg2rad(90.0)
    for k, frac in enumerate([0.33, 0.66, 1.0], start=2):
        l = 0.07 * frac; kp[k] = [-sx * (0.03 + l * np.cos(ang)), -l * np.sin(ang) * 0.6, -0.02 - l * 0.6]
    return kp


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("finger,canon_flex", [("index", [1, 2, 3]), ("middle", [5, 6, 7]), ("ring", [9, 10, 11])])
def test_dexpilot_finger_identity(side, finger, canon_flex):
    """Curling ONE human finger must flex THAT Allegro finger (canonical: index 0-3, middle 4-7, ring 8-11),
    for both hands. Live 2026-08-19: the left hand had index<->ring swapped (dex_retargeting's left config
    lists fingertips for the mirrored official left URDF; GeoRT/Isaac left hands use index = joint_0)."""
    rt = AllegroDexPilot(side)
    def settle(kp):
        rt.reset(); q = None
        for _ in range(20):
            q = rt.retarget(mp21_to_operator_wrist(kp, side))
        return q
    q_open = settle(_synth_mp21_per_finger({}, side))
    q_curl = settle(_synth_mp21_per_finger({finger: 1.0}, side))
    d = q_curl - q_open
    per_finger = {"index": d[[1, 2, 3]].sum(), "middle": d[[5, 6, 7]].sum(), "ring": d[[9, 10, 11]].sum()}
    assert per_finger[finger] > 0.5, per_finger
    others = [v for k, v in per_finger.items() if k != finger]
    assert per_finger[finger] > 2.0 * max(others), per_finger


@pytest.mark.parametrize("side", ["right", "left"])
def test_thumb_angle_mapping_extension_and_opposition(side):
    """Angle-based thumb mapping: an extended (radially out) thumb -> j12 at its min (the Allegro
    thumb base cannot go lower than 0.263) and j14 near min; an opposed thumb -> j12 near max.
    IK on the thumb chain kept j14 half-flexed for every pose (live 2026-08-20)."""
    from isaac_teleop.xr_hand import thumb_angles_from_mp21
    sx = 1.0 if side == "right" else -1.0

    def base_hand():
        kp = np.zeros((21, 3))
        for f, (b, x) in {"index": (5, -0.028), "middle": (9, -0.009), "ring": (13, 0.010), "pinky": (17, 0.028)}.items():
            kp[b] = [sx * x, 0.0, -0.095]
            for k in range(1, 4):
                kp[b + k] = kp[b] + [0, 0, -0.03 * k]
        kp[1] = [-sx * 0.030, -0.012, -0.030]
        return kp

    def with_thumb(direction):
        kp = base_hand(); d = np.asarray(direction, dtype=float); d /= np.linalg.norm(d)
        for k, l in enumerate((0.045, 0.032, 0.028), start=2):
            kp[k] = kp[k - 1] + l * d
        return kp

    q_out = thumb_angles_from_mp21(with_thumb([-sx, 0.0, -0.2]), side)       # radially out, palm plane
    q_opp = thumb_angles_from_mp21(with_thumb([sx * 0.4, -0.6, -0.6]), side)  # across the palm
    assert q_out[0] < 0.35 and q_out[2] < 0.1, q_out            # j12 ~min, j14 extended
    assert q_opp[0] > 1.2, q_opp                                # j12 ~max when opposed
    # curled thumb -> j14/j15 flex
    kp = base_hand(); d1 = np.array([-sx, 0.0, -0.2]); d1 /= np.linalg.norm(d1)
    kp[2] = kp[1] + 0.045 * d1
    d2 = np.array([-sx * 0.3, -0.7, -0.6]); d2 /= np.linalg.norm(d2)
    kp[3] = kp[2] + 0.032 * d2; kp[4] = kp[3] + 0.028 * np.array([sx * 0.2, -0.8, -0.4]) / np.linalg.norm([0.2, -0.8, -0.4])
    q_curl = thumb_angles_from_mp21(kp, side)
    assert q_curl[2] > 0.8, q_curl                              # j14 flexes with MCP bend


def test_pinch_snap_brings_robot_tips_together():
    """When the human thumb tip touches the index tip, the ROBOT thumb tip must land on the robot's
    index tip (verified through FK), and a far-apart thumb must be unaffected by the snap."""
    from isaac_teleop.xr_hand import AllegroDexPilot
    rt = AllegroDexPilot("right", ema_alpha=None, filter_type="off")
    sx = 1.0
    kp = np.zeros((21, 3))
    for b, x in ((5, -0.028), (9, -0.009), (13, 0.010), (17, 0.028)):
        kp[b] = [sx * x, 0.0, -0.095]
        for k in range(1, 4):
            kp[b + k] = kp[b] + [0, 0, -0.03 * k]
    kp[1] = [-sx * 0.030, -0.012, -0.030]
    # realistic pinch: the index curls toward the thumb, the thumb chain ends at the index tip
    p = kp[5]; ang = 0.0
    for k, (l, c) in enumerate(zip((0.045, 0.025, 0.022), np.deg2rad((45, 35, 15)))):
        ang += c; p = p + l * np.array([0, -np.sin(ang), -np.cos(ang)]); kp[5 + 1 + k] = p
    tgt = kp[8]
    for k, fr in ((2, 0.4), (3, 0.75), (4, 1.0)):
        kp[k] = kp[1] + fr * (tgt - kp[1])
    q = None
    for _ in range(25):
        q = rt.retarget(kp)
    qpos = np.empty(16); qpos[rt._perm] = q
    d = np.linalg.norm(rt._fk_tip(qpos, "thumb") - rt._fk_tip(qpos, "index"))
    # allegro fingertips are ~13 mm radius: <=20 mm centre distance = physically touching
    # (brute-force workspace floor for this pose is ~12 mm)
    assert d < 0.020, f"robot tips {d*1000:.1f} mm apart after pinch snap"
    # thumb far out -> snap must not fire (thumb angles preserved)
    rt2 = AllegroDexPilot("right", ema_alpha=None, filter_type="off")
    rt3 = AllegroDexPilot("right", ema_alpha=None, filter_type="off", pinch_snap=False)
    d1 = np.array([-1.0, 0.0, -0.2]); d1 /= np.linalg.norm(d1)
    for k, l in enumerate((0.045, 0.032, 0.028), start=2):
        kp[k] = kp[k - 1] + l * d1
    qa = qb = None
    for _ in range(10):
        qa = rt2.retarget(kp); qb = rt3.retarget(kp)
    np.testing.assert_allclose(qa[12:16], qb[12:16], atol=1e-9)


def test_oneeuro_filter_smooths_and_follows():
    from isaac_teleop.xr_hand import OneEuroFilter
    f = OneEuroFilter(min_cutoff=1.5, beta=0.8)
    rng = np.random.default_rng(0)
    # noisy constant -> variance shrinks
    xs = [f(1.0 + 0.05 * rng.standard_normal(3), t=i / 60) for i in range(120)]
    assert np.std(np.asarray(xs[60:]), axis=0).max() < 0.02
    # fast ramp -> small lag
    f2 = OneEuroFilter(min_cutoff=1.5, beta=0.8)
    y = None
    for i in range(60):
        y = f2(np.full(3, i / 60 * 2.0), t=i / 60)
    assert abs(y[0] - 2.0 * 59 / 60) < 0.15

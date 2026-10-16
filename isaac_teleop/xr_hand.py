"""Quest hand tracking (WebXR 25 joints) -> Allegro 16 joint targets via dex_retargeting DexPilot."""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

# MediaPipe-21 layout from XR-25: drop the four finger metacarpals (XR 5,10,15,20).
# MP: 0 wrist | 1-4 thumb cmc/mcp/ip/tip | 5-8 index mcp/pip/dip/tip | 9-12 middle | 13-16 ring | 17-20 pinky
XR_TO_MP21: List[int] = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]

CANON_JOINT_NAMES = [f"joint_{i}.0" for i in range(16)]           # index 0-3, middle 4-7, ring 8-11, thumb 12-15
CANON_ABDUCTION = (0, 4, 8)
CANON_THUMB_OPPOSE = 12
CANON_FLEX = tuple(k for k in range(16) if k not in CANON_ABDUCTION and k != CANON_THUMB_OPPOSE)

from .paths import allegro_assets  # noqa: E402

GEORT_ASSETS = allegro_assets()   # Allegro URDFs with GeoRT/thesis-Isaac joint numbering (assets/allegro)

# From dex_retargeting.constants (copied so the frame code has no dex_retargeting import).
OPERATOR2MANO = {
    "right": np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64),
    "left": np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=np.float64),
}


def xr25_to_mp21(joints: np.ndarray) -> np.ndarray:
    """(25,4,4) XR joint poses -> (21,3) MediaPipe-ordered positions."""
    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape != (25, 4, 4):
        raise ValueError(f"expected (25,4,4), got {joints.shape}")
    return joints[XR_TO_MP21, :3, 3].copy()


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """dex_retargeting's example convention: x = wrist->middle-MCP direction (in the palm plane),
    y = palm normal, z completes (~pinky->index). Input (21,3), wrist-relative."""
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]
    x_vector = points[0] - points[2]
    points = points - np.mean(points, axis=0, keepdims=True)
    _, _, v = np.linalg.svd(points)
    normal = v[2, :]
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    return np.stack([x, normal, z], axis=1)


def mp21_to_operator_wrist(mp21: np.ndarray, side: str) -> np.ndarray:
    """(21,3) positions in any frame -> (21,3) in the operator wrist frame expected by DexPilot."""
    kp = np.asarray(mp21, dtype=np.float64)
    kp = kp - kp[0:1, :]
    frame = estimate_frame_from_hand_points(kp)
    return kp @ frame @ OPERATOR2MANO[side]


# canonical flexion joints (abductions 0/4/8 and thumb opposition 12 excluded)
CANON_FLEX_ALL = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15)
# finger flexion joints only (thumb handled separately by the angle-based thumb mapping)
CANON_FLEX_FINGERS = (1, 2, 3, 5, 6, 7, 9, 10, 11)


class OneEuroFilter:
    """One-Euro filter (Casiez et al. 2012) for vector signals: light smoothing at speed, strong
    smoothing at rest -- strictly better lag/jitter trade-off than a fixed EMA."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.8, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff), float(beta), float(d_cutoff)
        self._x = None; self._dx = None; self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self._x = None; self._dx = None; self._t = None

    def __call__(self, x, t: float):
        x = np.asarray(x, dtype=np.float64)
        if self._x is None:
            self._x = x.copy(); self._dx = np.zeros_like(x); self._t = float(t)
            return x.copy()
        dt = max(float(t) - self._t, 1e-4); self._t = float(t)
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x.copy()


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def thumb_angles_from_mp21(mp21: np.ndarray, side: str) -> np.ndarray:
    """Human thumb pose -> Allegro thumb joints (canonical j12..j15), frame-invariant.

    IK on the thumb chain fails to express extension: the Allegro thumb is much longer than a human's
    and its base (joint_12, limits [0.26, 1.40]) points backwards, so a vector optimizer keeps j14
    half-flexed for every thumb pose (measured 2026-08-20: j14 = 0.66..0.78 for a fully extended
    thumb). Glove-teleop practice maps the thumb joint-to-joint instead:
      j12 (opposition roll)   <- angle between the thumb proximal segment and the hand's radial axis
      j13 (palmar abduction)  <- elevation of the proximal segment out of the palm plane
      j14 (MCP flexion)       <- human thumb MCP flexion angle
      j15 (IP flexion)        <- human thumb IP flexion angle
    All directions are built from the 21 keypoints themselves, so any input frame works.
    """
    kp = np.asarray(mp21, dtype=np.float64)
    sgn = 1.0 if side == "right" else -1.0
    radial = _unit(kp[5] - kp[13])                      # ring MCP -> index MCP (thumb side)
    distal = _unit(kp[9] - kp[0])                       # wrist -> middle MCP
    n_palm = sgn * np.cross(radial, distal)             # out of the palm (palm-facing side)
    n_palm = _unit(n_palm)
    prox = _unit(kp[2] - kp[1])                         # thumb CMC -> MCP
    med = _unit(kp[3] - kp[2])
    dist = _unit(kp[4] - kp[3])
    # opposition: how far the thumb column has rotated away from "radially out"
    phi = float(np.arccos(np.clip(np.dot(prox, radial), -1.0, 1.0)))
    j12 = 0.263 + np.clip((phi - np.deg2rad(20.0)) / np.deg2rad(75.0), 0.0, 1.0) * (1.396 - 0.263)
    # palmar abduction: elevation toward the palm side
    el = float(np.arcsin(np.clip(np.dot(prox, n_palm), -1.0, 1.0)))
    j13 = np.clip(1.2 * el, -0.105, 1.163)
    # flexion angles between consecutive segments
    mcp = float(np.arccos(np.clip(np.dot(prox, med), -1.0, 1.0)))
    ip = float(np.arccos(np.clip(np.dot(med, dist), -1.0, 1.0)))
    j14 = np.clip(1.5 * mcp - 0.10, -0.189, 1.644)
    j15 = np.clip(1.3 * ip - 0.05, -0.162, 1.719)
    return np.array([j12, j13, j14, j15])


class AllegroDexPilot:
    """Human-hand -> Allegro retargeting; returns 16 joints in CANONICAL joint_<k>.0 order.

    profile "chain" (default): dex_retargeting `vector` type tracking wrist->PIP/DIP/tip per finger, so
    the proximal and middle joints follow the human (DexPilot's fingertip-only objective left them at
    10-50% closure on a real fist -- live 2026-08-20). profile "dexpilot": the original behavior.
    `finger_gain` scales flexion joints about their open limit after retargeting (fist ~0.9 -> ~1.0)."""

    PINCH_START = 0.055   # human thumb-fingertip distance [m] where snapping begins
    PINCH_FULL = 0.020    # ... and where the robot tips are fully snapped together

    def __init__(self, side: str, urdf_path: Optional[str] = None, cfg_path: Optional[str] = None,
                 ema_alpha: Optional[float] = None, profile: str = "chain", finger_gain: float = 1.15,
                 thumb_mode: str = "angle", pinch_snap: bool = True,
                 filter_type: str = "oneeuro"):
        import dex_retargeting
        from dex_retargeting.retargeting_config import RetargetingConfig

        if side not in ("left", "right"):
            raise ValueError(side)
        if profile not in ("chain", "dexpilot"):
            raise ValueError(profile)
        self.side = side
        # our own configs (quest3_ws/configs): dex_retargeting's left config is for the mirrored official URDF
        self.cfg_path = cfg_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                 "configs", f"allegro_hand_{side}_{profile}.yml")
        self.urdf_path = urdf_path or os.path.join(GEORT_ASSETS, f"allegro_{side}", f"allegro_hand_{side}.urdf")
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(self.urdf_path)
        cfg = RetargetingConfig.load_from_file(self.cfg_path, override={"urdf_path": self.urdf_path})
        self.retargeting = cfg.build()
        names = list(self.retargeting.joint_names)
        self._perm = [names.index(n) for n in CANON_JOINT_NAMES]
        self._human_idx = np.asarray(self.retargeting.optimizer.target_link_human_indices)
        self._lo = np.asarray(self.retargeting.joint_limits[:, 0])[self._perm]
        self._hi = np.asarray(self.retargeting.joint_limits[:, 1])[self._perm]
        self.finger_gain = float(finger_gain)
        if thumb_mode not in ("angle", "ik"):
            raise ValueError(thumb_mode)
        self.thumb_mode = thumb_mode
        self.pinch_snap = bool(pinch_snap)
        robot = self.retargeting.optimizer.robot
        self._robot = robot
        self._tip_idx = {name: robot.get_link_index(link) for name, link in
                         (("thumb", "link_15.0_tip"), ("index", "link_3.0_tip"), ("middle", "link_7.0_tip"))}
        if filter_type not in ("oneeuro", "ema", "off"):
            raise ValueError(filter_type)
        self.filter_type = filter_type
        self._oneeuro = OneEuroFilter()
        self.ema_alpha = ema_alpha
        self._ema: Optional[np.ndarray] = None

    @property
    def joint_names_canonical(self) -> List[str]:
        return list(CANON_JOINT_NAMES)

    def reset(self) -> None:
        self.retargeting.reset()
        self._ema = None
        self._oneeuro.reset()

    def _fk_tip(self, qpos: np.ndarray, tip: str) -> np.ndarray:
        self._robot.compute_forward_kinematics(qpos)
        return self._robot.get_link_pose(self._tip_idx[tip])[:3, 3].copy()

    def _snap_thumb(self, q_canon: np.ndarray, kp: np.ndarray) -> np.ndarray:
        """DexPilot-style pinch snap: when the HUMAN thumb tip approaches the index/middle tip, pull
        the ROBOT thumb tip onto that robot fingertip (small damped-least-squares IK over the 4 thumb
        joints), blended in over [PINCH_START, PINCH_FULL] so there is no popping."""
        d_pairs = [(float(np.linalg.norm(kp[4] - kp[8])), "index"), (float(np.linalg.norm(kp[4] - kp[12])), "middle")]
        d, finger = min(d_pairs)
        if d >= self.PINCH_START:
            return q_canon
        w = min(1.0, (self.PINCH_START - d) / (self.PINCH_START - self.PINCH_FULL))
        qpos = np.empty(16); qpos[self._perm] = q_canon
        p_target = self._fk_tip(qpos, finger)
        thumb_pin = [self._perm[k] for k in (12, 13, 14, 15)]
        lo_pin = np.asarray(self.retargeting.joint_limits[:, 0]); hi_pin = np.asarray(self.retargeting.joint_limits[:, 1])

        def solve(seed):
            qp = qpos.copy(); qp[thumb_pin] = np.clip(seed, lo_pin[thumb_pin], hi_pin[thumb_pin])
            for _ in range(15):
                p_t = self._fk_tip(qp, "thumb")
                err = p_target - p_t
                if np.linalg.norm(err) < 0.004:
                    break
                J = np.zeros((3, 4))
                for c, pi in enumerate(thumb_pin):
                    qp[pi] += 1e-4
                    J[:, c] = (self._fk_tip(qp, "thumb") - p_t) / 1e-4
                    qp[pi] -= 1e-4
                step = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
                qp[thumb_pin] = np.clip(qp[thumb_pin] + np.clip(step, -0.3, 0.3), lo_pin[thumb_pin], hi_pin[thumb_pin])
            return qp, float(np.linalg.norm(p_target - self._fk_tip(qp, "thumb")))
        # two seeds: the current (angle-mapped) thumb, and a canonical deep-pinch pose -- the optimum
        # often sits in a limit corner the first basin misses
        cands = [solve(qpos[thumb_pin].copy()), solve(np.array([0.3, 1.1, 1.2, 1.2]))]
        qp_best, _ = min(cands, key=lambda c: c[1])
        q_snap = qp_best[self._perm]
        out = q_canon.copy()
        out[12:16] = (1.0 - w) * q_canon[12:16] + w * q_snap[12:16]
        return out

    def retarget(self, mp21_operator: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        """(21,3) keypoints in the operator wrist frame -> (16,) canonical joint angles [rad]."""
        kp = np.asarray(mp21_operator, dtype=np.float64)
        origin, task = self._human_idx[0], self._human_idx[1]
        ref = kp[task, :] - kp[origin, :]
        q = np.asarray(self.retargeting.retarget(ref), dtype=np.float64)[self._perm]
        if self.thumb_mode == "angle":
            q[12:16] = np.clip(thumb_angles_from_mp21(kp, self.side), self._lo[12:16], self._hi[12:16])
        f = list(CANON_FLEX_FINGERS if self.thumb_mode == "angle" else CANON_FLEX_ALL)
        if self.finger_gain != 1.0:
            q[f] = np.clip(self._lo[f] + self.finger_gain * (q[f] - self._lo[f]), self._lo[f], self._hi[f])
        if self.pinch_snap:
            q = self._snap_thumb(q, kp)
        if self.filter_type == "oneeuro":
            import time as _time
            q = self._oneeuro(q, _time.monotonic() if t is None else t)
        elif self.filter_type == "ema" and self.ema_alpha is not None:
            self._ema = q if self._ema is None else self.ema_alpha * q + (1.0 - self.ema_alpha) * self._ema
            q = self._ema
        return q


def canonical_to_slots(q_canon: np.ndarray, slot_of_canon: Sequence[int]) -> np.ndarray:
    """Reorder canonical joint_<k>.0 values into Isaac's 16-slot hand layout.
    slot_of_canon[k] = slot holding canonical joint k (env.right/left_hand_slot_of_canon)."""
    q_canon = np.asarray(q_canon, dtype=np.float64)
    out = np.empty(16, dtype=np.float64)
    out[np.asarray(list(slot_of_canon), dtype=int)] = q_canon
    return out

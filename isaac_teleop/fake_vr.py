"""Synthetic VR tracking for headless tests: a standing head and two hands doing a slow reach + open/close."""
from __future__ import annotations

import pickle
from typing import Optional

import numpy as np

from quest_bridge.messages import VRFrame, empty_hand_state
from .xr_hand import XR_TO_MP21


def _synth_mp21_local(close: float, side: str) -> np.ndarray:
    """21 keypoints in a hand-local frame: wrist at origin, palm in x-z plane, fingers along -z
    (WebXR: forward is -z), palm facing down (-y). close: 0 open .. 1 fist."""
    sx = 1.0 if side == "right" else -1.0
    kp = np.zeros((21, 3))
    finger_x = {"index": -0.03, "middle": -0.01, "ring": 0.01, "pinky": 0.03}
    base = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    for name, b in base.items():
        x = sx * finger_x[name]
        kp[b] = [x, 0.0, -0.09]
        for k, frac in enumerate([0.33, 0.66, 1.0], start=1):
            l = 0.08 * frac
            ang = close * np.deg2rad(150.0)
            kp[b + k] = [x, -l * np.sin(ang), -0.09 - l * np.cos(ang)]
    kp[1] = [-sx * 0.03, 0.0, -0.02]
    for k, frac in enumerate([0.33, 0.66, 1.0], start=2):
        l = 0.07 * frac
        ang = close * np.deg2rad(90.0)
        kp[k] = [-sx * (0.03 + l * np.cos(ang)), -l * np.sin(ang) * 0.6, -0.02 - l * 0.6]
    return kp


def synth_hand_xr25(T_wrist: np.ndarray, close: float, side: str) -> np.ndarray:
    """(25,4,4) XR joints: all rotations = wrist rotation, positions from the synthetic hand."""
    kp = _synth_mp21_local(float(np.clip(close, 0.0, 1.0)), side)
    xr = np.tile(np.asarray(T_wrist, dtype=np.float64), (25, 1, 1))
    R, p = T_wrist[:3, :3], T_wrist[:3, 3]
    for mp_i, xr_i in enumerate(XR_TO_MP21):          # XR_TO_MP21[mp_i] = xr index
        xr[xr_i, :3, 3] = p + R @ kp[mp_i]
    for meta, mcp in [(5, 5), (10, 9), (15, 13), (20, 17)]:
        xr[meta, :3, 3] = p + R @ (0.5 * kp[mcp])
    return xr


class FakeVRSource:
    """Same interface as ZmqVRSource: poll(t) -> VRFrame|None, stale(t) -> bool, close()."""

    def __init__(self, hz: float = 60.0, head_height: float = 1.65):
        self.hz = float(hz)
        self.head_height = float(head_height)
        self._seq = 0
        self._last_t: Optional[float] = None

    def poll(self, t: float) -> Optional[VRFrame]:
        if self._last_t is not None and (t - self._last_t) < 1.0 / self.hz:
            return None
        self._last_t = t
        self._seq += 1
        head = np.eye(4); head[:3, 3] = [0.0, self.head_height, 0.0]
        head[:3, :3] = _rot_x(-0.35)                                # look slightly down
        w = 2.0 * np.pi / 6.0                                        # 6 s reach cycle
        close = 0.5 - 0.5 * np.cos(2.0 * np.pi * t / 4.0)           # 4 s open/close cycle
        hands = {}
        for side, sx in (("left", -1.0), ("right", 1.0)):
            T = np.eye(4)
            T[:3, 3] = [sx * 0.22 + 0.05 * np.sin(w * t), 1.15 + 0.05 * np.sin(2 * w * t), -0.40 - 0.10 * np.sin(w * t)]
            T[:3, :3] = _rot_x(-np.pi / 2)                          # palm down, fingers forward (-z)
            hands[side] = synth_hand_xr25(T, close, side)
        st = empty_hand_state(); st["pinch_value"] = float(close)
        return VRFrame(t_wall=t, seq=self._seq, head=head, left=hands["left"], right=hands["right"],
                       left_state=dict(st), right_state=dict(st))

    def stale(self, t: float) -> bool:
        return False

    def close(self) -> None:
        pass


class ReplayVRSource:
    """Re-feed the `vr` arrays of a recorded episode pickle at their recorded wall-clock spacing."""

    def __init__(self, episode_pickle: str, loop: bool = True):
        with open(episode_pickle, "rb") as f:
            ep = pickle.load(f)
        self.vr = ep["vr"]; self.t = ep["t_wall"] - ep["t_wall"][0]
        self.n = int(ep["T"]); self.loop = loop
        self._i = 0; self._t0: Optional[float] = None; self._seq = 0

    def poll(self, t: float) -> Optional[VRFrame]:
        if self._t0 is None:
            self._t0 = t
        rel = t - self._t0
        if self._i >= self.n:
            if not self.loop:
                return None
            self._i = 0; self._t0 = t; rel = 0.0
        if rel < self.t[self._i]:
            return None
        i = self._i; self._i += 1; self._seq += 1

        def _hand(a):
            return None if np.isnan(a).any() else a

        def _state(v):
            return {"pinch": bool(v[0] > 0.5), "squeeze": bool(v[1] > 0.5), "pinch_value": float(v[2]), "squeeze_value": float(v[3])}

        head = self.vr["head"][i]
        return VRFrame(t_wall=t, seq=self._seq, head=None if np.isnan(head).any() else head,
                       left=_hand(self.vr["left"][i]), right=_hand(self.vr["right"][i]),
                       left_state=_state(self.vr["left_state"][i]), right_state=_state(self.vr["right_state"][i]))

    def stale(self, t: float) -> bool:
        return False

    def close(self) -> None:
        pass


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

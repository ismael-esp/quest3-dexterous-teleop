"""Adapter: Isaac Lab OpenXRDevice raw data (26-joint dicts, world frame) -> the bridge pipeline's
formats (xr25 (25,4,4) arrays, wrist pose, staleness). Pure numpy, unit-testable without Isaac.

OpenXRDevice._get_raw_data() returns, per hand, ``{joint_name: (x, y, z, qw, qx, qy, qz)}`` with the
26 OpenXR joint names (HAND_JOINT_NAMES: 'palm' first, then 'wrist', ...). Poses are already in the
Isaac world frame (mapped through the XR anchor). Untracked joints keep their previous value and start
at all-zeros, so validity is detected by (a) an all-zero wrist quat==identity/pos==0 never-tracked
state and (b) the pose not changing for `stale_after` seconds.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# OpenXR joint order without 'palm' == the WebXR 25-joint order used by xr_hand / the recorder.
XR25_JOINT_NAMES = [
    "wrist",
    "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip",
    "index_metacarpal", "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_metacarpal", "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_metacarpal", "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "little_metacarpal", "little_proximal", "little_intermediate", "little_distal", "little_tip",
]


def pose7_to_T(pose7) -> np.ndarray:
    """(x, y, z, qw, qx, qy, qz) -> (4,4)."""
    from .frames import quat_wxyz_to_mat

    p = np.asarray(pose7, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_mat(p[3:7])
    T[:3, 3] = p[:3]
    return T


def joints_to_xr25(joints: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """26-joint dict -> (25,4,4) world-frame array in WebXR order, or None if never tracked
    (wrist still at the all-zero default pose)."""
    w = np.asarray(joints.get("wrist", np.zeros(7)), dtype=np.float64)
    if np.allclose(w[:3], 0.0):
        return None
    out = np.empty((25, 4, 4))
    for i, name in enumerate(XR25_JOINT_NAMES):
        out[i] = pose7_to_T(joints.get(name, w))
    return out


class OpenXRHandsSource:
    """Wraps OpenXRDevice raw data into the bridge's VR-source contract (per-hand xr25 + staleness).

    ``update(raw, t)`` with ``raw = device._get_raw_data()`` each control step; then ``hand(side)``
    gives the (25,4,4) array or None, ``head()`` the (4,4) head pose or None, ``fresh(side)`` whether
    the hand pose changed within `stale_after` seconds (a frozen pose means tracking was lost --
    OpenXRDevice repeats the last value).
    """

    def __init__(self, stale_after: float = 0.5, live_after: int = 10):
        self.stale_after = float(stale_after)
        self._live_after = int(live_after)
        self._xr25 = {"left": None, "right": None}
        self._last_change_t = {"left": -1e9, "right": -1e9}
        self._last_wrist = {"left": None, "right": None}
        self._head: Optional[np.ndarray] = None
        self._head_prev: Optional[np.ndarray] = None
        self._head_changes = 0
        self._head_change_t = -1e9

    def update(self, raw_data: dict, t: float) -> None:
        # keys: OpenXRDevice.TrackingTarget.{HAND_LEFT, HAND_RIGHT, HEAD}; accept any enum-like dict
        items: Dict[str, object] = {}
        for k, v in raw_data.items():
            name = getattr(k, "name", str(k)).lower()
            items[name] = v
        for side, key in (("left", "hand_left"), ("right", "hand_right")):
            joints = items.get(key)
            if not isinstance(joints, dict):
                continue
            xr25 = joints_to_xr25(joints)
            self._xr25[side] = xr25
            if xr25 is not None:
                wrist = xr25[0, :3, 3]
                prev = self._last_wrist[side]
                if prev is None or not np.allclose(wrist, prev, atol=1e-9):
                    self._last_change_t[side] = t
                    self._last_wrist[side] = wrist.copy()
        head = items.get("head")
        if head is not None:
            h = np.asarray(head, dtype=np.float64)
            self._head = None if np.allclose(h[:3], 0.0) else pose7_to_T(h)
            if self._head is not None:
                if self._head_prev is not None and not np.allclose(self._head, self._head_prev, atol=1e-9):
                    self._head_changes += 1
                    self._head_change_t = t
                elif (t - self._head_change_t) > self.stale_after:
                    self._head_changes = 0          # frozen for a while (no headset / disconnected)
                self._head_prev = self._head.copy()

    def hand(self, side: str) -> Optional[np.ndarray]:
        """(25,4,4) world-frame joints, or None if never tracked. Check ``fresh`` for staleness."""
        return self._xr25[side]

    def head(self) -> Optional[np.ndarray]:
        return self._head

    def head_live(self, t: float) -> bool:
        """True when head tracking is real: the pose changed on >= live_after recent samples and
        within stale_after of t. The CloudXR runtime feeds a bit-static default head pose when no
        headset is connected -- a worn headset jitters every sample -- so a static pose is not live."""
        return (self._head is not None and self._head_changes >= self._live_after
                and (t - self._head_change_t) <= self.stale_after)

    def fresh(self, side: str, t: float) -> bool:
        """True if the hand pose changed within `stale_after` of time t."""
        return self._xr25[side] is not None and (t - self._last_change_t[side]) <= self.stale_after

    def last_change(self, side: str) -> float:
        return self._last_change_t[side]

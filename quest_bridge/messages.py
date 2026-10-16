"""Wire format shared by the Vuer bridge process and the Isaac teleop process.

Only numpy + msgpack here: this file is imported by both python environments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import msgpack
import numpy as np

# WebXR XRHand joint order (25). See MDN XRHand / vuer Hands docs.
XR_JOINT_NAMES = [
    "wrist",
    "thumb-metacarpal", "thumb-phalanx-proximal", "thumb-phalanx-distal", "thumb-tip",
    "index-finger-metacarpal", "index-finger-phalanx-proximal", "index-finger-phalanx-intermediate",
    "index-finger-phalanx-distal", "index-finger-tip",
    "middle-finger-metacarpal", "middle-finger-phalanx-proximal", "middle-finger-phalanx-intermediate",
    "middle-finger-phalanx-distal", "middle-finger-tip",
    "ring-finger-metacarpal", "ring-finger-phalanx-proximal", "ring-finger-phalanx-intermediate",
    "ring-finger-phalanx-distal", "ring-finger-tip",
    "pinky-finger-metacarpal", "pinky-finger-phalanx-proximal", "pinky-finger-phalanx-intermediate",
    "pinky-finger-phalanx-distal", "pinky-finger-tip",
]
XR_WRIST = 0
XR_THUMB_TIP = 4
XR_INDEX_TIP = 9
XR_MIDDLE_TIP = 14
XR_RING_TIP = 19
XR_PINKY_TIP = 24
NUM_XR_JOINTS = 25


def empty_hand_state() -> dict:
    return {"pinch": False, "squeeze": False, "pinch_value": 0.0, "squeeze_value": 0.0}


def matrix_from_webxr(flat16) -> np.ndarray:
    """WebXR/WebGL 16-float column-major matrix -> (4,4) row-major float64."""
    arr = np.asarray(flat16, dtype=np.float64)
    if arr.size != 16:
        raise ValueError(f"expected 16 floats, got {arr.size}")
    return arr.reshape(4, 4, order="F")


def hand_from_webxr(flat400) -> np.ndarray:
    """400 floats (25 column-major 4x4 blocks) -> (25,4,4) row-major."""
    arr = np.asarray(flat400, dtype=np.float64)
    if arr.size != 16 * NUM_XR_JOINTS:
        raise ValueError(f"expected {16 * NUM_XR_JOINTS} floats, got {arr.size}")
    blocks = arr.reshape(NUM_XR_JOINTS, 4, 4)          # each block still column-major
    return np.ascontiguousarray(np.transpose(blocks, (0, 2, 1)))


# ---- msgpack helpers -------------------------------------------------------

def _pack_arr(a: Optional[np.ndarray]):
    if a is None:
        return None
    a = np.ascontiguousarray(a, dtype=np.float32)
    return {"shape": list(a.shape), "data": a.tobytes()}


def _unpack_arr(d) -> Optional[np.ndarray]:
    if d is None:
        return None
    return np.frombuffer(d["data"], dtype=np.float32).reshape(d["shape"]).astype(np.float64)


@dataclass
class VRFrame:
    """One tracking sample from the headset. Poses are in the WebXR reference space."""
    t_wall: float
    seq: int
    head: Optional[np.ndarray] = None            # (4,4) or None
    left: Optional[np.ndarray] = None            # (25,4,4) or None
    right: Optional[np.ndarray] = None           # (25,4,4) or None
    left_state: dict = field(default_factory=empty_hand_state)
    right_state: dict = field(default_factory=empty_hand_state)

    def to_bytes(self) -> bytes:
        return msgpack.packb({
            "t": float(self.t_wall), "seq": int(self.seq),
            "head": _pack_arr(self.head), "left": _pack_arr(self.left), "right": _pack_arr(self.right),
            "ls": dict(self.left_state), "rs": dict(self.right_state),
        }, use_bin_type=True)

    @classmethod
    def from_bytes(cls, b: bytes) -> "VRFrame":
        d = msgpack.unpackb(b, raw=False)
        return cls(
            t_wall=float(d["t"]), seq=int(d["seq"]),
            head=_unpack_arr(d["head"]), left=_unpack_arr(d["left"]), right=_unpack_arr(d["right"]),
            left_state=dict(d["ls"]), right_state=dict(d["rs"]),
        )


@dataclass
class StereoFrame:
    """One stereo image pair (JPEG bytes) + HUD text, Isaac -> headset."""
    seq: int
    t_wall: float
    left_jpeg: bytes
    right_jpeg: bytes
    hud: str = ""

    def to_bytes(self) -> bytes:
        return msgpack.packb({
            "seq": int(self.seq), "t": float(self.t_wall),
            "l": bytes(self.left_jpeg), "r": bytes(self.right_jpeg), "hud": str(self.hud),
        }, use_bin_type=True)

    @classmethod
    def from_bytes(cls, b: bytes) -> "StereoFrame":
        d = msgpack.unpackb(b, raw=False)
        return cls(seq=int(d["seq"]), t_wall=float(d["t"]), left_jpeg=d["l"], right_jpeg=d["r"], hud=d["hud"])

"""Parse vuer client event payloads into numpy. No vuer import (unit-testable anywhere)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .messages import NUM_XR_JOINTS, empty_hand_state, hand_from_webxr, matrix_from_webxr

# WebXR body-tracking joint order (XRBodyJoint): hips, spine-lower, spine-middle,
# spine-upper, chest, neck, head, ...  (same table as robotis vr_publisher_sh5.py)
BODY_HEAD_INDEX = 6

# Axis conventions of the incoming head pose, expressed as the columns (right, up, back) of the
# camera-convention frame in the source frame's coordinates:  R_cam = R_src @ head_axes_matrix(name).
#   camera: WebXR/three.js camera (looks -z, up +y, right +x) -- CAMERA_MOVE, identity.
#   body:   WebXR body-tracking 'head' joint as seen on the Quest 3 / robotis vr_publisher_sh5.py:
#           +Y forward, +Z right, +X down  ->  right=+z, up=-x, back=-y.
HEAD_AXES = {
    "camera": np.eye(3),
    "body": np.array([[0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [1.0, 0.0, 0.0]]),
}


def head_axes_matrix(name: str) -> np.ndarray:
    try:
        return HEAD_AXES[name]
    except KeyError:
        raise ValueError(f"unknown head axes convention {name!r}; choose from {sorted(HEAD_AXES)}")


def _floats(v) -> Optional[np.ndarray]:
    """Flat float64 array from a client value, or None if it is not a numeric sequence.

    The vuer client serialises with msgpackr, which encodes a JS ``undefined`` (e.g. an untracked
    hand's ``poses``) as msgpack ext type 0 / ``b'\\x00'`` -> ``msgpack.ExtType(0, b'\\x00')`` on
    the Python side; ``np.asarray`` on that raises ``could not convert string to float``.
    ``bytes`` (a typed array sent as bin) is decoded as float32."""
    if v is None or isinstance(v, (str, dict, bool)):
        return None
    if isinstance(v, (bytes, bytearray, memoryview)):
        b = bytes(v)
        return np.frombuffer(b, dtype=np.float32).astype(np.float64) if len(b) % 4 == 0 else None
    if isinstance(v, tuple) and hasattr(v, "code") and hasattr(v, "data"):   # msgpack.ExtType
        return None
    try:
        return np.asarray(v, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None


def _num(v, default=0.0) -> float:
    if isinstance(v, bool) or v is None or not isinstance(v, (int, float)):
        return float(default)
    return float(v)


def _state(d) -> dict:
    s = empty_hand_state()
    if isinstance(d, dict):
        s["pinch"] = d.get("pinch", False) is True
        s["squeeze"] = d.get("squeeze", False) is True
        s["pinch_value"] = _num(d.get("pinchValue"))
        s["squeeze_value"] = _num(d.get("squeezeValue"))
    return s


def _hand(v) -> Optional[np.ndarray]:
    arr = _floats(v)
    if arr is None or arr.size != 16 * NUM_XR_JOINTS or not np.all(np.isfinite(arr)):
        return None
    return hand_from_webxr(arr)


def parse_hand_value(value) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict, dict]:
    """HAND_MOVE payload -> (left (25,4,4)|None, right (25,4,4)|None, left_state, right_state)."""
    if not isinstance(value, dict):
        return None, None, empty_hand_state(), empty_hand_state()
    return (
        _hand(value.get("left")),
        _hand(value.get("right")),
        _state(value.get("leftState")),
        _state(value.get("rightState")),
    )


def parse_head_value(etype: str, value, head_axes: str = "body") -> Optional[np.ndarray]:
    """CAMERA_MOVE ({'camera': {'matrix': [16]}} or {'matrix': [16]}) or BODY_MOVE ({'body': [...]})
    -> (4,4) head pose in the WebXR reference space, **camera convention** (looks -z, up +y), or None.
    `head_axes` names the convention of the BODY_MOVE head joint (see HEAD_AXES)."""
    if not isinstance(value, dict):
        return None
    if etype == "CAMERA_MOVE":
        cam = value.get("camera", value)
        arr = _floats(cam.get("matrix")) if isinstance(cam, dict) else None
        if arr is None or arr.size != 16 or not np.all(np.isfinite(arr)):
            return None
        return matrix_from_webxr(arr)
    if etype == "BODY_MOVE":
        arr = _floats(value.get("body"))
        s = 16 * BODY_HEAD_INDEX
        if arr is None or arr.size < s + 16:
            return None
        T = matrix_from_webxr(arr[s:s + 16])
        if not np.all(np.isfinite(T)) or abs(np.linalg.det(T[:3, :3])) < 1e-6:
            return None
        T = T.copy()
        T[:3, :3] = T[:3, :3] @ head_axes_matrix(head_axes)
        return T
    return None

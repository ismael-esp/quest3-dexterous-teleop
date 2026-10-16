"""Constant frames on the Allegro side of the UR wrist (pure numpy, importable without Isaac).

Isaac merges fixed joints at import, so `*_palm_link` and `*_tip` are NOT rigid bodies in the sim;
the wrist_3 body (index 6, what `DualArmTeleop` targets) is what the env exposes. BODex expresses
every grasp at `ra_palm_link` / `la_palm_link`, so this constant is the bridge between the two.

Chains (Isaac URDFs, verified 2026-09-09 to agree with the BODex URDFs to 1.5 mm / <0.1 deg):
  right  assets/ra_ur3_allegro_isaac_.urdf : wrist_3 -> flange (rpy 0,-pi/2,-pi/2)
                                             -> palm (rpy 0,-pi/2,3.14 ; xyz 0.095,-0.0015,0.0095)
  left   assets/la_ur3e_allegro_isaac_.urdf: wrist_3 -> flange (rpy pi/2,-pi/2,0)
                                             -> palm (rpy 0,pi/2,0 ; xyz 0.095,-0.0015,-0.0095)
Quaternions are WXYZ (Isaac convention). Fingertip frames are the last finger links + 26.7 mm
along the link z axis (`*_link_{3,7,11,15}_0_tip` joints in the same URDFs).
"""
from __future__ import annotations

import numpy as np

WRIST3_TO_PALM_POS = {
    "right": np.array([-0.0015, 0.0095, 0.095], dtype=np.float64),
    "left": np.array([0.0015, 0.0095, 0.095], dtype=np.float64),
}
def _unit(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


# 6-decimal URDF-derived values, re-normalised so compositions invert exactly.
WRIST3_TO_PALM_QUAT = {
    "right": _unit([0.707107, -0.000563, -0.000563, 0.707107]),
    "left": _unit([0.707107, 0.0, 0.0, 0.707107]),
}
FINGERTIP_LINKS = {
    "right": ["ra_link_3_0", "ra_link_7_0", "ra_link_11_0", "ra_link_15_0"],
    "left": ["la_link_3_0", "la_link_7_0", "la_link_11_0", "la_link_15_0"],
}
FINGERTIP_OFFSET = np.array([0.0, 0.0, 0.0267], dtype=np.float64)   # index, middle, ring, thumb


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4) WXYZ."""
    q = np.asarray(q, dtype=np.float64); v = np.asarray(v, dtype=np.float64)
    w = q[..., :1]
    u = q[..., 1:]
    uv = np.cross(u, v)
    return v + 2.0 * w * uv + 2.0 * np.cross(u, uv)


def compose(p1, q1, p2, q2):
    """T1 * T2 for (pos, quat) pairs: apply T2 in T1's frame."""
    return np.asarray(p1, dtype=np.float64) + quat_apply(q1, p2), quat_mul(q1, q2)


def invert(p, q):
    qi = quat_conj(q)
    return -quat_apply(qi, p), qi


def palm_from_wrist3(p_wrist, q_wrist, side: str):
    return compose(p_wrist, q_wrist, WRIST3_TO_PALM_POS[side], WRIST3_TO_PALM_QUAT[side])


def wrist3_from_palm(p_palm, q_palm, side: str):
    pi, qi = invert(WRIST3_TO_PALM_POS[side], WRIST3_TO_PALM_QUAT[side])
    return compose(p_palm, q_palm, pi, qi)


def fingertips_from_links(p_links: np.ndarray, q_links: np.ndarray) -> np.ndarray:
    """(4,3) fingertip positions from the last finger links' (4,3) positions and (4,4) WXYZ quats."""
    p_links = np.asarray(p_links, dtype=np.float64)
    off = np.broadcast_to(FINGERTIP_OFFSET, p_links.shape)
    return p_links + quat_apply(np.asarray(q_links, dtype=np.float64), off)

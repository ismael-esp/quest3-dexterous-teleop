"""Analytic wrap grasps for the handover rod.

Purpose: unblock the 22-DOF demo pipeline without depending on BODex onboarding. The
handover object is a plain cylinder (r=0.027 m, L=0.30 m), simple enough that a wrap
grasp can be placed geometrically. Entries produced here carry `source="analytic"` so
they are never mistaken for synthesised grasps.

Two properties matter and are enforced here rather than left to the caller:

1.  **End exclusivity.** The giver takes one end, the receiver the opposite end. If both
    hands aim for the middle the handover is unsolvable, so `axis_coord` sign is assigned
    per role and `GraspEntry.opposes()` can verify a pair.
2.  **Approach diversity.** Grasps are swept in yaw around the rod axis, so the library
    offers many distinct approaches rather than a single choreographed pose.

The joint values in `WrapProfile` are a documented heuristic, NOT a solved grasp: they
have not been validated for force closure. They exist to make the pipeline runnable and
are the first thing BODex should replace.

Palm frame convention (documented because it is otherwise unrecoverable):
    palm +Z  -> points from the palm toward the rod axis (approach direction)
    palm +X  -> aligned with the rod's long axis (rod lies across the knuckles)
    palm +Y  -> completes the right-handed frame (finger curl direction)
If the URDF's palm link disagrees, fix it in ONE place: `PALM_FRAME_CORRECTION`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import NUM_HAND_JOINTS, GraspEntry, GraspRole

# Allegro joint layout, mirroring hw/hardware_env.py.
ABDUCTION_IDS = (0, 4, 8)                  # finger spread; 0 = neutral
FLEX_IDS = (1, 2, 3, 5, 6, 7, 9, 10, 11)   # index / middle / ring flexion
THUMB_OPPOSE_ID = 12
THUMB_FLEX_IDS = (13, 14, 15)

# Applied after the geometric frame is built, in case the palm link's axes differ.
# Identity until validated in sim against the actual Allegro palm frame.
PALM_FRAME_CORRECTION = np.array([1.0, 0.0, 0.0, 0.0])  # WXYZ identity


# The scalar-grip path closes every flexion joint to this angle, and that closure is
# empirically validated -- it is what the 89.5% teacher and the 80% hardware student
# were trained on. A dexterous grasp should therefore be a PERTURBATION around a
# known-good closure, not an independent geometric guess.
#
# Learned the hard way: the first version of this profile curled to ~0.63 rad (40% of
# the reference). Every env latched LEFT_GRASPED on the stage machine's proximity test
# while the physics never actually gripped the rod -- 0/25 demos. Keep firm closure as
# the anchor and express dexterity as per-joint variation ON TOP of it.
REFERENCE_CLOSE_ANGLE = 1.57


@dataclass(frozen=True)
class WrapProfile:
    """Allegro posture for wrapping a cylinder, anchored on the proven close angle.

    Values are radians. Defaults target a ~27 mm-radius rod: firm enough to actually
    hold it, with the joints differing from one another so the hand is not a gripper.
    `from_radius` opens the hand slightly for a fatter object.
    """

    proximal_flex: float = 1.30
    middle_flex: float = 1.45
    distal_flex: float = 1.25
    thumb_oppose: float = 1.20
    thumb_flex: float = 1.35
    abduction: float = 0.0
    # Fraction of the grasp flexion held during approach. The hand is pre-shaped (thumb
    # opposed, fingers spread) but not curled, so closing squeezes ONTO the rod instead
    # of sweeping through it. Thumb opposition is NOT scaled -- it must already be in
    # place before the fingers close, exactly as a human pre-shapes.
    pregrasp_flex_ratio: float = 0.25

    @staticmethod
    def from_radius(radius: float) -> "WrapProfile":
        """Open the hand slightly as the object gets fatter, around the proven closure.

        k = 1.0 at the 27 mm reference rod. Bounded so a large object cannot open the
        hand below a grip that holds, nor drive joints past the Allegro's 1.57 limit.
        """
        k = float(np.clip(0.027 / max(radius, 1e-4), 0.70, 1.10))
        hi = REFERENCE_CLOSE_ANGLE
        return WrapProfile(
            proximal_flex=float(np.clip(1.30 * k, 0.70, hi)),
            middle_flex=float(np.clip(1.45 * k, 0.80, hi)),
            distal_flex=float(np.clip(1.25 * k, 0.70, hi)),
            thumb_oppose=float(np.clip(1.20 * k, 0.36, hi)),  # URDF floor: joint_12_0 >= 0.36
            thumb_flex=float(np.clip(1.35 * k, 0.70, hi)),
        )

    def to_pregrasp_joints(self, grasp_joints: np.ndarray) -> np.ndarray:
        """Pre-shaped hand: same configuration, fingers uncurled.

        Flexion (and thumb flexion) scale toward open; thumb OPPOSITION and abduction are
        preserved, because those position the fingers around the object and must be set
        before closing begins.
        """
        q = np.asarray(grasp_joints, dtype=np.float64).copy()
        r = float(np.clip(self.pregrasp_flex_ratio, 0.0, 1.0))
        for i in FLEX_IDS:
            q[i] = float(np.clip(q[i] * r, 0.0, REFERENCE_CLOSE_ANGLE))
        for i in THUMB_FLEX_IDS:
            q[i] = float(np.clip(q[i] * r, 0.0, REFERENCE_CLOSE_ANGLE))
        # q[THUMB_OPPOSE_ID] and abduction intentionally left at their grasp values.
        return q

    def to_joints(self) -> np.ndarray:
        """Expand the profile into the 16-vector the action space expects."""
        q = np.zeros(NUM_HAND_JOINTS, dtype=np.float64)
        for i in ABDUCTION_IDS:
            q[i] = self.abduction
        # FLEX_IDS run (proximal, middle, distal) per finger, three fingers.
        for n, i in enumerate(FLEX_IDS):
            q[i] = (self.proximal_flex, self.middle_flex, self.distal_flex)[n % 3]
        q[THUMB_OPPOSE_ID] = self.thumb_oppose
        for i in THUMB_FLEX_IDS:
            q[i] = self.thumb_flex
        return q


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, WXYZ."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> WXYZ quaternion (Shepperd's method, branch on trace)."""
    tr = float(np.trace(m))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                         (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                         (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 - m[0, 0] + m[1, 1] - m[2, 2]) * 2.0
        return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                         0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = np.sqrt(1.0 - m[0, 0] - m[1, 1] + m[2, 2]) * 2.0
    return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                     (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def make_cylinder_library(
    radius: float = 0.027,
    height: float = 0.30,
    n_per_role: int = 24,
    end_fraction: float = 0.70,
    palm_clearance: float = 0.045,
    profile: WrapProfile | None = None,
    joint_jitter: float = 0.12,
    seed: int = 0,
) -> list[GraspEntry]:
    """Build an object-frame wrap-grasp library for a cylinder.

    Args:
        radius, height: cylinder dimensions, metres. Long axis is object-local +Z.
        n_per_role: yaw samples per role (giver / receiver).
        end_fraction: where along the half-length to grip, as a fraction. 0.70 of
            half-length (=0.105 m for the 0.3 m rod) sits well onto one end while
            leaving margin before the cap.
        palm_clearance: distance from the rod SURFACE to the palm origin, metres.
        profile: hand posture; derived from `radius` when omitted.

    Returns:
        Entries for both roles. Givers sit at +end_fraction*height/2 along the axis,
        receivers at the negative mirror, so every giver/receiver pair `opposes()`.
    """
    if not 0.0 < end_fraction <= 1.0:
        raise ValueError(f"end_fraction must be in (0, 1], got {end_fraction}")
    if n_per_role < 1:
        raise ValueError(f"n_per_role must be >= 1, got {n_per_role}")

    prof = profile or WrapProfile.from_radius(radius)
    base_joints = prof.to_joints()
    axis_offset = end_fraction * (height / 2.0)
    palm_radius = radius + palm_clearance
    rng = np.random.default_rng(seed)

    # Per-entry posture jitter. Without it every entry carries the identical hand pose,
    # so BC would be cloning a single canned grasp rather than a distribution over
    # viable ones -- brittle, and it would not survive contact with a real object.
    # Abduction stays pinned at 0 and the thumb keeps its URDF opposition floor.
    def _jittered() -> np.ndarray:
        q = base_joints.copy()
        if joint_jitter > 0.0:
            noise = rng.normal(0.0, joint_jitter, size=NUM_HAND_JOINTS)
            for i in FLEX_IDS:
                q[i] = float(np.clip(q[i] + noise[i], 0.0, 1.57))
            for i in THUMB_FLEX_IDS:
                q[i] = float(np.clip(q[i] + noise[i], 0.0, 1.57))
            q[THUMB_OPPOSE_ID] = float(
                np.clip(q[THUMB_OPPOSE_ID] + noise[THUMB_OPPOSE_ID], 0.36, 1.57)
            )
        for i in ABDUCTION_IDS:
            q[i] = prof.abduction
        return q

    entries: list[GraspEntry] = []
    roles: tuple[tuple[GraspRole, float], ...] = (("giver", +1.0), ("receiver", -1.0))

    for role, sign in roles:
        for k in range(n_per_role):
            yaw = 2.0 * np.pi * k / n_per_role
            radial = np.array([np.cos(yaw), np.sin(yaw), 0.0])

            palm_pos = radial * palm_radius + np.array([0.0, 0.0, sign * axis_offset])

            # Approach points inward, toward the rod axis.
            palm_z = -radial
            # Rod axis lies across the knuckles.
            palm_x = np.array([0.0, 0.0, 1.0])
            palm_y = np.cross(palm_z, palm_x)
            palm_y /= np.linalg.norm(palm_y)
            palm_x = np.cross(palm_y, palm_z)  # re-orthogonalise

            rot = np.column_stack([palm_x, palm_y, palm_z])
            quat = _quat_mul(_quat_from_matrix(rot), PALM_FRAME_CORRECTION)

            grasp_q = _jittered()
            entries.append(
                GraspEntry(
                    role=role,
                    palm_pos=palm_pos,
                    palm_quat=quat,
                    hand_joints=grasp_q,
                    pregrasp_joints=prof.to_pregrasp_joints(grasp_q),
                    axis_coord=float(sign * axis_offset),
                    # Heuristic: no force-closure metric is computed. Flat score keeps
                    # ranking honest -- these are placeholders, not solved grasps.
                    quality=0.0,
                    source="analytic",
                    meta={"yaw": float(yaw), "radius": float(radius), "height": float(height)},
                )
            )
    return entries

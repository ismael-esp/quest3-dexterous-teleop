"""Grasp-library data types.

One `GraspEntry` describes a single grasp **in the object's local frame**. Storing in
the object frame is what makes the library cheap: the task's object-pose randomisation
(+/-5 cm, +/-10 deg) becomes a rigid transform of an existing entry plus a fresh arm IK,
instead of a ~4-5 s BODex synthesis per episode.

Deliberately free of Isaac and torch imports so it can be unit-tested without a
simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Which side of the handover an entry is for.
#   giver    - lifts the rod off the table, must take ONE END
#   receiver - takes it mid-air from the giver, must take the OPPOSITE END
GraspRole = Literal["giver", "receiver"]

NUM_HAND_JOINTS = 16


@dataclass
class GraspEntry:
    """A single object-frame grasp.

    Attributes:
        role: which side of the handover this grasp serves.
        palm_pos: (3,) palm position in the OBJECT frame, metres.
        palm_quat: (4,) palm orientation in the OBJECT frame, WXYZ (Isaac convention).
        pregrasp_joints: (16,) the PRE-SHAPED hand held during approach -- thumb already
            opposed and fingers spread into the grasp's configuration, but not yet curled
            in. Closing from here squeezes ONTO the object; closing from a flat open hand
            instead sweeps the fingers THROUGH it. Mirrors BODex's pregrasp IK stage.
        hand_joints: (16,) Allegro joint targets at full grasp, radians, robot joint order.
        axis_coord: signed coordinate along the object's long axis where this grasp sits,
            in the object frame (metres). Sign identifies the end: giver and receiver of
            the same pair must have OPPOSITE signs, which is what keeps one end free.
        quality: higher is better. Analytic entries use a heuristic score; BODex entries
            carry the solver's engagement/force-closure metric.
        source: provenance, e.g. "analytic" or "bodex". Kept so a mixed library stays
            auditable and so analytic fallbacks are never silently mistaken for
            synthesised grasps.
    """

    role: GraspRole
    palm_pos: np.ndarray
    palm_quat: np.ndarray
    hand_joints: np.ndarray
    axis_coord: float
    pregrasp_joints: np.ndarray | None = None
    quality: float = 0.0
    source: str = "analytic"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.palm_pos = np.asarray(self.palm_pos, dtype=np.float64).reshape(3)
        self.palm_quat = np.asarray(self.palm_quat, dtype=np.float64).reshape(4)
        self.hand_joints = np.asarray(self.hand_joints, dtype=np.float64).reshape(NUM_HAND_JOINTS)
        if self.pregrasp_joints is None:
            # No pre-shape supplied: fall back to the grasp pose itself. That disables
            # the approach->close motion rather than silently inventing one.
            self.pregrasp_joints = self.hand_joints.copy()
        else:
            self.pregrasp_joints = np.asarray(
                self.pregrasp_joints, dtype=np.float64
            ).reshape(NUM_HAND_JOINTS)

        norm = float(np.linalg.norm(self.palm_quat))
        if norm < 1e-9:
            raise ValueError("palm_quat has zero norm")
        self.palm_quat = self.palm_quat / norm

        if self.role not in ("giver", "receiver"):
            raise ValueError(f"role must be 'giver' or 'receiver', got {self.role!r}")

    def opposes(self, other: "GraspEntry") -> bool:
        """True if the two grasps sit on opposite ends of the object's long axis.

        The handover is only feasible when the receiver's end is clear of the giver's
        hand, so pairing must be checked, never assumed.
        """
        return float(self.axis_coord) * float(other.axis_coord) < 0.0

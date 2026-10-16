"""Object-frame grasp library for dexterous (16-DOF) manipulation demos.

Source-agnostic by design: `analytic` provides geometric wrap grasps that unblock the
pipeline today; a BODex-backed builder can populate the same `GraspEntry` records later
without touching the demo generator.
"""

from .analytic import WrapProfile, make_cylinder_library
from .api import GraspLibrary, quat_mul, quat_rotate
from .types import NUM_HAND_JOINTS, GraspEntry, GraspRole

__all__ = [
    "GraspEntry",
    "GraspLibrary",
    "GraspRole",
    "NUM_HAND_JOINTS",
    "WrapProfile",
    "make_cylinder_library",
    "quat_mul",
    "quat_rotate",
]

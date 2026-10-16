"""Dexterous (16-DOF) finger targets for demo generation.

Replaces the finger content of the demo action, NOT its timing.

`trl_utils/dualarm_ik_controller.py::GripperMapping` drives all 13 flexion joints to a
single shared angle (`torch.where(grip_scalar == 1.0, 0.0, 1.57)`) and pins the three
abduction joints to zero. That is a *binary, one-DOF* hand written across 16 numbers.
Demos collected through it would teach behaviour cloning that fingers move in lockstep,
which is exactly the "gripper wearing a hand" failure the dexterous work exists to fix.

What this module changes is only *what "closed" means*: instead of every joint slamming
to 1.57, the hand interpolates toward an object-conditioned grasp pose drawn from the
grasp library. The *when* still comes from the existing stage machine's grip scalar, so
the carefully-tuned handover timing is preserved untouched.

    finger_cmd = open_pose + closure * (grasp_pose - open_pose)
    closure    = 1 - grip_scalar        # grip_scalar 1.0 == open

Role convention: agent0 = RIGHT = giver (lifts the rod off the table), agent1 = LEFT =
receiver (takes the opposite end mid-air). This mirrors the hw obs14 contract and the
`_right_object_contact_local` / `_left_fixed_handover_pos` split already in
`generate_demo.py`. Flip with `right_is_giver=False` if a replay shows otherwise.
"""

from __future__ import annotations

import numpy as np
import torch

from .grasp_library import GraspLibrary, make_cylinder_library
from .grasp_library.types import NUM_HAND_JOINTS

# Allegro fully-open posture. Matches GripperMapping.open_angle (0.0) so the start of
# every demo is byte-identical to the legacy path; only the closed end differs.
OPEN_POSE = np.zeros(NUM_HAND_JOINTS, dtype=np.float64)

# The demo's grip channel is a joint angle spanning [0.0, 1.57] rad -- the same range
# `trl_utils/scaling.py::scale_hand_to_action` normalises against, and the same
# close_angle `GripperMapping` uses. Kept here so the closure mapping cannot silently
# drift away from the units the demo generator actually emits.
GRIP_CLOSE_ANGLE = 1.57


class DexterousGraspPlan:
    """Per-env grasp assignment for one episode.

    Sampled once at episode start (not per step) so the hand converges toward a single
    consistent target instead of chattering between library entries.
    """

    def __init__(
        self,
        num_envs: int,
        library: GraspLibrary,
        device,
        seed: int = 0,
        right_is_giver: bool = True,
    ):
        self.num_envs = int(num_envs)
        self.library = library
        self.device = device
        self.right_is_giver = bool(right_is_giver)

        rng = np.random.default_rng(seed)
        pairs = library.sample_pairs(self.num_envs, rng)  # verifies opposite ends

        giver_q = np.stack([g.hand_joints for g, _ in pairs])
        recv_q = np.stack([r.hand_joints for _, r in pairs])
        giver_pre = np.stack([g.pregrasp_joints for g, _ in pairs])
        recv_pre = np.stack([r.pregrasp_joints for _, r in pairs])
        giver_axis = np.array([g.axis_coord for g, _ in pairs])
        recv_axis = np.array([r.axis_coord for _, r in pairs])

        if right_is_giver:
            right_q, left_q = giver_q, recv_q
            right_pre, left_pre = giver_pre, recv_pre
            self.right_axis, self.left_axis = giver_axis, recv_axis
        else:
            right_q, left_q = recv_q, giver_q
            right_pre, left_pre = recv_pre, giver_pre
            self.right_axis, self.left_axis = recv_axis, giver_axis

        self.right_grasp = torch.tensor(right_q, dtype=torch.float32, device=device)
        self.left_grasp = torch.tensor(left_q, dtype=torch.float32, device=device)
        self.right_pregrasp = torch.tensor(right_pre, dtype=torch.float32, device=device)
        self.left_pregrasp = torch.tensor(left_pre, dtype=torch.float32, device=device)
        self.open_pose = torch.tensor(
            np.broadcast_to(OPEN_POSE, (self.num_envs, NUM_HAND_JOINTS)).copy(),
            dtype=torch.float32,
            device=device,
        )

        # Guard the invariant that makes the handover solvable at all.
        if not bool(np.all(self.right_axis * self.left_axis < 0.0)):
            raise RuntimeError(
                "sampled grasps put both hands on the same end of the rod; the handover "
                "would be geometrically unsolvable"
            )

    def finger_targets(self, grip_scalar: torch.Tensor, side: str) -> torch.Tensor:
        """Interpolate PREGRASP -> grasp for one hand.

        The hand holds the pre-shaped pose throughout the approach (thumb opposed,
        fingers spread but uncurled) and closes from there, so the fingers squeeze onto
        the rod. Interpolating from a flat open hand instead sweeps them through it --
        that produced 0/25 demos while the legacy scalar grip got 20/25.

        Args:
            grip_scalar: [N] or [N,1] from the demo stage machine. This is a JOINT ANGLE
                IN RADIANS, 0.0 = open .. GRIP_CLOSE_ANGLE = closed -- the same units
                `scale_hand_to_action(v, 0.0, 1.57)` consumes. It is NOT a 0/1 flag.
            side: "right" or "left".

        Returns:
            [N, 16] joint targets in radians.
        """
        if side not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', got {side!r}")
        if side == "right":
            grasp, pregrasp = self.right_grasp, self.right_pregrasp
        else:
            grasp, pregrasp = self.left_grasp, self.left_pregrasp

        g = grip_scalar.reshape(-1).to(self.open_pose.dtype)
        if int(g.shape[0]) != self.num_envs:
            raise ValueError(
                f"grip_scalar has {int(g.shape[0])} envs, plan built for {self.num_envs}"
            )
        # Closure rises WITH the commanded angle: 0 rad = open = pregrasp,
        # GRIP_CLOSE_ANGLE = closed = full grasp.
        #
        # Do not "simplify" this to (1 - g). That treats g as a 0/1 open flag and
        # inverts the whole hand: the giver opens while reaching and the receiver stays
        # clamped shut. It cost three collection runs (0/25 each) before the measured
        # flexion trace made it obvious.
        closure = (g / GRIP_CLOSE_ANGLE).clamp(0.0, 1.0).unsqueeze(-1)  # [N,1]
        return pregrasp + closure * (grasp - pregrasp)

    def describe(self) -> str:
        giver = "right" if self.right_is_giver else "left"
        return (
            f"DexterousGraspPlan(envs={self.num_envs}, giver={giver}, "
            f"library={len(self.library)} entries, "
            f"right_axis[0]={self.right_axis[0]:+.3f}, left_axis[0]={self.left_axis[0]:+.3f})"
        )


def synergy_finger_targets(
    plan: "DexterousGraspPlan",
    grip_action: torch.Tensor,
    side: str,
    slot_of_canon: torch.Tensor | None = None,
) -> torch.Tensor:
    """One normalised grip action per hand -> all 16 Allegro joint targets.

    This is the policy-facing counterpart of `DexterousGraspPlan.finger_targets`: it
    turns the teacher's 44-D action space (6 arm + 16 free finger joints per agent) back
    into a 14-D one (6 arm + 1 grip per agent) while still commanding every finger. The
    scalar travels the library's pregrasp->grasp line, i.e. a task-specific 1-D
    eigengrasp, so the 32 finger dimensions PPO used to explore -- and which no demo
    ever moves off that line -- disappear from the search.

    Args:
        plan: per-env grasp assignment. Build it with the SAME seed, library and
            `right_is_giver` the demos used, or the postures will not line up with the
            BC targets.
        grip_action: [N] or [N,1] in [-1, 1], the env's own convention (see
            `unscale_action_to_hand(a, 0.0, GRIP_CLOSE_ANGLE)`): **-1 = open/pregrasp,
            +1 = closed/full grasp**. The demo stage machine's internal grip scalar runs
            the OTHER way; use the value `scale_hand_to_action` recorded, not that one.
        side: "right" or "left".
        slot_of_canon: optional [16] index map from the library's canonical
            joint_<k>_0 order to Isaac's interleaved hand slots. When given, the result
            is returned in Isaac slot order, ready to write into `*_actions[:, 6:22]`.

    Returns:
        [N, 16] joint targets in radians.
    """
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")

    a = grip_action.reshape(-1)
    if int(a.shape[0]) != plan.num_envs:
        # Broadcasting here would quietly apply env 0's grasp to every env.
        raise ValueError(
            f"grip_action has {int(a.shape[0])} envs, plan built for {plan.num_envs}"
        )

    # Clamp rather than extrapolate: a stochastic policy routinely emits past +-1, and
    # letting that run past the library grasp drives the fingers into the rod.
    closure = ((a.to(plan.open_pose.dtype) + 1.0) * 0.5).clamp(0.0, 1.0).unsqueeze(-1)
    if side == "right":
        pregrasp, grasp = plan.right_pregrasp, plan.right_grasp
    else:
        pregrasp, grasp = plan.left_pregrasp, plan.left_grasp
    canon = pregrasp + closure * (grasp - pregrasp)

    if slot_of_canon is None:
        return canon
    out = torch.empty_like(canon)
    out[:, slot_of_canon] = canon
    return out


_DEFAULT_LIBRARY: GraspLibrary | None = None


def default_cylinder_library() -> GraspLibrary:
    """Analytic library for the handover rod (r=0.027, L=0.30), built once per process."""
    global _DEFAULT_LIBRARY
    if _DEFAULT_LIBRARY is None:
        _DEFAULT_LIBRARY = GraspLibrary(
            make_cylinder_library(radius=0.027, height=0.30, n_per_role=24),
            object_id="handover_cylinder",
        )
    return _DEFAULT_LIBRARY

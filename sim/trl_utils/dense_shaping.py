"""Potential-based reward shaping for staged (subgoal) tasks.

The handover reward is a cascade of binary stage bonuses with nothing in
between, which gives a policy no gradient until it happens to land inside a
threshold. This module adds

    F(s, s') = gamma * phi(s') - phi(s),      phi(s) = -coef * dist_to_stage_target(s)

which Ng, Harada & Russell (1999) prove leaves the optimal policy unchanged for
ANY phi that is a function of state alone -- so it can only affect how easily
gradient descent finds that policy, never what it converges to. `phi` here is a
function of the arm's latched stage index plus the current distances, both of
which are part of the environment state.

Kept free of Isaac imports so it stays unit-testable.
"""
from __future__ import annotations

from typing import Sequence

import torch


def select_stage_distance(stage: torch.Tensor, dists: torch.Tensor) -> torch.Tensor:
    """Distance to the target of each env's current stage.

    stage: [B] (float stage index, as the task stores it); dists: [B, S] with one
    column per stage. Stages at or past S are terminal for that track and
    contribute no distance, so the potential goes flat once an arm is finished.
    """
    n_stages = int(dists.shape[1])
    idx = stage.long()
    in_range = (idx >= 0) & (idx < n_stages)
    safe = idx.clamp(0, n_stages - 1).unsqueeze(1)
    gathered = dists.gather(1, safe).squeeze(1)
    return torch.where(in_range | (idx < 0), gathered, torch.zeros_like(gathered))


def stage_potential(stages: Sequence[torch.Tensor],
                    dist_tables: Sequence[torch.Tensor],
                    coef: float = 1.0) -> torch.Tensor:
    """phi = -coef * sum over tracks of the current stage's distance.

    One (stage, table) pair per independently-advancing track — the handover has
    two, the right arm and the left arm.
    """
    total = None
    for stage, table in zip(stages, dist_tables):
        d = select_stage_distance(stage, table)
        total = d if total is None else total + d
    if total is None:
        raise ValueError("stage_potential needs at least one track")
    return -float(coef) * total


def potential_shaping(phi_now: torch.Tensor, phi_prev: torch.Tensor,
                      valid: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """F = gamma * phi(s') - phi(s), zeroed where `valid` is False.

    `valid` must be False on the first step of an episode: phi_prev then belongs
    to the previous episode and would pay out a large spurious reward at reset.
    """
    f = float(gamma) * phi_now - phi_prev
    return torch.where(valid, f, torch.zeros_like(f))

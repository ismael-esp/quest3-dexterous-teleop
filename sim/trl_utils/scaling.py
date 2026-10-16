# utils/scaling.py
from __future__ import annotations
import torch
from typing import Tuple

_EPS = 1e-6

@torch.no_grad()
def scale_joint_to_action(
    joint_values: torch.Tensor,            # [B, J]
    lower_limits: torch.Tensor,            # [J]
    upper_limits: torch.Tensor,            # [J]
) -> torch.Tensor:
    """joint pos [rad] → [-1, 1] (고정 조인트는 0)"""
    denom = (upper_limits - lower_limits)
    mask = denom > _EPS  # [J]
    # [1, J]로 확장(브로드캐스팅용)
    lo = lower_limits.unsqueeze(0)
    hi = upper_limits.unsqueeze(0)
    d  = denom.unsqueeze(0)
    m  = mask.unsqueeze(0)
    out = torch.zeros_like(joint_values)
    out[:, m[0]] = 2.0 * (joint_values[:, m[0]] - lo[:, m[0]]) / d[:, m[0]] - 1.0
    out[:, ~m[0]] = 0.0
    return out

@torch.no_grad()
def unscale_action_to_joint(
    actions: torch.Tensor,                # [B, J]
    lower_limits: torch.Tensor,           # [J]
    upper_limits: torch.Tensor,           # [J]
) -> torch.Tensor:
    """[-1, 1] → joint pos [rad]; 고정 조인트는 lo로 고정"""
    denom = (upper_limits - lower_limits)
    mask = denom > _EPS
    lo = lower_limits.unsqueeze(0)
    d  = denom.unsqueeze(0)
    m  = mask.unsqueeze(0)
    out = torch.zeros_like(actions)
    out[:, m[0]] = lo[:, m[0]] + (actions[:, m[0]] + 1.0) * 0.5 * d[:, m[0]]
    out[:, ~m[0]] = lo[:, ~m[0]]
    return out

@torch.no_grad()
def scale_hand_to_action(
    joint_values: torch.Tensor,           # [B] or [B,1] or [B,K]
    min_val: float = 0.0,
    max_val: float = 1.57,
) -> torch.Tensor:
    """(min,max) → [-1,1]"""
    return 2.0 * (joint_values - min_val) / (max_val - min_val) - 1.0

@torch.no_grad()
def unscale_action_to_hand(
    actions: torch.Tensor,                # [B] or [B,1] or [B,K]
    min_val: float = 0.0,
    max_val: float = 1.57,
) -> torch.Tensor:
    """[-1,1] → (min,max)"""
    return min_val + (actions + 1.0) * 0.5 * (max_val - min_val)

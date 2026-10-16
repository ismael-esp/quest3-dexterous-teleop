"""Generic per-task IO/path helpers.

Functions in this module are task-agnostic counterparts to the colorcube-only
helpers in `utils/colorcube_io.py`. Each entry point takes an explicit
`task_id` so new tasks can opt in without adding new modules.

`utils/colorcube_io.py` keeps its dedicated functions intact; this file is a
parallel surface for tasks that don't need colorcube-specific fields.
"""

from __future__ import annotations

import glob
import os
import re


def sanitize_tag(tag: str) -> str:
    tag = str(tag or "").strip()
    if not tag:
        return ""
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
    return tag.strip("._-")


def _compact_number_token(value: float) -> str:
    try:
        value_f = float(value)
    except Exception:
        return sanitize_tag(str(value))
    if abs(value_f - round(value_f)) < 1e-9:
        return str(int(round(value_f)))
    return sanitize_tag(f"{value_f:.4f}".rstrip("0").rstrip(".").replace(".", "p"))


def build_demo_tag(
    task_id: str,
    obs_dim: int,
    share_obs_dim: int,
    *,
    use_vision: bool = False,
    vision_use_depth: bool = True,
    vision_front_only: bool = True,
    vision_use_front_camera: bool = True,
    extra_tokens: tuple[str, ...] = (),
) -> str:
    parts: list[str] = []
    if use_vision:
        parts.append("vision")
        parts.append("rgbd" if vision_use_depth else "rgb")
        if vision_front_only:
            parts.append("frontonly")
        elif vision_use_front_camera:
            parts.append("multiview")
        else:
            parts.append("toponly")
    parts.append(f"obs{int(obs_dim)}share{int(share_obs_dim)}")
    for token in extra_tokens:
        token_s = sanitize_tag(str(token))
        if token_s:
            parts.append(token_s)
    _ = task_id  # caller carries the task name via filename, not in the tag
    return sanitize_tag("_".join(parts))


def build_bc_cache_tag(
    task_id: str,
    policy_obs_dim: int,
    policy_share_obs_dim: int,
    *,
    bc_mse_stage_ratio: float,
    bc_noise_std: float,
    bc_patience: int,
    state_dependent_std: bool,
) -> str:
    noise_token = int(round(max(float(bc_noise_std), 0.0) * 100.0))
    parts = [
        f"obs{int(policy_obs_dim)}share{int(policy_share_obs_dim)}",
        f"mse{_compact_number_token(bc_mse_stage_ratio)}",
        f"noise{noise_token:03d}",
        f"es{int(bc_patience)}",
        "sds" if bool(state_dependent_std) else "nosds",
    ]
    _ = task_id
    return sanitize_tag("_".join(parts))


def build_default_run_stem(
    task_id: str,
    *,
    num_envs: int,
    num_mini_batch: int,
    clip_param_min_scale: float,
    clip_param_max_scale: float,
    bc_cache_tag: str,
) -> str:
    clip_fixed = abs(float(clip_param_min_scale) - float(clip_param_max_scale)) < 1e-9
    clip_tag = "clipfixed" if clip_fixed else "clipctrl"
    base = sanitize_tag(task_id) or "task"
    return sanitize_tag(
        f"ppo_{base}_{bc_cache_tag}_{int(num_envs)}env_mb{int(num_mini_batch)}_{clip_tag}_v1"
    )


def build_default_distill_run_stem(
    task_id: str,
    *,
    num_envs: int,
    distill_iters: int,
    distill_amp: bool,
    distill_beta_decay_steps: int,
    distill_small_obs: bool = False,
) -> str:
    iters_k = max(1, int(round(float(distill_iters) / 1000.0)))
    beta_k = max(0, int(round(float(distill_beta_decay_steps) / 1000.0)))
    base = sanitize_tag(task_id) or "task"
    parts = [
        f"{base}_faithful_stereo_distill",
        f"{int(num_envs)}env",
        f"{iters_k}k",
    ]
    if bool(distill_amp):
        parts.append("amp")
    parts.append(f"beta{beta_k}k")
    if bool(distill_small_obs):
        parts.append("smobs")
    parts.append("v1")
    return sanitize_tag("_".join(parts))


def get_default_bc_anchor_path(task_id: str) -> str:
    base = sanitize_tag(task_id) or "task"
    return os.path.join(
        "demos",
        "bc_cache",
        f"bc_{base}_state_bc_v1.pt",
    )


def resolve_demo_candidates(
    demo_dir: str,
    task_name: str,
    *,
    preferred_tag: str = "",
    use_vision: bool = False,
    vision_use_depth: bool = True,
    vision_front_only: bool = True,
    vision_use_front_camera: bool = True,
    extra_prefix_tokens: tuple[str, ...] = (),
) -> list[str]:
    name_prefix = f"Demo_success_{task_name}"
    for token in extra_prefix_tokens:
        token_s = sanitize_tag(str(token))
        if token_s:
            name_prefix = f"{name_prefix}_{token_s}"

    names: list[str] = []
    if use_vision:
        mode = "rgbd" if vision_use_depth else "rgb"
        if vision_front_only:
            names.append(f"{name_prefix}_vision_{mode}_frontonly.pickle")
        elif vision_use_front_camera:
            names.append(f"{name_prefix}_vision_{mode}_multiview.pickle")
        else:
            names.append(f"{name_prefix}_vision_{mode}_toponly.pickle")
        if vision_use_depth:
            names.extend([f"{name_prefix}_vision_rgbd.pickle", f"{name_prefix}_vision.pickle"])
        else:
            names.extend([f"{name_prefix}_vision_rgb.pickle", f"{name_prefix}_vision.pickle"])
    else:
        if preferred_tag:
            names.append(f"{name_prefix}_{preferred_tag}.pickle")
        names.append(f"{name_prefix}.pickle")

    candidates: list[str] = []
    seen: set[str] = set()
    for name in names:
        path = os.path.join(demo_dir, name)
        if path not in seen:
            candidates.append(path)
            seen.add(path)

    pattern = os.path.join(demo_dir, f"{name_prefix}*.pickle")
    globbed = sorted(
        glob.glob(pattern),
        key=lambda p: os.path.getmtime(p) if os.path.isfile(p) else -1,
        reverse=True,
    )
    for path in globbed:
        if path not in seen:
            candidates.append(path)
            seen.add(path)
    return candidates

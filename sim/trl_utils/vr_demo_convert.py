"""Variable-length VR episodes -> the fixed-length `Demo_success_<task>_<tag>.pickle` layout.

Rules (spec 2026-09-09 §2.4): accept by the operator's label; CUT at the first step the env's
success latched + `tail`; T = ceil(P95 of cut lengths / 10) * 10 (or an explicit T); PAD by holding
the last kept row; drop (and list) episodes longer than T. Never resample time.
Output layout == collect_demo_success.py: step-major lists of (N, 2, D) arrays.
"""
from __future__ import annotations

import math

import numpy as np

STEP_KEYS = ["observations", "share_observations", "actions", "raw_actions", "joint_positions", "joint_velocities"]


def episode_cut_index(ep: dict, tail: int = 30) -> int:
    T = int(ep["T"])
    se = ep.get("success_env_step")
    if isinstance(se, np.ndarray) and se.ndim == 1 and se.size == T and bool(se.any()):
        return min(T, int(np.argmax(se.astype(bool))) + 1 + int(tail))
    return T


def choose_T(lengths, pct: float = 95.0, round_to: int = 10) -> int:
    p = float(np.percentile(np.asarray(lengths, dtype=np.float64), pct))
    return int(math.ceil(p / round_to) * round_to)


def cut_and_pad(arr: np.ndarray, cut: int, T: int) -> np.ndarray:
    a = np.asarray(arr)[: int(cut)]
    if a.shape[0] >= T:
        return a[:T]
    pad = np.repeat(a[-1:], T - a.shape[0], axis=0)
    return np.concatenate([a, pad], axis=0)


def accept(ep: dict, include_manual_fail: bool = False) -> bool:
    m = ep.get("success_manual", None)
    if m is None:                                  # v1 pickle or env-ended episode: env flag decides
        m = bool(ep.get("success_env", ep.get("success", False)))
    return bool(m) or bool(include_manual_fail)


def build_payloads(episodes: list[dict], T: int, task: str, tag: str, tail: int = 30,
                   source: str = "vr", extra_meta: dict | None = None):
    kept, dropped, cuts = [], [], []
    for ep in episodes:
        cut = episode_cut_index(ep, tail=tail)
        if cut > T:
            dropped.append((ep.get("metadata", {}).get("object_id", "?"), int(ep["T"]), cut))
            continue
        kept.append(ep); cuts.append(cut)
    if not kept:
        raise ValueError(f"no episode fits T={T} (dropped {len(dropped)}); raise --T or --pct")
    stacked = {k: np.stack([cut_and_pad(ep[k], c, T) for ep, c in zip(kept, cuts)], axis=1) for k in STEP_KEYS}
    N = len(kept)
    obs_dim = int(stacked["observations"].shape[-1]); share_dim = int(stacked["share_observations"].shape[-1])
    object_ids = [str(ep.get("metadata", {}).get("object_id", "rod")) for ep in kept]
    env_success = [bool(ep.get("success_env", ep.get("success", False))) for ep in kept]
    meta_common = {
        "task": str(task), "seed": 0, "collect_target": N, "keep_target": N, "num_envs": N,
        "episode_length": int(T), "output_tag": str(tag), "source": source, "object_ids": object_ids,
        "cut_tail": int(tail), "cut_lengths": [int(c) for c in cuts], "env_success_rate": float(np.mean(env_success)),
        "n_dropped_long": len(dropped), "dropped": dropped,
    }
    meta_common.update(extra_meta or {})
    bc = {"observations": [], "share_observations": [], "actions": [],
          "metadata": dict(meta_common, obs_dim=obs_dim, share_obs_dim=share_dim, use_vision=False)}
    traj = {"raw_actions": [], "joint_positions": [], "joint_velocities": [],
            "init_cube_pos": np.stack([np.asarray(ep["object_pos"][0], dtype=np.float32) for ep in kept], axis=0),
            "metadata": dict(meta_common)}
    for t in range(T):
        bc["observations"].append(stacked["observations"][t])
        bc["share_observations"].append(stacked["share_observations"][t])
        bc["actions"].append(stacked["actions"][t])
        traj["raw_actions"].append(stacked["raw_actions"][t])
        traj["joint_positions"].append(stacked["joint_positions"][t])
        traj["joint_velocities"].append(stacked["joint_velocities"][t])
    report = {"n_in": len(episodes), "n_kept": N, "n_dropped_long": len(dropped), "dropped": dropped, "T": int(T),
              "cut_lengths": [int(c) for c in cuts], "object_ids": object_ids,
              "env_success_rate": meta_common["env_success_rate"]}
    return bc, traj, report

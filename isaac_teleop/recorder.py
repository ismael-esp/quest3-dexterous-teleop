"""Per-episode demonstration recorder (variable length). One pickle per episode + session.json.

Schema 2 (2026-09-09): optional per-step keys (STEP_ARRAY_KEYS_V2) are stacked when present, and an
episode carries TWO labels -- `success_env` (the env's strict latch: object at goal, then both hands
home and open) and `success_manual` (the operator's verdict; None when not given). `success` keeps
the v1 meaning (== success_env) so old readers still work. `stage_peak` (max FSM stage per arm) and
`t_success_env` (first step the env latched, -1 if never) are derived at write time.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCHEMA_VERSION = 2
STEP_ARRAY_KEYS = [
    "observations", "share_observations", "actions", "raw_actions",
    "joint_positions", "joint_velocities", "object_pos", "object_quat", "ee_pos", "ee_quat",
]
# Present only in schema-2 snapshots (isaac_teleop/snapshot.py); stacked when the first step has them.
STEP_ARRAY_KEYS_V2 = [
    "palm_pos", "palm_quat", "target_wrist_pos", "target_wrist_quat",
    "subgoal", "fingertip_pos", "success_env_step",
]
VR_KEYS = {  # step key -> name inside ep["vr"]
    "vr_head": "head", "vr_left": "left", "vr_right": "right",
    "vr_left_state": "left_state", "vr_right_state": "right_state", "engaged": "engaged",
}
AGENT_ORDER = "agent0=RIGHT, agent1=LEFT"


class EpisodeRecorder:
    def __init__(self, out_root, task: str, session_tag: str, session_meta: dict,
                 outcome_in_name: bool = True):
        self.dir = Path(out_root) / task / session_tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.outcome_in_name = bool(outcome_in_name)   # False: plain ep_XXX.pickle (labels stay inside)
        self._steps: List[dict] = []
        self._ep_meta: dict = {}
        self.active = False
        existing = sorted(self.dir.glob("ep_*.pickle"))
        self._next_idx = (int(existing[-1].name.split("_")[1]) + 1) if existing else 0
        meta = dict(session_meta)
        meta.update({"task": task, "session_tag": session_tag, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "agent_order": AGENT_ORDER, "schema_version": SCHEMA_VERSION})
        (self.dir / "session.json").write_text(json.dumps(meta, indent=2, default=str))

    @property
    def length(self) -> int:
        return len(self._steps)

    def set_meta(self, extra: dict) -> None:
        """Merge keys into the current episode's metadata (e.g. an updated T_world_vr after re-centering)."""
        self._ep_meta.update(extra)

    def start(self, episode_meta: dict) -> None:
        self._steps = []
        self._ep_meta = dict(episode_meta)
        self._ep_meta["t_start_wall"] = time.time()
        self.active = True

    def append(self, step: Dict[str, np.ndarray]) -> None:
        if not self.active:
            raise RuntimeError("append() before start()")
        self._steps.append(step)

    def discard(self) -> None:
        self._steps = []
        self.active = False

    def finish(self, success: Optional[bool] = None, *, success_env: Optional[bool] = None,
               success_manual: Optional[bool] = None) -> Optional[Path]:
        """v1 callers: finish(success=<env flag>). v2 callers: finish(success_env=..., success_manual=...).

        The file suffix (when `outcome_in_name`) follows the MANUAL label when one is given, else the
        env flag: `_ok` / `_fail`."""
        if success_env is None:
            success_env = bool(success) if success is not None else False
        label = success_manual if success_manual is not None else bool(success_env)
        return self._write("ok" if label else "fail", success_env=bool(success_env), success_manual=success_manual)

    def abort(self) -> Optional[Path]:
        return self._write("aborted", success_env=False, success_manual=False)

    def _write(self, suffix: str, success_env: bool, success_manual: Optional[bool]) -> Optional[Path]:
        self.active = False
        if not self._steps:
            self._steps = []
            return None
        ep: dict = {}
        for k in STEP_ARRAY_KEYS:
            ep[k] = np.stack([np.asarray(s[k]) for s in self._steps], axis=0)
        for k in STEP_ARRAY_KEYS_V2:
            if k in self._steps[0]:
                ep[k] = np.stack([np.asarray(s[k]) for s in self._steps], axis=0)
        ep["vr"] = {name: np.stack([np.asarray(s[k]) for s in self._steps], axis=0) for k, name in VR_KEYS.items()}
        ep["t_wall"] = np.asarray([float(s["t_wall"]) for s in self._steps], dtype=np.float64)
        ep["T"] = len(self._steps)
        ep["success"] = bool(success_env)          # v1 name, kept for old readers
        ep["success_env"] = bool(success_env)
        ep["success_manual"] = success_manual
        meta = dict(self._ep_meta)
        meta.update({"task": self.task, "agent_order": AGENT_ORDER, "t_end_wall": time.time(),
                     "schema_version": SCHEMA_VERSION,
                     "obs_dim": int(ep["observations"].shape[-1]),
                     "share_obs_dim": int(ep["share_observations"].shape[-1]),
                     "act_dim": int(ep["actions"].shape[-1])})
        if "subgoal" in ep:
            meta["stage_peak"] = [float(v) for v in ep["subgoal"].max(axis=0)]
        if "success_env_step" in ep:
            se = ep["success_env_step"].astype(bool)
            ep["success_env_step"] = se
            meta["t_success_env"] = int(np.argmax(se)) if se.any() else -1
        ep["metadata"] = meta
        tag = f"_{suffix}" if (self.outcome_in_name or suffix == "aborted") else ""
        path = self.dir / f"ep_{self._next_idx:03d}{tag}.pickle"
        with open(path, "wb") as f:
            pickle.dump(ep, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._next_idx += 1
        self._steps = []
        return path

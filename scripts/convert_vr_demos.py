"""Convert recorded VR teleop sessions (ep_*.pickle) into fixed-length Demo_success_<task>_<tag>.pickle.

    python scripts/convert_vr_demos.py --tag vrA --sessions demos/vr_cloudxr/handover_tshape/2026*/
Cuts every accepted episode at the env success latch + --tail, picks T = P95 of the cut lengths,
hold-pads the shorter ones. Prints the T it chose (the episode length a BC policy is trained with).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from isaac_teleop.paths import sim_root  # noqa: E402

if sim_root() not in sys.path:
    sys.path.insert(0, sim_root())

from trl_utils.task_io import sanitize_tag  # noqa: E402
from trl_utils.vr_demo_convert import accept, build_payloads, choose_T, episode_cut_index  # noqa: E402


def _load_sessions(patterns):
    files = []
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            files += sorted(glob.glob(os.path.join(d, "ep_*.pickle")))
    files = [f for f in files if "_aborted" not in os.path.basename(f)]
    eps = []
    for f in files:
        with open(f, "rb") as fh:
            ep = pickle.load(fh)
        ep.setdefault("metadata", {})["source_file"] = f
        # v1 pickles: derive object_id from session.json when present
        if "object_id" not in ep["metadata"]:
            sess = Path(f).parent / "session.json"
            ep["metadata"]["object_id"] = "rod"
            if sess.exists():
                ep["metadata"]["object_id"] = json.loads(sess.read_text()).get("object_id", "rod")
        eps.append(ep)
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True, help="session dirs or globs holding ep_*.pickle")
    ap.add_argument("--task", default="handover_tshape")
    ap.add_argument("--tag", required=True, help="-> demos/Demo_success_<task>_<tag>.pickle")
    ap.add_argument("--tail", type=int, default=30, help="steps kept after the env success latch")
    ap.add_argument("--pct", type=float, default=95.0, help="percentile of cut lengths that sets T")
    ap.add_argument("--T", type=int, default=0, help="explicit T (overrides --pct)")
    ap.add_argument("--include_manual_fail", action="store_true")
    ap.add_argument("--out_dir", default=str(_REPO_ROOT / "demos" / "converted"))
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    eps = _load_sessions(a.sessions)
    if not eps:
        sys.exit("no ep_*.pickle found")
    acc = [ep for ep in eps if accept(ep, include_manual_fail=a.include_manual_fail)]
    if not acc:
        sys.exit(f"{len(eps)} episodes loaded but none accepted (operator label False / env never latched); "
                 "use --include_manual_fail to keep them anyway")
    cuts = [episode_cut_index(ep, tail=a.tail) for ep in acc]
    T = int(a.T) if a.T > 0 else choose_T(cuts, pct=a.pct)
    print(f"[convert] episodes={len(eps)} accepted={len(acc)} cut lengths min/med/max="
          f"{min(cuts)}/{int(np.median(cuts))}/{max(cuts)} -> T={T}")
    print(f"[convert] env-success rate among accepted: "
          f"{np.mean([bool(ep.get('success_env', ep.get('success', False))) for ep in acc]):.2f} "
          "(low => operators skipped the home-return tail)")
    bc, traj, rep = build_payloads(acc, T=T, task=a.task, tag=a.tag, tail=a.tail,
                                   extra_meta={"sessions": list(a.sessions)})
    for oid in sorted(set(rep["object_ids"])):
        print(f"[convert]   {oid}: {rep['object_ids'].count(oid)} demos")
    for d in rep["dropped"]:
        print(f"[convert]   DROPPED (longer than T): object={d[0]} T={d[1]} cut={d[2]}")
    if a.dry_run:
        return
    os.makedirs(a.out_dir, exist_ok=True)
    tag = sanitize_tag(a.tag)
    bc_path = os.path.join(a.out_dir, f"Demo_success_{a.task}_{tag}.pickle")
    traj_path = os.path.join(a.out_dir, f"Demo_Traj_{a.task}_{tag}.pickle")
    with open(bc_path, "wb") as f:
        pickle.dump([bc], f)
    with open(traj_path, "wb") as f:
        pickle.dump([traj], f)
    print(f"[convert] wrote {bc_path}\n[convert] wrote {traj_path}")
    print(f"[convert] fixed episode length T={T} (train a policy on these demos with that horizon)")


if __name__ == "__main__":
    main()

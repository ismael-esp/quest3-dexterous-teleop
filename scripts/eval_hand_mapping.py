#!/usr/bin/env python3
"""Per-finger flexion agreement between the operator's tracked hand and the retargeted Allegro hand.

Replays the Quest 3 hand-tracking stream stored in recorded episodes (ep["vr"]["left"/"right"], the exact
input the teleop fed to the retargeter) through isaac_teleop.xr_hand.AllegroDexPilot, for both profiles:
  chain    (default)  wrist->PIP/DIP/tip vectors per finger + angle-mapped thumb + pinch snap
  dexpilot            fingertip-only DexPilot objective
Human flexion = sum of the two bend angles along a finger's keypoint chain; robot flexion = sum of that
finger's three flexion joints. Reports Pearson r and the robot flexion range per finger; writes a scatter
plot + CSV. Needs dex_retargeting (the Isaac Lab env), no simulator.

    python scripts/eval_hand_mapping.py            # after scripts/download_assets.sh demos
"""
import argparse
import glob
import os
import pickle
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

QWS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, QWS)
from isaac_teleop.xr_hand import AllegroDexPilot, mp21_to_operator_wrist, xr25_to_mp21  # noqa: E402

HUMAN = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12], "ring": [13, 14, 15, 16]}
ROBOT = {"thumb": [13, 14, 15], "index": [1, 2, 3], "middle": [5, 6, 7], "ring": [9, 10, 11]}   # canonical joints
COLOR = {"thumb": "#e15759", "index": "#4e79a7", "middle": "#59a14f", "ring": "#f28e2b"}


def flexion(k, chain):
    v = [k[chain[i + 1]] - k[chain[i]] for i in range(3)]
    return sum(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9), -1, 1))
               for a, b in zip(v[:-1], v[1:]))


def replay(files, side, profile):
    """-> human flexion (N,4), robot flexion (N,4) over every tracked frame of every episode."""
    h, r = [], []
    for f in files:
        ep = pickle.load(open(f, "rb"))
        joints, t = ep["vr"][side], np.asarray(ep["t_wall"], dtype=np.float64)
        rt = AllegroDexPilot(side, profile=profile)        # fresh filter state per episode, like a new session
        for k in range(len(joints)):
            if not np.isfinite(joints[k]).all():
                continue
            mp = mp21_to_operator_wrist(xr25_to_mp21(joints[k]), side)
            q = rt.retarget(mp, t=t[k])
            h.append([flexion(mp, HUMAN[n]) for n in HUMAN])
            r.append([q[ROBOT[n]].sum() for n in HUMAN])
    return np.asarray(h), np.asarray(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", nargs="+", default=[os.path.join(QWS, "demos/vr_cloudxr/handover_tshape/20260821_185201")])
    p.add_argument("--profiles", nargs="+", default=["chain", "dexpilot"])
    p.add_argument("--out", default=os.path.join(QWS, "docs/results"))
    a = p.parse_args()
    files = sorted(f for s in a.sessions for f in glob.glob(os.path.join(s, "ep_*.pickle")))
    if not files:
        sys.exit("no ep_*.pickle found (scripts/download_assets.sh demos)")
    os.makedirs(a.out, exist_ok=True)

    rows, data = [], {}
    for profile in a.profiles:
        for side in ("right", "left"):
            h, r = replay(files, side, profile)
            data[(profile, side)] = (h, r)
            for i, n in enumerate(HUMAN):
                cc = float(np.corrcoef(h[:, i], r[:, i])[0, 1])
                lo, hi = np.percentile(r[:, i], [2, 98])
                rows.append((profile, side, n, len(h), cc, lo, hi))
                print(f"{profile:8s} {side:5s} {n:6s} r = {cc:.3f}   robot flexion P2..P98 = {lo:.2f}..{hi:.2f} rad"
                      f"   ({len(h)} frames)")

    fig, axs = plt.subplots(2, 4, figsize=(15, 6.4), dpi=130)
    for row, side in enumerate(("right", "left")):
        for c, n in enumerate(HUMAN):
            ax = axs[row, c]
            title = []
            for profile, mk, alpha in zip(a.profiles, ("o", "x"), (0.25, 0.12)):
                h, r = data[(profile, side)]
                cc = np.corrcoef(h[:, c], r[:, c])[0, 1]
                ax.scatter(h[::3, c], r[::3, c], s=4 if mk == "o" else 6, marker=mk, alpha=alpha,
                           color=COLOR[n] if profile == a.profiles[0] else "0.45", label=f"{profile}")
                title.append(f"{profile} {cc:.2f}")
            ax.set_title(f"{side} {n}   r: " + " / ".join(title), fontsize=10)
            ax.set_xlabel("human finger flexion [rad]")
            if c == 0:
                ax.set_ylabel("Allegro finger flexion [rad]")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    axs[0, 0].legend(loc="upper left", fontsize=8, markerscale=3, frameon=False)
    fig.suptitle("Quest 3 hand tracking → Allegro: per-finger flexion over the recorded teleop episodes", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "hand_mapping_flexion.png"))
    with open(os.path.join(a.out, "hand_mapping_flexion.csv"), "w") as f:
        f.write("profile,hand,finger,frames,pearson_r,robot_flex_p2,robot_flex_p98\n")
        for row in rows:
            f.write("{},{},{},{},{:.3f},{:.3f},{:.3f}\n".format(*row))
    print("wrote", os.path.join(a.out, "hand_mapping_flexion.png"))


if __name__ == "__main__":
    main()

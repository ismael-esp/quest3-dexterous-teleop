"""Step A (Isaac Lab env, dex_retargeting): recorded Quest hand frames -> Allegro joints, saved for rendering.

    python scripts/media/retarget_frames.py --episode demos/.../ep_002.pickle --side right --start 0 --stop 684 --every 2 --out /tmp/r.npz
"""
import argparse
import os
import pickle
import sys

import numpy as np

QWS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, QWS)
from isaac_teleop.xr_hand import CANON_JOINT_NAMES, AllegroDexPilot, mp21_to_operator_wrist, xr25_to_mp21  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--episode", required=True)
p.add_argument("--side", default="right", choices=["right", "left"])
p.add_argument("--profile", default="chain", choices=["chain", "dexpilot"])
p.add_argument("--start", type=int, default=0)
p.add_argument("--stop", type=int, default=-1)
p.add_argument("--every", type=int, default=2)
p.add_argument("--out", required=True)
a = p.parse_args()
ep = pickle.load(open(a.episode, "rb"))
joints, t = ep["vr"][a.side], np.asarray(ep["t_wall"], dtype=np.float64)
rt = AllegroDexPilot(a.side, profile=a.profile)
kp, q = [], []
stop = len(joints) if a.stop < 0 else a.stop
for k in range(stop):                          # run the filter over every frame, keep every N-th in the window
    if not np.isfinite(joints[k]).all():
        continue
    mp = mp21_to_operator_wrist(xr25_to_mp21(joints[k]), a.side)
    qk = rt.retarget(mp, t=t[k])
    if k >= a.start and (k - a.start) % a.every == 0:
        kp.append(mp); q.append(qk)
np.savez(a.out, kp=np.asarray(kp), q_sim=np.asarray(q), names=np.asarray(CANON_JOINT_NAMES), urdf=rt.urdf_path)
print("wrote", a.out, len(kp), "frames")

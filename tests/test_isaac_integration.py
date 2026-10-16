"""Opt-in end-to-end smoke: runs the teleop script headless with the fake VR source.
Run:  GPU=1 python -m pytest -m isaac -o addopts="" tests/test_isaac_integration.py -v -s
(GPU=<n> pins physics + renderer to that physical GPU; see scripts/run_teleop.sh.)
"""
import glob
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

QWS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.isaac
def test_fake_vr_episode_end_to_end(tmp_path):
    out = tmp_path / "demos"
    cmd = [os.path.join(QWS, "scripts", "run_teleop.sh"), "--fake_vr", "--auto_engage", "--object", "rod",
           "--steps", "240", "--out_dir", str(out), "--session_tag", "pytest", "--eye_res", "320x240"]
    r = subprocess.run(cmd, cwd=QWS, capture_output=True, text=True, timeout=1500)
    sys.stdout.write(r.stdout[-4000:]); sys.stderr.write(r.stderr[-4000:])
    assert r.returncode == 0, "teleop script failed"
    assert "[teleop] cameras ok" in r.stdout
    files = glob.glob(str(out / "handover_tshape" / "pytest" / "ep_*.pickle"))
    assert len(files) == 1, files
    ep = pickle.load(open(files[0], "rb"))
    assert ep["T"] == 240
    assert ep["observations"].shape == (240, 2, 74)
    assert ep["actions"].shape == (240, 2, 22)
    assert np.isfinite(ep["actions"]).all() and np.abs(ep["actions"]).max() <= 1.0
    assert ep["vr"]["head"].shape == (240, 4, 4)
    assert ep["vr"]["engaged"][-1].tolist() == [1, 1]
    # schema 2 (spec 2026-09-09 §2.2)
    assert ep["metadata"]["schema_version"] == 2
    assert ep["palm_pos"].shape == (240, 2, 3) and ep["palm_quat"].shape == (240, 2, 4)
    assert ep["target_wrist_pos"].shape == (240, 2, 3) and ep["target_wrist_quat"].shape == (240, 2, 4)
    assert ep["fingertip_pos"].shape == (240, 2, 4, 3) and ep["subgoal"].shape == (240, 2)
    assert ep["success_env_step"].shape == (240,) and ep["success_env"] is False and ep["success_manual"] is None
    assert ep["metadata"]["stage_peak"] == [0.0, 0.0] and ep["metadata"]["t_success_env"] == -1
    # palm is a rigid 95.5 mm offset from the wrist_3 body at every step, both arms
    d = np.linalg.norm(ep["palm_pos"] - ep["ee_pos"], axis=-1)
    assert np.all(np.abs(d - 0.0955) < 0.002), (d.min(), d.max())
    # fingertips are a plausible distance from the palm origin and finite
    ft = np.linalg.norm(ep["fingertip_pos"] - ep["palm_pos"][:, :, None, :], axis=-1)
    assert np.isfinite(ft).all() and 0.04 < ft.mean() < 0.20, ft.mean()
    # the commanded target moves with the fake wrists
    assert np.abs(ep["target_wrist_pos"][-1] - ep["target_wrist_pos"][0]).max() > 0.01
    assert "[teleop] palm check right" in r.stdout
    # the arms actually moved under the fake wrist motion
    q = ep["raw_actions"][:, :, :6]
    assert np.abs(q[-1] - q[0]).max() > 0.02
    # hands got non-trivial finger targets from DexPilot
    assert ep["raw_actions"][:, :, 6:].max() > 0.2
    sess = json.load(open(out / "handover_tshape" / "pytest" / "session.json"))
    assert sess["object_id"] == "rod" and sess["schema_version"] == 2 and "T_wrist3_palm" in sess

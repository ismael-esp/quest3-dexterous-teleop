import json
import pickle

import numpy as np

from isaac_teleop.recorder import EpisodeRecorder


def _step(t, v2=False, success_env=False, stage=(0.0, 0.0)):
    d = {
        "observations": np.full((2, 74), t, np.float32),
        "share_observations": np.full((2, 92), t, np.float32),
        "actions": np.full((2, 22), 0.1 * t, np.float32),
        "raw_actions": np.full((2, 22), t, np.float32),
        "joint_positions": np.zeros((2, 22), np.float32),
        "joint_velocities": np.zeros((2, 22), np.float32),
        "object_pos": np.array([0, 0.4, 1.0], np.float32),
        "object_quat": np.array([1, 0, 0, 0], np.float32),
        "ee_pos": np.zeros((2, 3), np.float32),
        "ee_quat": np.tile([1, 0, 0, 0], (2, 1)).astype(np.float32),
        "vr_head": np.eye(4, dtype=np.float32),
        "vr_left": np.full((25, 4, 4), np.nan, np.float32),
        "vr_right": np.zeros((25, 4, 4), np.float32),
        "vr_left_state": np.zeros(4, np.float32),
        "vr_right_state": np.zeros(4, np.float32),
        "engaged": np.array([1, 0], np.int8),
        "t_wall": 100.0 + t,
    }
    if v2:
        d.update({
            "palm_pos": np.full((2, 3), 0.5, np.float32), "palm_quat": np.tile([1, 0, 0, 0], (2, 1)).astype(np.float32),
            "target_wrist_pos": np.zeros((2, 3), np.float32), "target_wrist_quat": np.tile([1, 0, 0, 0], (2, 1)).astype(np.float32),
            "subgoal": np.array(stage, np.float32), "fingertip_pos": np.zeros((2, 4, 3), np.float32),
            "success_env_step": bool(success_env),
        })
    return d


def test_recorder_writes_session_json_and_episode(tmp_path):
    rec = EpisodeRecorder(tmp_path, task="handover_tshape", session_tag="s1",
                          session_meta={"cli": "--foo", "obs_dim": 74})
    assert (tmp_path / "handover_tshape" / "s1" / "session.json").exists()
    meta = json.loads((tmp_path / "handover_tshape" / "s1" / "session.json").read_text())
    assert meta["obs_dim"] == 74 and meta["task"] == "handover_tshape"

    assert not rec.active
    rec.start({"episode_meta": 1})
    assert rec.active and rec.length == 0
    for t in range(3):
        rec.append(_step(t))
    assert rec.length == 3
    path = rec.finish(success=True)                      # v1 call style still works
    assert path.name == "ep_000_ok.pickle"
    assert not rec.active
    ep = pickle.load(open(path, "rb"))
    assert ep["T"] == 3 and ep["success"] is True and ep["success_env"] is True and ep["success_manual"] is None
    assert ep["observations"].shape == (3, 2, 74)
    assert ep["actions"].shape == (3, 2, 22)
    assert ep["vr"]["left"].shape == (3, 25, 4, 4) and np.isnan(ep["vr"]["left"]).all()
    assert ep["vr"]["engaged"].shape == (3, 2)
    assert ep["t_wall"].shape == (3,)
    assert ep["metadata"]["episode_meta"] == 1 and ep["metadata"]["agent_order"] == "agent0=RIGHT, agent1=LEFT"
    assert ep["metadata"]["schema_version"] == 2 and "palm_pos" not in ep
    np.testing.assert_allclose(ep["actions"][2, 0, 0], 0.2, atol=1e-6)


def test_recorder_fail_abort_and_numbering(tmp_path):
    rec = EpisodeRecorder(tmp_path, task="handover_tshape", session_tag="s2", session_meta={})
    rec.start({}); rec.append(_step(0)); p0 = rec.finish(success=False)
    assert p0.name == "ep_000_fail.pickle"
    rec.start({}); rec.append(_step(0)); p1 = rec.abort()
    assert p1.name == "ep_001_aborted.pickle"
    rec.start({}); assert rec.abort() is None            # empty episode -> nothing written
    rec.start({}); rec.append(_step(0)); rec.discard()
    assert not rec.active
    rec.start({}); rec.append(_step(0)); p2 = rec.finish(success=True)
    assert p2.name == "ep_002_ok.pickle"                 # discard does not consume a number


def test_outcome_not_in_name_when_disabled(tmp_path):
    rec = EpisodeRecorder(tmp_path, task="t", session_tag="s", session_meta={}, outcome_in_name=False)
    rec.start({})
    rec.append(_step(0))
    p = rec.finish(success=False)
    assert p.name == "ep_000.pickle"                 # no label in the name; the flags stay inside
    ep = pickle.load(open(p, "rb"))
    assert ep["success"] is False
    rec.start({})
    rec.append(_step(0))
    assert rec.finish(success=True).name == "ep_001.pickle"
    rec.start({})
    rec.append(_step(0))
    assert rec.abort().name == "ep_002_aborted.pickle"   # aborted stays labeled (incomplete data)


def test_v2_fields_labels_and_stage_peak(tmp_path):
    rec = EpisodeRecorder(tmp_path, task="handover_tshape", session_tag="s3", session_meta={})
    rec.start({})
    rec.append(_step(0, v2=True, success_env=False, stage=(0, 0)))
    rec.append(_step(1, v2=True, success_env=False, stage=(2, 1)))
    rec.append(_step(2, v2=True, success_env=True, stage=(5, 4)))
    rec.append(_step(3, v2=True, success_env=True, stage=(5, 5)))
    # env says success, operator says the motion was bad -> manual label names the file
    p = rec.finish(success_env=True, success_manual=False)
    assert p.name == "ep_000_fail.pickle"
    ep = pickle.load(open(p, "rb"))
    assert ep["success_env"] is True and ep["success_manual"] is False and ep["success"] is True
    assert ep["palm_pos"].shape == (4, 2, 3) and ep["fingertip_pos"].shape == (4, 2, 4, 3)
    assert ep["subgoal"].shape == (4, 2) and ep["success_env_step"].dtype == np.bool_
    assert ep["metadata"]["stage_peak"] == [5.0, 5.0]
    assert ep["metadata"]["t_success_env"] == 2
    # operator OK, env never latched (forgot the home-return tail)
    rec.start({}); rec.append(_step(0, v2=True)); p2 = rec.finish(success_env=False, success_manual=True)
    assert p2.name == "ep_001_ok.pickle"
    ep2 = pickle.load(open(p2, "rb"))
    assert ep2["success_env"] is False and ep2["success_manual"] is True and ep2["metadata"]["t_success_env"] == -1

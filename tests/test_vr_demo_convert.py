"""VR pickle -> Demo_success layout: cut at env success + tail, T = P95, hold-pad (no Isaac)."""
import unittest

import numpy as np


def make_ep(T, latch_at=None, manual=True, obj="rod"):
    t = np.arange(T, dtype=np.float32)
    ep = {
        "T": T,
        "observations": np.broadcast_to(t[:, None, None], (T, 2, 74)).copy(),
        "share_observations": np.broadcast_to(t[:, None, None], (T, 2, 74)).copy(),
        "actions": np.broadcast_to(0.01 * t[:, None, None], (T, 2, 22)).copy(),
        "raw_actions": np.broadcast_to(t[:, None, None], (T, 2, 22)).copy(),
        "joint_positions": np.zeros((T, 2, 22), np.float32),
        "joint_velocities": np.zeros((T, 2, 22), np.float32),
        "object_pos": np.tile([-0.24, 0.45, 0.99], (T, 1)).astype(np.float32),
        "object_quat": np.tile([0.7071, 0, 0.7071, 0], (T, 1)).astype(np.float32),
        "success_env_step": np.zeros(T, bool),
        "success_env": latch_at is not None, "success_manual": manual,
        "metadata": {"object_id": obj, "schema_version": 2},
    }
    if latch_at is not None:
        ep["success_env_step"][latch_at:] = True
    return ep


class ConvertTests(unittest.TestCase):
    def test_cut_index(self):
        from trl_utils.vr_demo_convert import episode_cut_index
        self.assertEqual(episode_cut_index(make_ep(500, latch_at=300), tail=30), 331)
        self.assertEqual(episode_cut_index(make_ep(500, latch_at=490), tail=30), 500)   # clipped
        self.assertEqual(episode_cut_index(make_ep(500, latch_at=None), tail=30), 500)  # never latched
        legacy = make_ep(120, latch_at=None); del legacy["success_env_step"]
        self.assertEqual(episode_cut_index(legacy, tail=30), 120)

    def test_choose_T(self):
        from trl_utils.vr_demo_convert import choose_T
        self.assertEqual(choose_T([100, 200, 300, 400, 1000], pct=95.0, round_to=10), 880)   # linear P95 = 880
        self.assertEqual(choose_T([531], pct=95.0, round_to=10), 540)

    def test_cut_and_pad_holds_last_row(self):
        from trl_utils.vr_demo_convert import cut_and_pad
        a = np.arange(10, dtype=np.float32)[:, None] * np.ones((1, 3), np.float32)
        out = cut_and_pad(a, cut=6, T=9)
        self.assertEqual(out.shape, (9, 3))
        np.testing.assert_allclose(out[:6, 0], np.arange(6))
        np.testing.assert_allclose(out[6:, 0], 5.0)          # hold last kept row
        np.testing.assert_allclose(cut_and_pad(a, cut=10, T=4)[:, 0], np.arange(4))   # truncate

    def test_accept_rules(self):
        from trl_utils.vr_demo_convert import accept
        self.assertTrue(accept(make_ep(10, manual=True)))
        self.assertFalse(accept(make_ep(10, manual=False)))
        self.assertTrue(accept(make_ep(10, manual=False), include_manual_fail=True))
        v1 = make_ep(10); del v1["success_manual"]; del v1["success_env"]; v1["success"] = True
        self.assertTrue(accept(v1))                           # v1 pickle: env flag decides

    def test_build_payloads_layout(self):
        from trl_utils.vr_demo_convert import build_payloads
        eps = [make_ep(400, latch_at=300), make_ep(600, latch_at=500, obj="tumbler"), make_ep(2000, latch_at=1900)]
        bc, traj, rep = build_payloads(eps, T=560, task="handover_tshape", tag="unit", tail=30)
        self.assertEqual(rep["n_in"], 3); self.assertEqual(rep["n_kept"], 2); self.assertEqual(rep["n_dropped_long"], 1)
        self.assertEqual(rep["T"], 560); self.assertEqual(sorted(rep["object_ids"]), ["rod", "tumbler"])
        self.assertEqual(len(bc["observations"]), 560)
        self.assertEqual(bc["observations"][0].shape, (2, 2, 74))            # (N, agents, obs)
        self.assertEqual(bc["actions"][559].shape, (2, 2, 22))
        self.assertEqual(bc["metadata"]["episode_length"], 560)
        self.assertEqual(bc["metadata"]["num_envs"], 2)
        self.assertEqual(bc["metadata"]["source"], "vr")
        # episode 0 was cut at 331 and padded: step 330 is the last real row, 331.. hold it
        np.testing.assert_allclose(bc["observations"][330][0, 0, 0], 330.0)
        np.testing.assert_allclose(bc["observations"][559][0, 0, 0], 330.0)
        # episode 1 cut at 531
        np.testing.assert_allclose(bc["observations"][559][1, 0, 0], 530.0)
        self.assertEqual(traj["init_cube_pos"].shape, (2, 3))
        self.assertEqual(len(traj["raw_actions"]), 560)
        self.assertEqual(traj["metadata"]["episode_length"], 560)


if __name__ == "__main__":
    unittest.main()

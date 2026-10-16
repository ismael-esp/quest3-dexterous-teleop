"""SnapshotSource.base(): the per-step record shared by both teleop scripts (fakes, no Isaac)."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

from isaac_teleop.paths import sim_root  # noqa: E402

TWS = sim_root()
if TWS not in sys.path:
    sys.path.insert(0, TWS)

from isaac_teleop.snapshot import SnapshotSource  # noqa: E402


class _Robot:
    def __init__(self, prefix, base):
        names = [f"{prefix}_link_{i}_0" for i in (3, 7, 11, 15)]
        self._names = names
        n = len(names)
        self.data = SimpleNamespace(
            joint_pos=torch.arange(22, dtype=torch.float32).view(1, 22) + base,
            joint_vel=torch.zeros(1, 22),
            body_pos_w=torch.tensor([[[1.0, 2.0, 3.0]] * n]),
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * n]),
        )

    def find_bodies(self, keys, preserve_order=False):
        import re
        ids = [next(i for i, n in enumerate(self._names) if re.fullmatch(k, n)) for k in keys]
        return ids, [self._names[i] for i in ids]


def _fake_raw():
    return SimpleNamespace(
        env_origins=torch.tensor([[10.0, 0.0, 0.0]]),
        right_robot=_Robot("ra", 0.0), left_robot=_Robot("la", 100.0),
        cube=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[10.2, 0.4, 1.0]]),
                                                  root_quat_w=torch.tensor([[0.7071, 0.0, 0.7071, 0.0]]))),
        subgoal=torch.tensor([[3.0, 2.0]]), success_buf=torch.tensor([True]),
    )


def _fake_ctrl():
    pR = torch.tensor([[10.5, 0.1, 1.2]]); qR = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    pL = torch.tensor([[9.5, 0.1, 1.2]]); qL = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    return SimpleNamespace(
        teleop=SimpleNamespace(_compute_frame_pose_world=lambda: (pR, qR, pL, qL)),
        target_pose_world=lambda side: (np.array([11.0, 0.0, 1.0]) if side == "right" else np.array([9.0, 0.0, 1.0]),
                                        np.array([1.0, 0.0, 0.0, 0.0])),
        engaged=lambda side: side == "right",
    )


def test_base_snapshot_fields():
    raw, ctrl = _fake_raw(), _fake_ctrl()
    src = SnapshotSource(raw, ctrl)
    obs = torch.zeros(1, 2, 74); share = torch.zeros(1, 2, 74); act = torch.zeros(1, 2, 22); raw_act = torch.zeros(1, 2, 22)
    d = src.base(obs, share, act, raw_act, t_now=123.0)
    assert d["observations"].shape == (2, 74) and d["actions"].shape == (2, 22)
    assert d["joint_positions"].shape == (2, 22) and float(d["joint_positions"][1, 0]) == 100.0
    np.testing.assert_allclose(d["object_pos"], [0.2, 0.4, 1.0], atol=1e-6)          # env origin removed
    np.testing.assert_allclose(d["ee_pos"][0], [0.5, 0.1, 1.2], atol=1e-6)
    np.testing.assert_allclose(d["ee_pos"][1], [-0.5, 0.1, 1.2], atol=1e-6)
    # palm = wrist + constant (identity wrist rotation)
    np.testing.assert_allclose(d["palm_pos"][0], [0.5 - 0.0015, 0.1 + 0.0095, 1.2 + 0.095], atol=1e-6)
    np.testing.assert_allclose(d["palm_pos"][1], [-0.5 + 0.0015, 0.1 + 0.0095, 1.2 + 0.095], atol=1e-6)
    assert d["palm_quat"].shape == (2, 4)
    np.testing.assert_allclose(d["target_wrist_pos"][0], [1.0, 0.0, 1.0], atol=1e-6)   # origin removed
    np.testing.assert_allclose(d["target_wrist_quat"][1], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    assert d["fingertip_pos"].shape == (2, 4, 3)
    np.testing.assert_allclose(d["fingertip_pos"][0, 0], [1.0 - 10.0, 2.0, 3.0 + 0.0267], atol=1e-6)
    np.testing.assert_allclose(d["subgoal"], [3.0, 2.0])
    assert d["success_env_step"] is True
    assert d["engaged"].tolist() == [1, 0] and d["t_wall"] == 123.0
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            assert v.dtype == np.float32, k

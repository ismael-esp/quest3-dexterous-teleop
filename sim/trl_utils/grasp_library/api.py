"""Grasp library: storage, pair sampling, and object-frame -> world transforms.

This is the layer the demo generator talks to. It hides where grasps came from
(analytic today, BODex later) behind one interface, so swapping the source does not
touch `generate_demo.py`.

Why object-frame storage matters: the task randomises object pose every reset. A
force-closure grasp is a property of the OBJECT, so a new pose is a rigid transform of
an existing entry (microseconds) rather than a fresh ~4-5 s BODex synthesis. That is
what makes per-reset grasping affordable at 500 envs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .types import NUM_HAND_JOINTS, GraspEntry


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by WXYZ quaternions. q: [...,4], v: [...,3] -> [...,3]."""
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of WXYZ quaternions, batched."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


class GraspLibrary:
    """An indexed set of object-frame grasps for one object."""

    def __init__(self, entries: list[GraspEntry], object_id: str = "handover_cylinder"):
        if not entries:
            raise ValueError("GraspLibrary requires at least one entry")
        self.object_id = str(object_id)
        self.entries = list(entries)
        self._by_role: dict[str, list[int]] = {}
        for i, e in enumerate(self.entries):
            self._by_role.setdefault(e.role, []).append(i)
        for role in ("giver", "receiver"):
            if not self._by_role.get(role):
                raise ValueError(
                    f"library has no '{role}' entries -- the handover needs both roles"
                )

    def __len__(self) -> int:
        return len(self.entries)

    def counts(self) -> dict[str, int]:
        return {r: len(ix) for r, ix in self._by_role.items()}

    def sources(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e.source] = out.get(e.source, 0) + 1
        return out

    # ---- sampling ---------------------------------------------------------------

    def sample_pair(self, rng: np.random.Generator) -> tuple[GraspEntry, GraspEntry]:
        """Sample a (giver, receiver) pair that occupies OPPOSITE ends.

        End-exclusivity is verified, not assumed: a pair on the same end makes the
        handover geometrically unsolvable and would silently poison the demos.
        """
        givers = self._by_role["giver"]
        receivers = self._by_role["receiver"]
        for _ in range(32):
            g = self.entries[int(rng.choice(givers))]
            r = self.entries[int(rng.choice(receivers))]
            if g.opposes(r):
                return g, r
        raise RuntimeError(
            "could not sample an opposing giver/receiver pair after 32 tries -- "
            "check that the library's axis_coord signs differ by role"
        )

    def sample_pairs(self, n: int, rng: np.random.Generator):
        return [self.sample_pair(rng) for _ in range(n)]

    # ---- transforms -------------------------------------------------------------

    @staticmethod
    def to_world(
        palm_pos: torch.Tensor,
        palm_quat: torch.Tensor,
        obj_pos: torch.Tensor,
        obj_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Object-frame palm pose -> world, batched over envs.

        Args:
            palm_pos:  [N,3] palm position in the object frame.
            palm_quat: [N,4] palm orientation in the object frame (WXYZ).
            obj_pos:   [N,3] object position in world.
            obj_quat:  [N,4] object orientation in world (WXYZ).

        Returns:
            (world_pos [N,3], world_quat [N,4])
        """
        world_pos = obj_pos + quat_rotate(obj_quat, palm_pos)
        world_quat = quat_mul(obj_quat, palm_quat)
        return world_pos, world_quat

    def batch(self, entries: list[GraspEntry], device="cpu"):
        """Stack entries into tensors: (palm_pos [N,3], palm_quat [N,4], joints [N,16])."""
        pos = torch.tensor(np.stack([e.palm_pos for e in entries]), dtype=torch.float32, device=device)
        quat = torch.tensor(np.stack([e.palm_quat for e in entries]), dtype=torch.float32, device=device)
        joints = torch.tensor(np.stack([e.hand_joints for e in entries]), dtype=torch.float32, device=device)
        return pos, quat, joints

    # ---- persistence ------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            object_id=np.array(self.object_id),
            role=np.array([e.role for e in self.entries]),
            palm_pos=np.stack([e.palm_pos for e in self.entries]),
            palm_quat=np.stack([e.palm_quat for e in self.entries]),
            hand_joints=np.stack([e.hand_joints for e in self.entries]),
            axis_coord=np.array([e.axis_coord for e in self.entries]),
            quality=np.array([e.quality for e in self.entries]),
            source=np.array([e.source for e in self.entries]),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GraspLibrary":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no grasp library at {path}")
        d = np.load(path, allow_pickle=False)
        entries = [
            GraspEntry(
                role=str(d["role"][i]),
                palm_pos=d["palm_pos"][i],
                palm_quat=d["palm_quat"][i],
                hand_joints=d["hand_joints"][i],
                axis_coord=float(d["axis_coord"][i]),
                quality=float(d["quality"][i]),
                source=str(d["source"][i]),
            )
            for i in range(len(d["role"]))
        ]
        return cls(entries, object_id=str(d["object_id"]))

    def summary(self) -> str:
        c = self.counts()
        s = self.sources()
        return (
            f"GraspLibrary({self.object_id}): {len(self)} entries "
            f"[{', '.join(f'{k}={v}' for k, v in sorted(c.items()))}] "
            f"sources[{', '.join(f'{k}={v}' for k, v in sorted(s.items()))}] "
            f"joints/entry={NUM_HAND_JOINTS}"
        )

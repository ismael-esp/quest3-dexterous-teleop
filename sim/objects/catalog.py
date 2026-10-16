"""One `ObjectSpec` per object drives the Isaac spawn, the BODex mesh id and the handover
reference points. Pure python at import time; the USD branch of `apply_to_cube_cfg` imports
Isaac lazily so tests and the converter can use the catalog without a simulator.

Frames: every mesh is CENTRED at its geometric centre with its long axis along mesh +z. The
spawn rotation (0.7071, 0, 0.7071, 0) = 90 deg about y lays mesh +z along world +x, exactly as
the historical rod; +axis is the RECEIVER (right arm) end. `rod` reproduces the historical
CylinderCfg numbers bit-for-bit (r 0.027, L 0.30, 0.25 kg, z 0.99), so existing checkpoints and
demos are unaffected.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

# The rod (r = 0.027) rests at z = 0.99 on the two supporter pegs -> its underside is at 0.963.
# Every object is placed so its underside lands on that same height.
TABLE_TOP_Z = 0.963
# Historical reference points for the rod (L = 0.30): receiver contact at +0.11 along the axis,
# giver grasp offset at -0.09. Kept as fractions so other lengths scale the same way.
RECEIVER_CONTACT_FRAC = 0.11 / 0.30
GIVER_OFFSET_FRAC = 0.09 / 0.30
# End-exclusivity (spec §5): each hand needs `m` from the split plane plus a finger span.
HANDOVER_SPLIT_MARGIN_M = 0.055
HANDOVER_FINGER_SPAN_M = 0.09


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    kind: str                          # "cylinder" (Isaac primitive) | "usd" (UsdFileCfg)
    radius: float                      # half-extent ACROSS the axis: resting height above the table
    length_along_axis: float           # metres along the handover axis (mesh +z)
    mass: float
    obj_path: str                      # mesh for BODex, relative to the thesis root
    usd_path: str | None               # Isaac asset relative to the thesis root; None for primitives
    bodex_id: str                      # id under BODex custom_objs / object/custom (P-C onboards it)
    scale: tuple = (1.0, 1.0, 1.0)     # applied identically to USD and OBJ
    spawn_quat_wxyz: tuple = (0.7071, 0.0, 0.7071, 0.0)
    handover_axis: tuple = (0.0, 0.0, 1.0)
    init_xy: tuple = (-0.2425, 0.45)
    notes: str = ""

    def init_pos(self) -> tuple[float, float, float]:
        return (float(self.init_xy[0]), float(self.init_xy[1]), TABLE_TOP_Z + float(self.radius))

    def receiver_contact_local(self) -> tuple[float, float, float]:
        k = RECEIVER_CONTACT_FRAC * float(self.length_along_axis)
        ax = self.handover_axis
        return (k * ax[0], k * ax[1], k * ax[2])

    def giver_axis_offset(self) -> float:
        return -GIVER_OFFSET_FRAC * float(self.length_along_axis)

    def check_handover_feasible(self, m: float = HANDOVER_SPLIT_MARGIN_M,
                                finger_span: float = HANDOVER_FINGER_SPAN_M) -> None:
        need = 2.0 * m + finger_span
        if float(self.length_along_axis) < need:
            raise ValueError(
                f"object '{self.object_id}' is {self.length_along_axis:.3f} m along its handover axis; "
                f"two Allegro hands need >= {need:.3f} m for end-exclusive grasps (spec §5)."
            )

    def to_dict(self) -> dict:
        return asdict(self)


ROD = ObjectSpec(
    object_id="rod", kind="cylinder", radius=0.027, length_along_axis=0.30, mass=0.25,
    obj_path="assets/meshes/handover_cylinder.obj", usd_path=None, bodex_id="handover_cylinder",
    notes="historical yellow cardboard rod; CylinderCfg in tasks/task_handover_tshape.py",
)
ROD_SLIM = ObjectSpec(
    object_id="rod_slim", kind="cylinder", radius=0.018, length_along_axis=0.30,
    mass=0.25 * (0.018 / 0.027) ** 2,
    obj_path="assets/meshes/handover_cylinder_slim.obj", usd_path=None, bodex_id="handover_cylinder_slim",
    notes="rod with r=0.018 (mass scaled by cross-section); scripts/assets/make_handover_cylinder_obj.py --radius 0.018",
)
TUMBLER = ObjectSpec(
    object_id="tumbler", kind="usd", radius=0.040, length_along_axis=0.25, mass=0.25,
    obj_path="assets/objects/tumbler/tumbler_centered.obj",
    usd_path="assets/objects/tumbler/tumbler_centered.usd", bodex_id="tumbler_centered",
    notes="BODex custom_objs/tumbler.obj (OBB 0.080 x 0.080 x 0.250), re-centred; lies on its side like the rod",
)

CATALOG: dict[str, ObjectSpec] = {s.object_id: s for s in (ROD, ROD_SLIM, TUMBLER)}


def get_object(object_id: str) -> ObjectSpec:
    try:
        return CATALOG[str(object_id)]
    except KeyError:
        raise KeyError(f"unknown object_id {object_id!r}; known: {sorted(CATALOG)}") from None


def apply_to_cube_cfg(spec: ObjectSpec, cube_cfg, base_dir: str):
    """Mutate the task's `cube_cfg` (a RigidObjectCfg) in place to spawn `spec`.

    Cylinder specs only touch radius/height/mass on the existing CylinderCfg, so `rod` leaves the
    historical cfg untouched. USD specs replace `spawn` with a UsdFileCfg (Isaac import is lazy).
    """
    if spec.kind == "cylinder":
        cube_cfg.spawn.radius = float(spec.radius)
        cube_cfg.spawn.height = float(spec.length_along_axis)
        cube_cfg.spawn.mass_props.mass = float(spec.mass)
    elif spec.kind == "usd":
        import isaaclab.sim as sim_utils  # lazy: only inside a simulator process
        cube_cfg.spawn = sim_utils.UsdFileCfg(
            usd_path=os.path.join(base_dir, spec.usd_path),
            scale=tuple(float(s) for s in spec.scale),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0005),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(spec.mass)),
        )
    else:
        raise ValueError(f"unknown spec.kind {spec.kind!r}")
    cube_cfg.init_state.pos = spec.init_pos()
    cube_cfg.init_state.rot = tuple(float(v) for v in spec.spawn_quat_wxyz)
    return cube_cfg

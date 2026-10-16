"""Object catalog: spawn numbers, handover reference points, feasibility rule (no Isaac)."""
import unittest
from types import SimpleNamespace

import numpy as np


class CatalogTests(unittest.TestCase):
    def test_rod_reproduces_the_historical_cylinder(self):
        from objects.catalog import get_object
        rod = get_object("rod")
        self.assertEqual(rod.kind, "cylinder")
        self.assertAlmostEqual(rod.radius, 0.027)
        self.assertAlmostEqual(rod.length_along_axis, 0.30)
        self.assertAlmostEqual(rod.mass, 0.25)
        np.testing.assert_allclose(rod.init_pos(), (-0.2425, 0.45, 0.99), atol=1e-9)
        self.assertEqual(rod.spawn_quat_wxyz, (0.7071, 0.0, 0.7071, 0.0))
        np.testing.assert_allclose(rod.receiver_contact_local(), (0.0, 0.0, 0.11), atol=1e-9)
        self.assertAlmostEqual(rod.giver_axis_offset(), -0.09, places=9)
        self.assertEqual(rod.bodex_id, "handover_cylinder")

    def test_slim_rod_and_tumbler(self):
        from objects.catalog import get_object
        slim = get_object("rod_slim")
        self.assertAlmostEqual(slim.radius, 0.018)
        self.assertAlmostEqual(slim.mass, 0.25 * (0.018 / 0.027) ** 2, places=6)
        self.assertAlmostEqual(slim.init_pos()[2], 0.963 + 0.018, places=9)
        tum = get_object("tumbler")
        self.assertEqual(tum.kind, "usd")
        self.assertEqual(tum.usd_path, "assets/objects/tumbler/tumbler_centered.usd")
        self.assertAlmostEqual(tum.length_along_axis, 0.25)
        self.assertAlmostEqual(tum.init_pos()[2], 0.963 + 0.040, places=9)

    def test_ids_and_unknown(self):
        from objects.catalog import CATALOG, get_object
        self.assertEqual(sorted(CATALOG), ["rod", "rod_slim", "tumbler"])
        with self.assertRaises(KeyError):
            get_object("banana")

    def test_handover_feasibility_rule(self):
        from objects.catalog import CATALOG, ObjectSpec
        for spec in CATALOG.values():
            spec.check_handover_feasible()           # all three are elongated enough
        short = ObjectSpec(object_id="puck", kind="cylinder", radius=0.03, length_along_axis=0.15,
                           mass=0.1, obj_path="x.obj", usd_path=None, bodex_id="puck")
        with self.assertRaises(ValueError):
            short.check_handover_feasible()

    def test_apply_cylinder_to_fake_cfg(self):
        from objects.catalog import apply_to_cube_cfg, get_object
        cfg = SimpleNamespace(spawn=SimpleNamespace(radius=0.0, height=0.0, mass_props=SimpleNamespace(mass=0.0)),
                              init_state=SimpleNamespace(pos=(0, 0, 0), rot=(1, 0, 0, 0)))
        apply_to_cube_cfg(get_object("rod_slim"), cfg, base_dir="/nowhere")
        self.assertAlmostEqual(cfg.spawn.radius, 0.018)
        self.assertAlmostEqual(cfg.spawn.height, 0.30)
        self.assertAlmostEqual(cfg.spawn.mass_props.mass, 0.25 * (0.018 / 0.027) ** 2, places=6)
        np.testing.assert_allclose(cfg.init_state.pos, (-0.2425, 0.45, 0.981), atol=1e-9)
        self.assertEqual(cfg.init_state.rot, (0.7071, 0.0, 0.7071, 0.0))

    def test_to_dict_is_json_safe(self):
        import json
        from objects.catalog import get_object
        json.dumps(get_object("tumbler").to_dict())


if __name__ == "__main__":
    unittest.main()

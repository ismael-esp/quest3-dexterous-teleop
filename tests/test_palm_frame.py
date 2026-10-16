"""wrist_3 -> palm constants and fingertip helper (pure numpy, no Isaac)."""
import unittest

import numpy as np


class PalmFrameTests(unittest.TestCase):
    def test_identity_wrist_gives_the_constant(self):
        from trl_utils.palm_frame import WRIST3_TO_PALM_POS, WRIST3_TO_PALM_QUAT, palm_from_wrist3
        for side in ("right", "left"):
            p, q = palm_from_wrist3(np.zeros(3), np.array([1.0, 0, 0, 0]), side)
            np.testing.assert_allclose(p, WRIST3_TO_PALM_POS[side], atol=1e-9)
            np.testing.assert_allclose(q, WRIST3_TO_PALM_QUAT[side], atol=1e-9)

    def test_palm_offset_magnitude_and_rotation(self):
        from trl_utils.palm_frame import WRIST3_TO_PALM_POS, WRIST3_TO_PALM_QUAT
        for side in ("right", "left"):
            self.assertAlmostEqual(float(np.linalg.norm(WRIST3_TO_PALM_POS[side])), 0.0955, places=3)
            w, x, y, z = WRIST3_TO_PALM_QUAT[side]
            angle = np.degrees(2.0 * np.arccos(np.clip(w, -1, 1)))
            axis = np.array([x, y, z]) / max(np.linalg.norm([x, y, z]), 1e-12)
            self.assertAlmostEqual(angle, 90.0, delta=0.2)
            self.assertGreater(abs(axis[2]), 0.999)          # rotation about the wrist z axis

    def test_round_trip_random_poses(self):
        from trl_utils.palm_frame import palm_from_wrist3, wrist3_from_palm
        rng = np.random.default_rng(0)
        for side in ("right", "left"):
            for _ in range(20):
                p = rng.normal(size=3)
                q = rng.normal(size=4); q /= np.linalg.norm(q)
                pp, qp = palm_from_wrist3(p, q, side)
                p2, q2 = wrist3_from_palm(pp, qp, side)
                np.testing.assert_allclose(p2, p, atol=1e-9)
                self.assertAlmostEqual(abs(float(np.dot(q2, q))), 1.0, places=9)

    def test_fingertips_from_links(self):
        from trl_utils.palm_frame import FINGERTIP_LINKS, FINGERTIP_OFFSET, fingertips_from_links
        self.assertEqual(FINGERTIP_LINKS["right"], ["ra_link_3_0", "ra_link_7_0", "ra_link_11_0", "ra_link_15_0"])
        self.assertEqual(FINGERTIP_LINKS["left"], ["la_link_3_0", "la_link_7_0", "la_link_11_0", "la_link_15_0"])
        p = np.arange(12, dtype=float).reshape(4, 3)
        q = np.tile([1.0, 0, 0, 0], (4, 1))
        tips = fingertips_from_links(p, q)
        np.testing.assert_allclose(tips, p + FINGERTIP_OFFSET, atol=1e-12)
        # 90 deg about x maps local z to world -y
        q90x = np.tile([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0], (4, 1))
        tips2 = fingertips_from_links(np.zeros((4, 3)), q90x)
        np.testing.assert_allclose(tips2, np.tile([0, -0.0267, 0], (4, 1)), atol=1e-9)


if __name__ == "__main__":
    unittest.main()

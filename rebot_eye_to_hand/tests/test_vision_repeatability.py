import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from eyehand.geometry import transform
from scripts.check_vision_stability import repeatability


class RepeatabilityTests(unittest.TestCase):
    def test_constant_marker_has_zero_jitter(self):
        t=transform(Rotation.from_euler("xyz",[.2,-.3,.1]).as_matrix(),[.1,.2,.5])
        r=repeatability([{"T_camera_target":t.tolist()} for _ in range(60)])
        self.assertLess(r["translation_jitter_mm"]["max"],1e-10)
        self.assertLess(r["rotation_jitter_deg"]["max"],1e-10)

    def test_known_translation_and_rotation_spread_and_drift(self):
        observations=[]
        for sign in [-1.]*30+[1.]*30:
            t=transform(Rotation.from_euler("z",sign,degrees=True).as_matrix(),[sign*.002,0,.5])
            observations.append({"T_camera_target":t.tolist()})
        r=repeatability(observations)
        self.assertAlmostEqual(r["translation_jitter_mm"]["rms"],2.)
        self.assertAlmostEqual(r["rotation_jitter_deg"]["rms"],1.)
        np.testing.assert_allclose(r["first_vs_last_third_drift_mm_deg"],[4.,2.],atol=1e-10)

    def test_empty_observations_rejected(self):
        with self.assertRaises(ValueError):
            repeatability([])

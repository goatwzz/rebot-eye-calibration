from __future__ import annotations

import copy
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from eyehand.cli import synthetic_dataset
from eyehand.geometry import errors, evaluate, solve


class ReprojectionRefinementTests(unittest.TestCase):
    def make_data(self, seed=7):
        data, x, y = synthetic_dataset(seed=seed, count=25)
        obj = np.array([[-.03, .03, 0], [.03, .03, 0], [.03, -.03, 0], [-.03, -.03, 0]])
        for s in data["samples"]:
            c = np.asarray(s["T_camera_target"])
            uv, _ = cv2.projectPoints(obj, cv2.Rodrigues(c[:3, :3])[0], c[:3, 3],
                                      np.asarray(data["K"], dtype=float), np.asarray(data["D"]))
            s.update(object_points=obj.tolist(), image_points=uv.reshape(-1, 2).tolist(), K=data["K"], D=data["D"])
        return data, x, y

    def test_exact_recovery_and_independent_image_corner_prediction(self):
        data, x, y = self.make_data()
        before = copy.deepcopy(data)
        result = solve(data["samples"], refinement="reprojection")
        np.testing.assert_allclose(result["T_base_camera"], x, atol=1e-8)
        np.testing.assert_allclose(result["T_ee_target"], y, atol=1e-8)
        self.assertEqual(data, before)
        test, _, _ = self.make_data(seed=44)
        report = evaluate(test["samples"], np.asarray(result["T_base_camera"]), np.asarray(result["T_ee_target"]))
        self.assertLess(report["summary"]["predicted_reprojection_rms_px"]["max"], 1e-5)

    def test_noisy_pnp_orientation_does_not_override_precise_image_corners(self):
        data, x, y = self.make_data()
        rng = np.random.default_rng(42)
        for s in data["samples"]:
            c = np.asarray(s["T_camera_target"])
            c[:3, :3] = c[:3, :3] @ Rotation.from_rotvec(rng.normal(0, np.radians(1.5), 3)).as_matrix()
            s["T_camera_target"] = c.tolist()
        result = solve(data["samples"], refinement="reprojection")
        self.assertLess(errors(np.asarray(result["T_base_camera"]), x)[0], 1e-4)
        self.assertLess(errors(np.asarray(result["T_ee_target"]), y)[1], 1e-5)
        self.assertEqual(len(result["training_ids"]), 25)
        self.assertEqual(result["optimization"]["samples_removed"], 0)
        # PnP rotation closure remains noisy; pixel fitting must not hide it.
        self.assertGreater(result["training"]["summary"]["rotation_deg"]["rms"], 1.)
        self.assertLess(result["training"]["summary"]["predicted_reprojection_rms_px"]["rms"], 1e-5)

    def test_missing_corners_invalid_observations_and_conflicting_options_rejected(self):
        data, _, _ = self.make_data()
        with self.assertRaisesRegex(ValueError, "no-refine"):
            solve(data["samples"], refine=False, refinement="reprojection")
        data["samples"][0]["image_points"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid reprojection"):
            solve(data["samples"], refinement="reprojection")
        del data["samples"][0]["image_points"]
        with self.assertRaisesRegex(ValueError, "needs object_points"):
            solve(data["samples"], refinement="reprojection")


if __name__ == "__main__":
    unittest.main()

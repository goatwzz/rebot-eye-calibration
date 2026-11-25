from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pinocchio as pin

from eyehand.cli import save, validate
from eyehand.geometry import evaluate
from eyehand.kinematics import CorrectedKinematics, corrected_dataset, export_corrected_urdf, identify_kinematics
import test_diagnostic_kinematics as diagnostic_fixture


class KinematicIdentificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        factory = diagnostic_fixture.KinematicsDiagnosticTests()
        cls.train, cls.x, cls.y, cls.offsets = factory.make_data(11, 30)
        cls.test, _, _, _ = factory.make_data(35, 10)
        cls.correction, cls.result = identify_kinematics(cls.train)

    def test_recovers_known_corrections_and_predicts_independent_poses(self):
        np.testing.assert_allclose(self.correction["joint_offsets_rad"], self.offsets, atol=1e-7)
        data = corrected_dataset(self.test, self.correction)
        report = evaluate(data["samples"], np.array(self.result["T_base_camera"]), np.array(self.result["T_ee_target"]))
        self.assertLess(report["summary"]["translation_mm"]["max"], 1e-4)
        self.assertLess(report["summary"]["predicted_reprojection_rms_px"]["max"], 1e-5)
        self.assertEqual(self.result["optimization"]["normalized_jacobian_rank"], 16)

    def test_exported_urdf_matches_original_with_offsets_at_random_configurations(self):
        before = Path(self.correction["original_urdf"]).read_bytes()
        corrected = CorrectedKinematics(self.train["robot"], self.correction)
        with tempfile.TemporaryDirectory() as directory:
            urdf = Path(directory) / "corrected.urdf"
            export_corrected_urdf(self.correction, urdf)
            model = pin.buildModelFromUrdf(str(urdf))
            state = model.createData()
            fid = model.getFrameId("gripper_end")
            rng = np.random.default_rng(52)
            for _ in range(100):
                raw = rng.uniform(-1.5, 1.5, 6)
                q = pin.neutral(model)
                q[:6] = raw
                pin.framesForwardKinematics(model, state, q)
                np.testing.assert_allclose(state.oMf[fid].homogeneous, corrected.fk(raw), atol=1e-12)
            self.assertAlmostEqual(sum(i.mass for i in model.inertias),sum(i.mass for i in corrected.model.inertias))
        self.assertEqual(Path(self.correction["original_urdf"]).read_bytes(), before)

    def test_correction_uses_raw_feedback_once_and_preserves_dataset(self):
        original = copy.deepcopy(self.test)
        once = corrected_dataset(self.test, self.correction)
        twice = corrected_dataset(once, self.correction)
        self.assertEqual(once, twice)
        self.assertEqual(original, self.test)
        changed = {**self.correction, "joint_offsets_rad": [0., 0., 0., 0., 0., 0.]}
        with self.assertRaisesRegex(ValueError, "different kinematic"):
            corrected_dataset(once, changed)

    def test_validation_applies_paired_correction_automatically(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            dataset = Path(directory) / "dataset.json"
            report = Path(directory) / "report.json"
            save(result, self.result)
            save(dataset, self.test)
            args = argparse.Namespace(result=str(result),dataset=str(dataset),output=str(report),
                                      translation_mm=5,rotation_deg=1,pixel_px=3)
            with contextlib.redirect_stdout(io.StringIO()):
                validate(args)
            import json
            saved = json.loads(report.read_text())
            self.assertEqual(saved["status"],"PASS_INTERNAL_CONSISTENCY")
            self.assertEqual(saved["kinematic_correction_applied"],self.correction)
            self.assertFalse(saved["physical_accuracy_verified"])

    def test_mismatched_model_gauge_offsets_and_missing_feedback_rejected(self):
        with self.assertRaisesRegex(ValueError, "different URDF"):
            CorrectedKinematics(self.train["robot"],{**self.correction,"original_urdf_sha256":"wrong"})
        with self.assertRaisesRegex(ValueError, "gauge"):
            CorrectedKinematics(self.train["robot"],{**self.correction,"joint_offsets_rad":[.1,0,0,0,0,0]})
        data = copy.deepcopy(self.test)
        del data["samples"][0]["robot_reads"]
        with self.assertRaisesRegex(ValueError, "raw joint feedback"):
            corrected_dataset(data,self.correction)
        data = copy.deepcopy(self.train)
        data["samples"][0]["T_base_ee"][0][3] += .01
        with self.assertRaisesRegex(ValueError, "nominal FK"):
            identify_kinematics(data)


if __name__ == "__main__":
    unittest.main()

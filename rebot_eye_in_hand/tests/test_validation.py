"""Offline geometry/provenance tests. No robot/camera instances are created."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wrist import validation as v
_, geo = v.dependencies(Path(__file__).resolve().parents[2] / "rebot_eye_to_hand")


def fixture(n=20, degenerate=False):
    rng = np.random.default_rng(17)
    x = geo.transform(Rotation.from_euler("xyz", [.2, -.3, .6]).as_matrix(), [-.117, .034, .106])
    y = geo.transform(Rotation.from_euler("xyz", [.1, -.2, .05]).as_matrix(), [.1, .05, .6])
    k = np.array([[906., 0, 640], [0, 905., 360], [0, 0, 1.]])
    obj = np.array([[-.03, .03, 0], [.03, .03, 0], [.03, -.03, 0], [-.03, -.03, 0]])
    samples = []
    for i in range(n):
        r = Rotation.from_rotvec(rng.uniform(-.35, .35, 3)).as_matrix() if not degenerate else np.eye(3)
        g = geo.transform(r, rng.uniform(-.05, .05, 3))
        c = geo.inverse(g @ x) @ y
        uv, _ = cv2.projectPoints(obj, cv2.Rodrigues(c[:3, :3])[0], c[:3, 3], k, np.zeros(5))
        samples.append({"id": str(i), "image": f"{i}.png", "T_base_ee": g.tolist(),
                        "T_camera_target": c.tolist(), "object_points": obj.tolist(),
                        "image_points": uv.reshape(-1, 2).tolist(), "K": k.tolist(), "D": [0.] * 5})
    return {"samples": samples, "synthetic": False}, x, y


class ValidatorTests(unittest.TestCase):
    def test_cli_defaults_follow_the_single_wrist_configuration(self):
        cli = Mock()
        def preview(args):
            self.assertEqual(args.serial, "332322072170")
            self.assertEqual(args.fps, 30)
            self.assertEqual(v.read_json(args.target)["marker_m"], .06)
            self.assertTrue(Path(args.reference_script).is_file())
        cli.preview.side_effect = preview
        with patch.object(v, "dependencies", return_value=(cli, geo)):
            v.main(["preview"])
        cli.preview.assert_called_once()

    def test_correct_transform_and_projection(self):
        d, x, y = fixture()
        r = v.evaluate(d, x, np.eye(4), geo)
        self.assertEqual(r["status"], "PASS_CONSISTENCY_ONLY")
        self.assertFalse(r["physical_accuracy_verified"])
        self.assertFalse(r["hand_eye_refitted"])
        self.assertLess(r["summary"]["translation_mm"]["max"], 1e-9)
        self.assertLess(r["summary"]["predicted_reprojection_rms_px"]["max"], 1e-8)
        np.testing.assert_allclose(r["mean_T_base_target_without_compensation"], y, atol=1e-12)

    def test_wrong_translation_is_detected(self):
        d, x, _ = fixture()
        x[:3, 3] += [.03, -.02, .01]
        r = v.evaluate(d, x, np.eye(4), geo)
        self.assertEqual(r["status"], "FAIL_CONSISTENCY")
        self.assertGreater(r["summary"]["translation_mm"]["p95"], 3.)

    def test_wrong_rotation_is_detected(self):
        d, x, _ = fixture()
        x[:3, :3] = x[:3, :3] @ Rotation.from_euler("z", 5, degrees=True).as_matrix()
        r = v.evaluate(d, x, np.eye(4), geo)
        self.assertEqual(r["status"], "FAIL_CONSISTENCY")

    def test_global_compensation_cannot_improve_consistency(self):
        d, x, _ = fixture()
        z = geo.transform(translation=[0, 0, -.031])
        r = v.evaluate(d, x, z, geo)
        before = np.array(r["mean_T_base_target_without_compensation"])
        after = np.array(r["mean_T_base_target_with_compensation"])
        np.testing.assert_allclose(after[:3, 3] - before[:3, 3], [0, 0, -.031])
        for row in r["samples"]:
            self.assertAlmostEqual(row["translation_mm"], row["with_compensation_translation_mm"], places=10)
            self.assertAlmostEqual(row["rotation_deg"], row["with_compensation_rotation_deg"], places=10)

    def test_degenerate_poses_cannot_pass(self):
        d, x, _ = fixture(degenerate=True)
        self.assertEqual(v.evaluate(d, x, np.eye(4), geo)["status"], "INSUFFICIENT_POSES")

    def test_few_poses_cannot_pass(self):
        d, x, _ = fixture(n=5)
        self.assertEqual(v.evaluate(d, x, np.eye(4), geo)["status"], "INSUFFICIENT_POSES")

    def test_synthetic_never_claims_hardware_pass(self):
        d, x, _ = fixture()
        d["synthetic"] = True
        self.assertEqual(v.evaluate(d, x, np.eye(4), geo)["status"], "SYNTHETIC_ONLY")

    def test_duplicate_samples_rejected(self):
        d, x, _ = fixture()
        d["samples"][1]["id"] = d["samples"][0]["id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            v.evaluate(d, x, np.eye(4), geo)

    def test_invalid_transform_rejected(self):
        d, x, _ = fixture()
        x[0, 0] += .1
        with self.assertRaises(ValueError):
            v.evaluate(d, x, np.eye(4), geo)

    def test_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "report.json"
            v.write_json(p, {"original": True})
            with self.assertRaises(FileExistsError):
                v.write_json(p, {"replacement": True})
            self.assertEqual(v.read_json(p), {"original": True})

    def test_candidate_metadata_and_joint_offset_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "candidate.npz"
            fields = dict(T_result=np.eye(4), mode=["eye_in_hand"], n_samples=[25], method=["PARK+reprojection"],
                          camera_serial=["wrist"], urdf_sha256=["model"], training_dataset_ids=["training"],
                          training_dataset_sha256=["training-hash"], joint_offsets_rad=np.zeros(6))
            np.savez(p, **fields)
            _, info = v.load_result(p, geo)
            self.assertEqual(info["camera_serial"], "wrist")
            self.assertEqual(info["training_dataset_ids"], ["training"])
            fields["joint_offsets_rad"] = np.array([0, .01, 0, 0, 0, 0])
            np.savez(p, **fields)
            with self.assertRaisesRegex(ValueError, "nonzero joint offsets"):
                v.load_result(p, geo)
            _, paired = v.load_result(p, geo, allow_joint_offsets=True)
            np.testing.assert_array_equal(paired["joint_offsets_rad"], fields["joint_offsets_rad"])

    def test_candidate_training_data_cannot_be_its_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            result = p / "hand_eye_under_test.npz"
            np.savez(result, T_result=np.eye(4), mode=["eye_in_hand"], n_samples=[25], method=["PARK+reprojection"],
                     training_dataset_ids=["same-training-id"], training_dataset_sha256=["other-hash"])
            v.write_json(p / "validation_session.json", {"purpose": "existing_wrist_eye_in_hand_validation", "result_sha256": v.sha256(result)})
            v.write_json(p / "dataset.json", {"dataset_id": "same-training-id"})
            with self.assertRaisesRegex(ValueError, "used to fit"):
                v.report(argparse.Namespace(dataset=tmp), geo)

    def test_camera_failure_retains_samples_and_generates_partial_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            dataset = {"samples": [{"id": "saved-sample"}]}
            v.write_json(directory / "dataset.json", dataset)
            cli = Mock()
            cli.collect.side_effect = RuntimeError("Frame didn't arrive within 1000")
            with patch.object(v, "report") as make_report:
                with self.assertRaisesRegex(RuntimeError, "Saved data retained"):
                    v.collect_and_report(cli, argparse.Namespace(), directory, geo)
                make_report.assert_called_once()
            self.assertEqual(v.read_json(directory / "dataset.json"), dataset)
            self.assertEqual(v.read_json(directory / "collection_end.json")["status"], "ERROR")

    def test_startup_failure_does_not_claim_samples_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cli = Mock()
            cli.collect.side_effect = RuntimeError("set_xu UVC control query timed out")
            with patch.object(v, "report") as make_report:
                with self.assertRaisesRegex(RuntimeError, "No samples were recorded"):
                    v.collect_and_report(cli, argparse.Namespace(), directory, geo)
                make_report.assert_not_called()
            end = v.read_json(directory / "collection_end.json")
            self.assertEqual(end["sample_count"], 0)
            self.assertFalse(end["dataset_created"])

    def test_offline_report_and_camera_provenance(self):
        d, x, _ = fixture()
        d.update(dataset_id="test", camera={"serial": "wrist"}, robot={"ee_frame": "gripper_end", "base_frame": "base_link", "urdf_sha256": "model"}, target={"type": "aruco"}, K=d["samples"][0]["K"])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            result = p / "hand_eye_under_test.npz"
            np.savez(result, T_result=x, mode=["eye_in_hand"], n_samples=[16], method=["TSAI"])
            v.write_json(p / "validation_session.json", {"purpose": "existing_wrist_eye_in_hand_validation", "result_sha256": v.sha256(result), "requested_camera_serial": "wrist", "urdf_sha256": "model", "runtime_compensation_base_m": [0, 0, -.031]})
            v.write_json(p / "target.json", d["target"])
            v.write_json(p / "dataset.json", d)
            args = argparse.Namespace(dataset=tmp, output=None, min_samples=15, translation_p95_mm=3., rotation_p95_deg=1., reprojection_p95_px=3.)
            with contextlib.redirect_stdout(io.StringIO()):
                r = v.report(args, geo)
            self.assertEqual(r["status"], "PASS_CONSISTENCY_ONLY")
            d["camera"]["serial"] = "external-camera"
            (p / "dataset.json").write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError, "serial"):
                v.report(args, geo)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eyehand.cli import main
from eyehand.gravity_parameters import DEFAULT_GRAVITY_CONFIG, load_gravity_parameters, set_gravity_parameters
from eyehand.reference_manual import ReferenceManualReader


class IndependentGravityTests(unittest.TestCase):
    def make_config(self, directory):
        path = Path(directory) / "gravity.json"
        path.write_bytes(DEFAULT_GRAVITY_CONFIG.read_bytes())
        return path

    def test_loads_only_selected_local_json_without_following_external_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_config(directory)
            expected = json.loads(path.read_text())
            expected["source_config"] = "/nonexistent/demo10/config.yaml"
            path.write_text(json.dumps(expected))
            reads = []
            read_bytes = Path.read_bytes

            def track(instance):
                reads.append(instance.resolve())
                return read_bytes(instance)

            with patch.object(Path, "read_bytes", track):
                result = load_gravity_parameters(path)
            self.assertEqual(reads, [path.resolve()])
            self.assertEqual(result["tau_scale"], expected["tau_scale"])
            self.assertEqual(result["config_path"], str(path.resolve()))
            self.assertNotIn("source_config", result)

    def test_invalid_update_does_not_modify_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_config(directory)
            before = path.read_bytes()
            for scale in [[1, 1], [1, 1, -1, 1, 1, 1], [1, 1, float("nan"), 1, 1, 1]]:
                with self.assertRaisesRegex(ValueError, "tau_scale"):
                    set_gravity_parameters(path, tau_scale=scale)
                self.assertEqual(path.read_bytes(), before)

    def test_cli_sets_one_joint_and_gain_without_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_config(directory)
            before = json.loads(path.read_text())
            argv = ["eyehand", "gravity", "set", "--config", str(path), "--joint", "2", "--scale", "1.11", "--kd", "1.2"]
            with patch.object(sys, "argv", argv), patch("eyehand.cli.Camera") as camera, \
                    patch("eyehand.cli.RebotReader") as robot, contextlib.redirect_stdout(io.StringIO()):
                main()
            after = load_gravity_parameters(path)
            expected = before["tau_scale"]
            expected[1] = 1.11
            self.assertEqual(after["tau_scale"], expected)
            self.assertEqual(after["kp"], before["kp"])
            self.assertEqual(after["kd"], 1.2)
            self.assertEqual(after["payload_mass_kg"], before["payload_mass_kg"])
            self.assertEqual(after["payload_com_m"], before["payload_com_m"])
            self.assertEqual(after["wrist_camera_mass_kg"], before["wrist_camera_mass_kg"])
            camera.assert_not_called()
            robot.assert_not_called()

    def test_cli_rejects_incomplete_joint_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_config(directory)
            before = path.read_bytes()
            with patch.object(sys, "argv", ["eyehand", "gravity", "set", "--config", str(path), "--joint", "2"]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()
            self.assertEqual(path.read_bytes(), before)

    def test_cli_sets_full_scale_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_config(directory)
            scale = [1., 1.03, .96, 1., 1., 1.]
            with patch.object(sys, "argv", ["eyehand", "gravity", "set", "--config", str(path), "--tau-scale", *map(str, scale)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(load_gravity_parameters(path)["tau_scale"], scale)

    def test_reference_start_prints_loaded_local_values(self):
        reader = ReferenceManualReader.__new__(ReferenceManualReader)
        reader.gravity_parameters = load_gravity_parameters()
        calls = []

        class Controller:
            def start(self):
                calls.append("original_start")

        reader.controller = Controller()
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            reader.start()
        self.assertEqual(calls, ["original_start"])
        self.assertIn("Gravity parameters", stream.getvalue())
        self.assertIn(json.dumps(reader.gravity_parameters["tau_scale"]), stream.getvalue())

    def test_cli_updates_camera_without_changing_aruco_or_scales(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_config(directory)
            before = load_gravity_parameters(path)
            argv = ["eyehand", "gravity", "set", "--config", str(path),
                    "--wrist-camera-mass-kg", "0.160", "--wrist-camera-frame", "link6",
                    "--wrist-camera-com-m", "0", "0", "0"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            after = load_gravity_parameters(path)
            self.assertEqual(after["wrist_camera_mass_kg"], .160)
            self.assertEqual(after["wrist_camera_frame"], "link6")
            self.assertEqual(after["payload_mass_kg"], before["payload_mass_kg"])
            self.assertEqual(after["tau_scale"], before["tau_scale"])


if __name__ == "__main__":
    unittest.main()

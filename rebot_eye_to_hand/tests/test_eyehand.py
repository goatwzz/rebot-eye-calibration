from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import tempfile
import time
import unittest
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from eyehand.cli import check_points, provenance, save, synthetic_dataset, validate
from eyehand.geometry import checked, errors, evaluate, inverse, solve, transform
from eyehand.hardware import DEFAULT_SDK, PoseFile, RebotReader, write_pose
from eyehand.manual import ManualGravityReader
from eyehand.vision import Target


class SolverTests(unittest.TestCase):
    def test_exact_eye_to_hand_and_holdout(self):
        train, x, y = synthetic_dataset(count=25)
        result = solve(train["samples"])
        np.testing.assert_allclose(result["T_base_camera"], x, atol=1e-8)
        np.testing.assert_allclose(result["T_ee_target"], y, atol=1e-8)
        test, _, _ = synthetic_dataset(seed=44, count=10)
        report = evaluate(test["samples"], np.asarray(result["T_base_camera"]), np.asarray(result["T_ee_target"]))
        self.assertLess(report["summary"]["translation_mm"]["max"], 1e-5)
        self.assertGreater(evaluate(test["samples"], inverse(x), y)["summary"]["translation_mm"]["rms"], 100)

    def test_noisy_recovery(self):
        data, x, _ = synthetic_dataset(count=35, noisy=True)
        result = solve(data["samples"])
        mm, deg = errors(x, np.asarray(result["T_base_camera"]))
        self.assertLess(mm, 1)
        self.assertLess(deg, 0.1)

    def test_translation_only_and_one_rotation_axis_rejected(self):
        for rotation in [False, True]:
            data, _, _ = synthetic_dataset(count=20)
            for i, s in enumerate(data["samples"]):
                g = np.asarray(s["T_base_ee"])
                g[:3, :3] = Rotation.from_euler("z", i * 0.1 if rotation else 0).as_matrix()
                s["T_base_ee"] = g.tolist()
            with self.assertRaisesRegex(ValueError, "Degenerate"):
                solve(data["samples"])

    def test_invalid_transform_and_sample_count(self):
        t = np.eye(4)
        t[0, 0] = -1
        with self.assertRaises(ValueError):
            checked(t)
        data, _, _ = synthetic_dataset(count=4)
        with self.assertRaises(ValueError):
            solve(data["samples"])

    def test_outlier_report_preserves_samples(self):
        data, _, _ = synthetic_dataset(count=25)
        data["samples"][0]["T_camera_target"][0][3] += 0.1
        result = solve(data["samples"])
        self.assertEqual(len(result["training_ids"]), 25)
        self.assertTrue(any(data["samples"][0]["id"] in w for w in result["warnings"]))

    def test_validation_is_independent_and_does_not_refit_mount(self):
        train, _, _ = synthetic_dataset(count=25)
        result = solve(train["samples"])
        result.update(provenance=provenance(train), training_dataset_id=train["dataset_id"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "result.json", result)
            save(root / "validation.json", train)
            args = argparse.Namespace(result=str(root / "result.json"), dataset=str(root / "validation.json"),
                                      output=str(root / "report.json"), translation_mm=5, rotation_deg=1, pixel_px=3)
            with self.assertRaisesRegex(ValueError, "separately"):
                validate(args)
            test, _, _ = synthetic_dataset(seed=99, count=10)
            save(root / "validation.json", test)
            with contextlib.redirect_stdout(io.StringIO()):
                validate(args)
            self.assertEqual(json.loads((root / "report.json").read_text())["status"], "PASS_INTERNAL_CONSISTENCY")
            # A changed mounting offset on validation must fail, even if internally constant.
            for sample in test["samples"]:
                c = np.asarray(sample["T_camera_target"])
                sample["T_camera_target"] = (c @ transform(translation=[0.05, 0, 0])).tolist()
            save(root / "validation.json", test)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exc:
                validate(args)
            self.assertEqual(exc.exception.code, 2)


class PoseTests(unittest.TestCase):
    def test_measured_points_check_does_not_fit_out_bias(self):
        _, x, _ = synthetic_dataset(count=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "result.json", {"T_base_camera": x.tolist()})
            header = ["id", "camera_x_m", "camera_y_m", "camera_z_m", "base_x_m", "base_y_m", "base_z_m"]
            with (root / "points.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                for i in range(10):
                    c = np.array([0.01 * i, 0.01, 0.6])
                    b = (x @ np.r_[c, 1])[:3] + np.array([0.01, 0, 0])
                    writer.writerow([i, *c, *b])
            args = argparse.Namespace(result=str(root / "result.json"), csv=str(root / "points.csv"),
                                      output=str(root / "report.json"), threshold_mm=5)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                check_points(args)
            report = json.loads((root / "report.json").read_text())
            self.assertAlmostEqual(report["rms_mm"], 10)
            self.assertFalse(report["fit_performed"])

    def test_pose_file_stale_and_frame_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.json"
            reader = PoseFile(path, "base_link", "gripper_end")
            write_pose(path, np.eye(4), acquired_unix_s=time.time())
            np.testing.assert_equal(reader.read()[0], np.eye(4))
            write_pose(path, np.eye(4), acquired_unix_s=time.time() - 10)
            with self.assertRaisesRegex(ValueError, "stale"):
                reader.read()
            write_pose(path, np.eye(4), ee_frame="tool0", acquired_unix_s=time.time())
            with self.assertRaisesRegex(ValueError, "mismatch"):
                reader.read()

    def test_robot_parameter_reader_uses_only_reads_and_close(self):
        calls = []

        class FakeMotor:
            def robstride_get_param_f32(self, index, timeout_ms):
                calls.append(("read", index))
                return 0.0

        class FakeController:
            def __init__(self, channel):
                calls.append(("open", channel))

            def add_robstride_motor(self, *args):
                calls.append(("register", args))
                return FakeMotor()

            def close_bus(self):
                calls.append(("close_bus",))

            def close(self):
                calls.append(("close",))

        with patch("motorbridge.Controller", FakeController):
            reader = RebotReader(DEFAULT_SDK)
            pose, info = reader.read()
            self.assertEqual(len(info["q_rad"]), 6)
            checked(pose)
            reader.close()
        self.assertEqual(sum(c[0] == "read" for c in calls), 6)
        self.assertEqual(calls[-2:], [("close_bus",), ("close",)])
        self.assertTrue(all(c[1] == 0x7019 for c in calls if c[0] == "read"))


class VisionTests(unittest.TestCase):
    def test_user_marker_bits_match_id_zero_and_sixty_mm_config(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "target_aruco_60mm.json").read_text())
        self.assertEqual(config["marker_m"], 0.06)
        target = Target(config)
        pixels = cv2.aruco.generateImageMarker(target.dictionary, config["id"], 6)
        # Inner white/black cells transcribed from the user's close-up photo.
        expected = [[1, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0]]
        np.testing.assert_equal(pixels[1:5, 1:5] // 255, expected)

    def test_aruco_render_detect_pnp_and_id_guard(self):
        config = {"type": "aruco", "dictionary": "DICT_4X4_50", "id": 0, "marker_m": 0.08}
        target = Target(config)
        k = np.array([[1000., 0, 640], [0, 1000, 360], [0, 0, 1]])
        d = np.zeros(5)
        obj = np.array([[-0.04, 0.04, 0], [0.04, 0.04, 0], [0.04, -0.04, 0], [-0.04, -0.04, 0]])
        r = Rotation.from_euler("xyz", [2.7, 0.25, 0.1]).as_rotvec()
        t = np.array([0., 0., 0.5])
        uv, _ = cv2.projectPoints(obj, r, t, k, d)
        marker = cv2.aruco.generateImageMarker(target.dictionary, 0, 600)
        homography = cv2.getPerspectiveTransform(np.float32([[0, 0], [599, 0], [599, 599], [0, 599]]), uv.reshape(-1, 2).astype(np.float32))
        frame = cv2.warpPerspective(marker, homography, (1280, 720), borderValue=255)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        result = target.detect(frame, k, d)
        mm, deg = errors(np.asarray(result["T_camera_target"]), transform(Rotation.from_rotvec(r).as_matrix(), t))
        self.assertLess(mm, 5)
        self.assertLess(deg, 2)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            target.detect(frame, k, d, ambiguity_gap=1000)
        wrong = Target({**config, "id": 1})
        with self.assertRaises(ValueError):
            wrong.detect(frame, k, d)


class ManualTests(unittest.TestCase):
    def test_manual_requires_terminal_before_hardware(self):
        with patch.object(sys.stdin, "isatty", return_value=False), self.assertRaisesRegex(ValueError, "terminal"):
            ManualGravityReader(DEFAULT_SDK)

    def test_drag_hold_latched_fault_and_supported_release(self):
        calls = []
        worker_sent = threading.Event()
        launch_worker = ManualGravityReader._launch_worker

        class Motor:
            def robstride_get_param_f32(self, index, timeout_ms):
                calls.append(("read", index))
                return 0.

            def ensure_mode(self, *args):
                calls.append(("mode",))

            def enable(self):
                calls.append(("enable",))

            def disable(self):
                calls.append(("disable",))

            def send_mit(self, pos, vel, kp, kd, tau):
                calls.append(("command", pos, vel, kp, kd, tau))
                if threading.current_thread().name == "rebot-gravity":
                    worker_sent.set()

        class Controller:
            def __init__(self, channel):
                pass

            def add_robstride_motor(self, *args):
                calls.append(("register", args[0]))
                return Motor()

            def close_bus(self):
                calls.append(("close_bus",))

            def close(self):
                calls.append(("close",))

        with patch.object(sys.stdin, "isatty", return_value=True), patch("motorbridge.Controller", Controller), \
                patch.object(ManualGravityReader, "_launch_worker"), contextlib.redirect_stdout(io.StringIO()):
            reader = ManualGravityReader(DEFAULT_SDK)
            reader.start()
            self.assertEqual(sum(c[0] == "enable" for c in calls), 6)
            self.assertEqual([c[1] for c in calls if c[0] == "register"], [1, 2, 3, 4, 5, 6])
            reader._started_at -= 3
            reader.set_mode("drag")
            reader._send()
            self.assertTrue(all(c[3] == 0 for c in calls[-6:]))
            reader.set_mode("hold")
            reader._send()
            self.assertTrue(all(c[3] == 8 for c in calls[-6:]))
            pose, info = reader.read()
            checked(pose)
            self.assertEqual(info["manual_mode"], "hold")
            launch_worker(reader)
            self.assertTrue(worker_sent.wait(timeout=1), "Control worker must read feedback and send a command")
            reader._latch_fault("simulated register timeout")
            with self.assertRaisesRegex(RuntimeError, "fault"):
                reader.read()
            with self.assertRaises(RuntimeError):
                reader.set_mode("drag")
            with patch("builtins.input", side_effect=[KeyboardInterrupt(), "release"]):
                reader.close()
            self.assertEqual(sum(c[0] == "disable" for c in calls), 6)
            self.assertEqual(calls[-2:], [("close_bus",), ("close",)])


class CharucoTests(unittest.TestCase):
    def test_charuco_perspective_detection(self):
        config = {"type": "charuco", "dictionary": "DICT_4X4_50", "squares_x": 5,
                  "squares_y": 7, "square_m": 0.025, "marker_m": 0.018}
        target = Target(config)
        k = np.array([[1000., 0, 640], [0, 1000, 360], [0, 0, 1]])
        obj = np.array([[0., 0, 0], [0.125, 0, 0], [0.125, 0.175, 0], [0, 0.175, 0]])
        r, t = np.array([0.4, -0.3, 0.1]), np.array([-0.05, -0.08, 0.55])
        uv, _ = cv2.projectPoints(obj, r, t, k, np.zeros(5))
        board = target.board.generateImage((1000, 1400), marginSize=0, borderBits=1)
        h = cv2.getPerspectiveTransform(np.float32([[0, 0], [999, 0], [999, 1399], [0, 1399]]), uv.reshape(-1, 2).astype(np.float32))
        frame = cv2.cvtColor(cv2.warpPerspective(board, h, (1280, 720), borderValue=255), cv2.COLOR_GRAY2BGR)
        result = target.detect(frame, k, np.zeros(5))
        self.assertGreaterEqual(len(result["image_points"]), 8)
        mm, deg = errors(np.asarray(result["T_camera_target"]), transform(cv2.Rodrigues(r)[0], t))
        self.assertLess(mm, 5)
        self.assertLess(deg, 2)


if __name__ == "__main__":
    unittest.main()

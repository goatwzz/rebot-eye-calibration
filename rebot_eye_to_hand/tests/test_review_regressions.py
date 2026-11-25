from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eyehand.cli import collect
from eyehand.hardware import DEFAULT_SDK
from eyehand.manual import ManualGravityReader


class ReviewRegressions(unittest.TestCase):
    def test_refresh_after_mode_change_and_send_failure_attempts_all_joints(self):
        shared = {"q": 0., "modes": 0, "last_mode": 0., "fail_motor": None, "sent": []}

        class Motor:
            def __init__(self, index):
                self.index = index

            def robstride_get_param_f32(self, index, timeout_ms):
                return shared["q"]

            def ensure_mode(self, *args):
                shared["modes"] += 1
                if shared["modes"] == 6:
                    # Model hand movement while the six mode changes take place.
                    shared["q"] = 0.1
                    shared["last_mode"] = time.monotonic()

            def enable(self):
                pass

            def disable(self):
                pass

            def send_mit(self, pos, vel, kp, kd, tau):
                shared["sent"].append((self.index, pos))
                if shared["fail_motor"] == self.index:
                    raise RuntimeError("injected bus failure")

        class Controller:
            def __init__(self, channel):
                pass

            def add_robstride_motor(self, index, *args):
                return Motor(index)

            def close_bus(self):
                pass

            def close(self):
                pass

        with patch.object(sys.stdin, "isatty", return_value=True), patch("motorbridge.Controller", Controller), \
                patch.object(ManualGravityReader, "_launch_worker"), contextlib.redirect_stdout(io.StringIO()):
            reader = ManualGravityReader(DEFAULT_SDK)
            try:
                reader.start()
                np.testing.assert_allclose(reader._target, [0.1] * 6)
                self.assertGreaterEqual(reader._started_at, shared["last_mode"])
                self.assertTrue(all(abs(pos - 0.1) < 1e-8 for _, pos in shared["sent"]))
                shared["sent"].clear()
                shared["fail_motor"] = 1
                with self.assertRaisesRegex(RuntimeError, "joint1"):
                    reader._send()
                self.assertEqual([index for index, _ in shared["sent"]], list(range(1, 7)))
                shared["fail_motor"] = None
                # Host wall-clock adjustment must not make valid local feedback stale.
                with patch("time.time", return_value=0):
                    reader.read()
                reader._snapshot[1]["read_started_monotonic_s"] -= 1
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    reader.read()
            finally:
                reader.release_supported()

    def test_camera_frame_or_gui_failure_prevents_manual_hardware_initialization(self):
        target = Path(__file__).resolve().parents[1] / "target_aruco_60mm.json"
        for failure in ["frame", "gui"]:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                args = argparse.Namespace(target=str(target), output=str(Path(directory) / "train"), manual=True)

                class Camera:
                    k = np.eye(3)
                    closed = False
                    metadata = {"kind": "fake-camera"}

                    def read(self):
                        if failure == "frame":
                            raise RuntimeError("injected frame failure")
                        return np.zeros((720, 1280, 3), np.uint8)

                    def close(self):
                        self.closed = True

                camera = Camera()
                with patch("eyehand.cli.open_camera", return_value=camera), \
                        patch("eyehand.cli.cv2.namedWindow", side_effect=RuntimeError("injected GUI failure")), \
                        patch("eyehand.cli.cv2.destroyAllWindows"), \
                        patch("eyehand.manual.ManualGravityReader") as hardware:
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        collect(args)
                    hardware.assert_not_called()
                    self.assertTrue(camera.closed)


if __name__ == "__main__":
    unittest.main()

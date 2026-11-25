from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch

from eyehand.cli import open_camera
from eyehand.vision import select_realsense_serial


class CameraSelectionTests(unittest.TestCase):
    devices = [{"name": "RealSense D435", "serial": "332322072170"},
               {"name": "RealSense D435", "serial": "250122070127"}]

    def test_multiple_cameras_never_silently_choose_first(self):
        with self.assertRaisesRegex(ValueError, "explicit --serial"):
            select_realsense_serial(self.devices)

    def test_missing_requested_camera_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, "not connected"):
            select_realsense_serial(self.devices[:1], "250122070127")

    def test_explicit_selection_and_single_device(self):
        self.assertEqual(select_realsense_serial(self.devices, "250122070127"), "250122070127")
        self.assertEqual(select_realsense_serial(self.devices[:1]), "332322072170")

    def test_cli_uses_project_pin_and_allows_explicit_override(self):
        args = argparse.Namespace(camera="realsense", width=1280, height=720, fps=30, serial=None,
                                  device=0, intrinsics=None, command="preview")
        with patch("eyehand.cli.Camera") as camera:
            open_camera(args)
            self.assertEqual(camera.call_args.args[4], "250122070127")
            args.serial = "332322072170"
            open_camera(args)
            self.assertEqual(camera.call_args.args[4], "332322072170")


if __name__ == "__main__":
    unittest.main()

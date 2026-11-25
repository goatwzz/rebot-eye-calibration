"""No hardware access: emulate RealSense start and cleanup failures."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eyehand.vision import Camera


class StartupTests(unittest.TestCase):
    def test_failed_start_is_cleaned_up_without_masking_original_error(self):
        device = Mock()
        device.get_info.side_effect = lambda field: {"name": "D435", "serial": "wrist"}[field]
        pipeline = Mock()
        pipeline.start.side_effect = RuntimeError("set_xu UVC control query timed out")
        pipeline.stop.side_effect = RuntimeError("stop before start")
        rs = SimpleNamespace(camera_info=SimpleNamespace(name="name", serial_number="serial"),
                             context=lambda: SimpleNamespace(query_devices=lambda: [device]),
                             pipeline=lambda: pipeline, config=lambda: Mock(),
                             stream=SimpleNamespace(color=1), format=SimpleNamespace(bgr8=1))
        with patch.dict(sys.modules, {"pyrealsense2": rs}):
            with self.assertRaisesRegex(RuntimeError, "set_xu UVC control query timed out"):
                Camera("realsense", serial="wrist")
        pipeline.start.assert_called_once()
        pipeline.stop.assert_called_once()
        device.hardware_reset.assert_not_called()


if __name__ == '__main__':
    unittest.main()

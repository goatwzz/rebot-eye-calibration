import argparse
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eyehand.cli import read_collection_frame


class TimeoutTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(camera_timeout_retries=2, camera_timeout_ms=1000)

    def test_transient_timeout_recovers_and_preserves_fresh_request(self):
        camera = Mock(kind="realsense")
        camera.read.side_effect = [RuntimeError("Frame didn't arrive within 1000"), "image"]
        with patch("eyehand.cli.cv2.waitKey", return_value=-1), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(read_collection_frame(camera, self.args, fresh=True), "image")
        self.assertEqual(camera.read.call_count, 2)
        self.assertTrue(all(c.kwargs == {"fresh": True, "timeout_ms": 1000} for c in camera.read.call_args_list))
        camera.close.assert_not_called()

    def test_permanent_timeout_is_bounded(self):
        camera = Mock(kind="realsense")
        camera.read.side_effect = RuntimeError("Frame didn't arrive within 1000")
        with patch("eyehand.cli.cv2.waitKey", return_value=-1), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Frame didn't arrive"):
                read_collection_frame(camera, self.args)
        self.assertEqual(camera.read.call_count, 3)
        camera.close.assert_not_called()

    def test_unrelated_camera_errors_are_not_retried(self):
        camera = Mock(kind="realsense")
        camera.read.side_effect = RuntimeError("USB device unavailable")
        with self.assertRaisesRegex(RuntimeError, "USB device unavailable"):
            read_collection_frame(camera, self.args)
        self.assertEqual(camera.read.call_count, 1)

    def test_quit_during_retry_uses_existing_cleanup_path(self):
        camera = Mock(kind="realsense")
        camera.read.side_effect = RuntimeError("Frame didn't arrive within 1000")
        with patch("eyehand.cli.cv2.waitKey", return_value=ord('q')), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                read_collection_frame(camera, self.args)
        self.assertEqual(camera.read.call_count, 1)

    def test_default_callers_keep_original_read_signature(self):
        camera = Mock()
        read_collection_frame(camera, argparse.Namespace())
        camera.read.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()

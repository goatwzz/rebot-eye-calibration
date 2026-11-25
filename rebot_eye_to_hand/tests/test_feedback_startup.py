from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from eyehand.hardware import RebotReader


class FeedbackStartupTests(unittest.TestCase):
    def test_requires_two_consecutive_full_reads_after_transient_failure(self):
        reader = RebotReader.__new__(RebotReader)
        info = {"read_started_monotonic_s": 1., "read_ended_monotonic_s": 1.049}
        complete = (np.eye(4), info)
        with patch.object(RebotReader, "read", side_effect=[complete, RuntimeError("timeout"), complete, complete]) as read, \
                patch("eyehand.hardware.time.sleep"), contextlib.redirect_stdout(io.StringIO()):
            pose, result = reader.wait_for_feedback(1.)
            self.assertEqual(read.call_count, 4)
            self.assertEqual(result, info)
            np.testing.assert_equal(pose, np.eye(4))

    def test_persistent_timeout_does_not_fabricate_positions_or_enable(self):
        reader = RebotReader.__new__(RebotReader)
        with patch.object(RebotReader, "read", side_effect=RuntimeError("joint1 unavailable")), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "motors NOT enabled"):
            reader.wait_for_feedback(0.001)


if __name__ == "__main__":
    unittest.main()

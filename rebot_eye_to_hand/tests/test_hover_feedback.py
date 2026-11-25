"""Isolate the modern SDK from tests importing the legacy collector SDK."""
from pathlib import Path
import subprocess
import sys
import unittest


class ActiveFeedbackTests(unittest.TestCase):
    def test_native_command_reply_startup_refresh_and_expiry(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root/'tests/hover_feedback_checks.py')],
                                cwd=root, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()

"""Keep modern control SDK checks isolated from legacy collection imports."""
from pathlib import Path
import subprocess
import sys
import unittest


class SDKHoverRegressionTests(unittest.TestCase):
    def test_official_cartesian_path_and_tracking_outer(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root/'tests/sdk_hover_checks.py')],
                                cwd=root, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

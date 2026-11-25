"""Isolate modern demo SDK from the legacy calibration collector SDK imports."""
from pathlib import Path
import subprocess
import sys
import unittest


class ExternalCornerPlanningTests(unittest.TestCase):
    def test_real_model_official_planner_and_home_arrival(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root/'tests/corner_planner_checks.py')],
                                cwd=root, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

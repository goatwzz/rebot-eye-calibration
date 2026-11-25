"""Exercise wrist corner poses with the actual URDF and public SDK offline."""
from pathlib import Path
import subprocess
import sys
import unittest


class WristCornerPlanningTests(unittest.TestCase):
    def test_real_model_official_planner_and_home_arrival(self):
        root = Path(__file__).resolve().parents[2]/'rebot_eye_to_hand'
        result = subprocess.run([sys.executable, str(root/'tests/corner_planner_checks.py'), '--wrist'],
                                cwd=root, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

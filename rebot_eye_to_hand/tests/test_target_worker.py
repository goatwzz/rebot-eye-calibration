"""Run native-SDK target production checks apart from legacy collector imports."""
from pathlib import Path
import subprocess
import sys
import unittest

class TargetProducerRegressionTests(unittest.TestCase):
    def test_target_worker_safety_and_real_model_convergence(self):
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,str(root/'tests/target_worker_checks.py')],
                              cwd=root,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

"""Isolate the modern control SDK from legacy collection SDK tests."""
from pathlib import Path
import subprocess
import sys
import unittest

class XYOuterRegressionTests(unittest.TestCase):
    def test_actual_external_model_xy_outer_and_z_independence(self):
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,str(root/'tests/xy_outer_checks.py')],cwd=root,capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__=='__main__':unittest.main()

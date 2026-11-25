from pathlib import Path
import subprocess
import sys
import unittest
from eyehand.click_hover import ROOT


class HomePlanningTests(unittest.TestCase):
    def test_actual_model_zero_hover_home_and_controller_state(self):
        result=subprocess.run([sys.executable,str(ROOT/'tests/home_planner_checks.py')],
                              cwd=ROOT,capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()

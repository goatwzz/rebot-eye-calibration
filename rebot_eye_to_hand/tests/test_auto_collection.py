"""Keep modern native SDK imports isolated from legacy collection tests."""
from pathlib import Path
import subprocess
import sys
import unittest


class AutoCollectionTests(unittest.TestCase):
    def test_automatic_acquisition_contract(self):
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,str(root/'tests/auto_collection_checks.py')],
            cwd=root,capture_output=True,text=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

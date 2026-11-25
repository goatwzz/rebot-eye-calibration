"""Isolated public SDK feedback regressions, never opens hardware."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
suite=unittest.defaultTestLoader.loadTestsFromNames(['feedback_lifecycle_checks','native_adapter_checks'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())

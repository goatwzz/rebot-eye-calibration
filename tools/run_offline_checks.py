#!/usr/bin/env python3
"""Run public offline tests in separate SDK processes. No camera/CAN/motor access."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
# These regressions intentionally depend on private raw datasets/session logs.
PRIVATE_MODULES = {
    'rebot_eye_to_hand': {'test_home_planning','test_sdk_hover','test_xy_outer'},
    'rebot_eye_in_hand': {'test_cartesian_outer','test_sdk_click_approach',
                         'test_transit_profile','test_paired_calibration','test_low_validation'},
}
PRIVATE_CASES = {
    'test_click_hover.ClickGeometryTests.test_mesh_distal_tip_report_and_config_agree',
    'test_click_hover.ClickGeometryTests.test_real_model_planning_in_isolated_sdk_process',
}


def cases(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from cases(test)
        else:
            yield test


def child(project):
    root = ROOT/project
    sys.path[:0] = [str(root), str(root/'tests')]
    if project == 'rebot_eye_to_hand':
        # Reader-only tests mock Controller; import verified dependency first.
        from eyehand.native_runtime import activate
        activate()
    suite = unittest.TestSuite()
    for path in sorted((root/'tests').glob('test_*.py')):
        if path.stem in PRIVATE_MODULES[project]:
            print('[EXCLUDED private dataset/log regression]', path.name, flush=True)
            continue
        loaded = unittest.defaultTestLoader.loadTestsFromName(path.stem)
        for test in cases(loaded):
            if test.id() in PRIVATE_CASES:
                print('[EXCLUDED private dataset/log regression]', test.id(), flush=True)
            else:
                suite.addTest(test)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', choices=list(PRIVATE_MODULES))
    args = parser.parse_args()
    if args.child:
        return child(args.child)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for project in PRIVATE_MODULES:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', project],
                                   cwd=ROOT/project, env=env)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

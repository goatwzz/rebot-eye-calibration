#!/usr/bin/env python3
"""Offline release integrity and paired calibration/planner smoke checks."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def check_model(mount):
    if mount == 'wrist':
        sys.path.insert(0, str(ROOT/'rebot_eye_in_hand'))
        from wrist.click_hover import WristPlanner, load_config
        p = WristPlanner(load_config())
        from eyehand.corner_demo import load_settings
        s = load_settings(ROOT/'rebot_eye_in_hand/config/aruco_corners.json', 'wrist')
        ready = p.plan_ready(np.zeros(6))
        np.testing.assert_allclose(p.tcp(ready.goal), [.3, 0., .3], atol=.0002)
        home = p.plan_home(ready.goal)
        expected = np.radians([0., 2.452577744182385, -.10893790754488918,
                               -1.7512200328350365, -.5317715483456086, 0.])
        assert s['clearance_m'] == .003
    else:
        sys.path.insert(0, str(ROOT/'rebot_eye_to_hand'))
        from eyehand.click_hover import HoverPlanner, load_config
        from eyehand.corner_demo import load_settings
        p = HoverPlanner(load_config(ROOT/'rebot_eye_to_hand/aruco_corners_control_config.json'))
        s = load_settings(ROOT/'rebot_eye_to_hand/aruco_corners_config.json', 'external')
        home = p.plan_home(np.array([0., .6, .5, 0., .2, 0.]))
        expected = np.radians([0., 1.7753664760636807, 1.3056954539696328,
                               -2.7445581492229962, -.8979697820635353, 0.])
        assert s['clearance_m'] == .0005
    np.testing.assert_allclose(p.offsets, expected, atol=1e-14)
    np.testing.assert_array_equal(home.goal, np.zeros(6))
    assert s['marker_m'] == .06 and s['marker_id'] == 0
    assert np.isfinite(p.gravity_torque(np.zeros(6))).all()
    print(f'[OK] {mount}: paired model, zero/home convention, 60mm ID0, gravity and clearance')


def main():
    if len(sys.argv) > 1:
        check_model(sys.argv[1]); return
    manifest = json.loads((ROOT/'vendor/sdk-manifest.json').read_text())
    for name, spec in manifest['sdks'].items():
        p = ROOT/'vendor/sdks'/name/'urdf/RS/urdf/ReBot_Arm_RS.urdf'
        assert hashlib.sha256(p.read_bytes()).hexdigest() == spec['urdf_sha256']
    for path in ROOT.rglob('*.json'):
        if 'vendor/sdks' in path.as_posix() or '.git' in path.parts:
            continue
        json.loads(path.read_text())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for name in ('wrist', 'external'):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), name],
                       cwd=ROOT, env=env, check=True)
    print('[OK] No camera, CAN, motor connect or enable calls were made.')


if __name__ == '__main__':
    main()

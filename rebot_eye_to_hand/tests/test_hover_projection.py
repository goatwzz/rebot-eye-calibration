"""Surface-point geometry and projection checks; no camera or CAN access."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
import numpy as np
from eyehand.click_hover import ROOT, hover_target, load_config
from scripts.click_hover_eth import project_base_pixel


class ClickProjectionTests(unittest.TestCase):
    def bundle(self):
        result=json.loads((ROOT.parent/'examples/legacy_click/result.json').read_text())
        return SimpleNamespace(result=result,calibration=result['provenance'])

    def test_observed_surface_vs_100mm_hover_projection(self):
        p=self.bundle()
        surface=np.array([.4778837163392661,.10567914646371761,.02571822317584893])
        np.testing.assert_allclose(project_base_pixel(p,surface),[885.,574.],atol=1e-8)
        np.testing.assert_allclose(project_base_pixel(p,surface+[0,0,.1]),
                                   [938.3230238811866,554.2768357414922],atol=1e-8)
        camera_back=np.array(p.result['T_base_camera'])@np.array([0,0,-1,1.])
        self.assertIsNone(project_base_pixel(p,camera_back[:3]))

    def test_hover_target_preserves_base_xy_and_keeps_positive_clearance(self):
        p=self.bundle()
        point=[.13532953345582105,.11493379716354234,.5400000256486237]
        height=load_config()['hover_m']
        self.assertGreater(height,0.)
        surface,target=hover_target(p.result['T_base_camera'],point,height)
        np.testing.assert_array_equal(target[:2],surface[:2])
        self.assertAlmostEqual(target[2]-surface[2],height)
        np.testing.assert_allclose(project_base_pixel(p,surface),[885.,574.],atol=1e-8)
        self.assertGreater(np.linalg.norm(np.array(project_base_pixel(p,target))-[885,574]),1.)
        for invalid in (0.,-.01,float('nan'),float('inf')):
            with self.assertRaises(ValueError):hover_target(p.result['T_base_camera'],point,invalid)

    def test_real_model_replays_recorded_hover_clicks_in_isolated_sdk(self):
        result=subprocess.run([sys.executable,str(ROOT/'tests/hover_target_checks.py')],
                              cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()

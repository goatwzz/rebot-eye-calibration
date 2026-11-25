"""Frozen paired-result evaluation and nominal feedback capture; no hardware."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wrist.low_validation import load_settings,ValidationPlanner,report,geometry
from eyehand.auto_collection import collection_plan
from eyehand.kinematics import CorrectedKinematics


class LowValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings,cls.cfg,cls.rows=load_settings(ROOT/'config/low_pose_validation.json')
        cls.p=ValidationPlanner(cls.cfg)

    def test_low_profile_is_validation_only_with_60mm_marker(self):
        self.assertEqual(len(self.rows),62)
        self.assertEqual(self.settings['target']['marker_m'],.06)
        self.assertEqual(self.settings['split'],'validation')
        self.assertLess(min(r['xyz_rpy'][2] for r in self.rows),.1)
        self.assertAlmostEqual(max(r['xyz_rpy'][4] for r in self.rows),np.pi/2)
        self.assertEqual(self.p.gravity['wrist_camera_mass_kg'],.110)
        with self.assertRaisesRegex(ValueError,'validation only'):
            load_settings(ROOT/'config/low_pose_validation.json','train')

    def test_official_path_uses_paired_angles_but_saved_fk_stays_nominal(self):
        p=self.p
        plan=collection_plan(p,np.zeros(6),self.rows[0],self.settings)
        np.testing.assert_allclose(p.planning_fk(plan.goal)[:3,3],self.rows[0]['xyz_rpy'][:3],atol=1e-6)
        nominal=CorrectedKinematics(p.robot_metadata)
        np.testing.assert_allclose(p.nominal_fk(plan.goal),nominal.fk(plan.goal),atol=1e-12)
        self.assertGreater(np.linalg.norm(p.nominal_fk(plan.goal)[:3,3]-p.planning_fk(plan.goal)[:3,3]),.005)
        self.assertFalse(plan.tracking_outer_loop)
        self.assertGreater(len(plan.points),100)
        np.testing.assert_allclose(plan.points[0],np.zeros(6),atol=1e-12)

    def fixture(self,folder):
        p=self.p;p.snapshot_validation_result(folder)
        old=json.loads((ROOT/'runs/eih_auto_train_01/dataset.json').read_text())
        samples=[];obj=np.array([[-.03,.03,0],[.03,.03,0],[.03,-.03,0],[-.03,-.03,0]])
        k=np.array(old['K']);target=geometry.transform(translation=[.4,0,0])
        # Analytic fixture from saved raw angles, NOT new real measurements.
        for i,rawsample in enumerate(old['samples'][:20]):
            s=copy.deepcopy(rawsample);q=np.array(s['robot_reads'][-1]['q_rad'])
            g=p.planning_fk(q);c=geometry.inverse(g@geometry.unpack(geometry.pack(np.array(json.loads((ROOT/'config/calibration/active_calibration.json').read_text())['T_gripper_camera']))))@target
            uv,_=cv2.projectPoints(obj,cv2.Rodrigues(c[:3,:3])[0],c[:3,3],k,np.zeros(5))
            s.update(id='fixture-'+str(i),object_points=obj.tolist(),image_points=uv.reshape(-1,2).tolist(),T_camera_target=c.tolist(),requested_pose=dict(group='fixture'))
            samples.append(s)
        data=dict(old,dataset_id='independent-fixture',target=self.settings['target'],samples=samples,
                  acquisition=dict(split='validation',calibration_mode='eye_in_hand'))
        (folder/'dataset.json').write_text(json.dumps(data))
        (folder/'session.json').write_text(json.dumps(dict(control_config=self.cfg)))

    def test_report_uses_frozen_candidate_without_solver_or_refit(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);self.fixture(folder)
            with patch('wrist.low_validation.geometry.solve',side_effect=AssertionError('Must not fit')) as solver:
                result=report(folder)
            solver.assert_not_called()
            self.assertLess(result['axis_errors_mm']['z']['max'],1e-8)
            self.assertEqual(result['status'],'PASS_CONSISTENCY_ONLY')
            self.assertFalse(result['refit_performed']);self.assertFalse(result['physical_accuracy_verified'])
            self.assertEqual(result['calibration']['sha256'],self.cfg['validation_result_sha256'])
            with self.assertRaises(FileExistsError):report(folder)

    def test_modified_result_snapshot_cannot_be_reported_as_original(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);self.fixture(folder)
            f=folder/'hand_eye_under_test.npz';f.write_bytes(f.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'Frozen candidate changed'):report(folder)

    def test_validation_snapshot_is_captured_before_source_can_change(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);source=folder/'source.npz';source.write_bytes(self.p.result_bytes)
            cfg=dict(self.cfg,validation_result_source=str(source));p=ValidationPlanner(cfg)
            source.write_bytes(b'changed after initialization')
            output=folder/'capture';output.mkdir();p.snapshot_validation_result(output)
            self.assertEqual(hashlib.sha256((output/'hand_eye_under_test.npz').read_bytes()).hexdigest(),cfg['validation_result_sha256'])


if __name__=='__main__':unittest.main()

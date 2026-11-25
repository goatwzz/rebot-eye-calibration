"""Offline direction/fit/provenance and SDK planning checks; no devices opened."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wrist.auto_calibration import fit_candidate,geometry
from eyehand.auto_collection import load_settings,CollectionPlanner,collection_plan
from eyehand.auto_collection_runtime import AutoSequence
from wrist.validation import load_result


def synthetic_data():
    rng=np.random.default_rng(210)
    x=geometry.transform(Rotation.from_euler('xyz',[.2,-.4,.1]).as_matrix(),[-.1,.03,.09])
    y=geometry.transform(translation=[.35,0,.09])
    k=np.array([[900.,0,640],[0,900,360],[0,0,1.]])
    obj=np.array([[-.05,.05,0],[.05,.05,0],[.05,-.05,0],[-.05,-.05,0]])
    samples=[]
    for i in range(30):
        c=geometry.transform(Rotation.from_euler('xyz',rng.uniform(-.5,.5,3)).as_matrix(),
                             [*rng.uniform(-.08,.08,2),rng.uniform(.4,.65)])
        g=y@geometry.inverse(c)@geometry.inverse(x)
        uv,_=cv2.projectPoints(obj,cv2.Rodrigues(c[:3,:3])[0],c[:3,3],k,np.zeros(5))
        samples.append(dict(id=str(i),image=str(i)+'.png',T_base_ee=g.tolist(),T_camera_target=c.tolist(),
            object_points=obj.tolist(),image_points=uv.reshape(-1,2).tolist(),K=k.tolist(),D=[0.]*5,
            robot_reads=[dict(q_rad=[0.]*6)],requested_pose=dict(group='unit_fixture')))
    data=dict(dataset_id='train-fixture',synthetic=False,acquisition=dict(calibration_mode='eye_in_hand',split='train'),
        target=dict(type='aruco',dictionary='DICT_4X4_50',id=0,marker_m=.1),
        camera=dict(serial='332322072170'),robot=dict(urdf_sha256='test',base_frame='base_link',ee_frame='gripper_end'),
        image_size=[1280,720],K=k.tolist(),D=[0.]*5,samples=samples)
    return data,x


class AutoCalibrationTests(unittest.TestCase):
    def test_opencv_direction_recovers_known_wrist_extrinsic_and_compatible_npz(self):
        data,x=synthetic_data()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'dataset.json';p.write_text(json.dumps(data))
            report=fit_candidate(p,Path(d)/'fit')
            actual,_=load_result(Path(d)/'fit/hand_eye_candidate.npz',geometry)
            np.testing.assert_allclose(actual,x,atol=1e-7)
            self.assertLess(report['training']['summary']['translation_mm']['max'],1e-4)
            self.assertLess(report['training']['summary']['predicted_reprojection_rms_px']['max'],1e-4)
            self.assertFalse(report['active_calibration_changed'])
            with self.assertRaises(FileExistsError):fit_candidate(p,Path(d)/'fit')

    def test_independent_validation_cannot_reuse_training_samples(self):
        data,_=synthetic_data();v=copy.deepcopy(data)
        v['dataset_id']='validation-fixture';v['acquisition']['split']='validation'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'dataset.json';p.write_text(json.dumps(data))
            vp=Path(d)/'validation.json';vp.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'independently'):fit_candidate(p,Path(d)/'fit',vp)
            self.assertFalse((Path(d)/'fit').exists())
            for s in v['samples']:s['id']='independent-'+s['id']
            vp.write_text(json.dumps(v))
            report=fit_candidate(p,Path(d)/'fit',vp)
            self.assertEqual(report['validation']['status'],'PASS_CONSISTENCY_ONLY')

    def test_wrist_config_mass_raw_fk_and_official_path(self):
        s,c,rows=load_settings(ROOT/'config/auto_collection.json',mount='wrist')
        sv,_,validation=load_settings(ROOT/'config/auto_collection.json','validation',mount='wrist')
        self.assertEqual(len(rows),150);self.assertEqual(len(validation),92)
        self.assertEqual(s['target']['marker_m'],.06);self.assertEqual(c['camera_fps'],30)
        self.assertTrue(c['manual_disable_only']);self.assertEqual(c['home_completion_mode'],'trajectory')
        self.assertNotIn('result',c)
        self.assertTrue(set(tuple(r['xyz_rpy']) for r in rows).isdisjoint(tuple(r['xyz_rpy']) for r in validation))
        p=CollectionPlanner(c)
        self.assertEqual(p.gravity['wrist_camera_mass_kg'],.110)
        self.assertEqual(p.gravity['payload_mass_kg'],0.)
        delta=sum(i.mass for i in p.gravity_model.inertias)-sum(i.mass for i in p.model.inertias)
        self.assertAlmostEqual(delta,.110-.036,places=10)
        self.assertFalse(np.any(p.offsets));self.assertFalse(hasattr(p,'T_gripper_camera'))
        plan=collection_plan(p,np.zeros(6),rows[0],s)
        self.assertGreater(len(plan.points),100)
        self.assertFalse(plan.tracking_outer_loop)
        self.assertEqual(plan.kind,'calibration')

    def test_home_trajectory_completion_does_not_add_accuracy_gate(self):
        ctrl=Mock();ctrl.motion_epoch=1
        ctrl.take_completion.return_value=dict(status='HOME_COMMAND_FINISHED_HOLD')
        seq=AutoSequence(Mock(),ctrl,[],{},Mock())
        seq.worker=Mock(active=False);seq.worker.poll.return_value=None
        seq.phase='HOMING';seq.pump()
        self.assertEqual(seq.phase,'DONE');self.assertTrue(seq.home_ok)
        self.assertFalse(seq.home_verified)


if __name__=='__main__':unittest.main()

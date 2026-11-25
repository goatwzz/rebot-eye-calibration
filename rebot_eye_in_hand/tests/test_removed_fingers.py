"""Offline mass/kinematics and held-out capture checks; no devices opened."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as et

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wrist.auto_calibration import fit_candidate
from eyehand.auto_collection import load_settings,CollectionPlanner


class RemovedFingersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings,cls.cfg,cls.rows=load_settings(ROOT/'config/auto_collection.json',mount='wrist')
        cls.planner=CollectionPlanner(cls.cfg)
        cls.original=CollectionPlanner(dict(cls.cfg,collection_gravity_link_mass_reductions_kg={}))

    def test_measured_36g_removed_not_whole_150g_gripper_links(self):
        p=self.planner;o=self.original
        self.assertAlmostEqual(sum(i.mass for i in o.gravity_model.inertias)-sum(i.mass for i in p.gravity_model.inertias),.036,places=12)
        self.assertEqual(p.gravity['wrist_camera_mass_kg'],.110)
        self.assertEqual(p.gravity['payload_mass_kg'],0.)
        self.assertAlmostEqual(p.gravity_exclusion['removed_mass_kg'],.036,places=12)
        self.assertEqual(p.gravity_exclusion['link_mass_reductions_kg'],dict(gripper_left=.018,gripper_right=.018))
        self.assertIs(p.sdk_law.model,p.gravity_model)
        self.assertIsNot(p.model,p.gravity_model)

    def test_gravity_difference_matches_two_removed_point_masses(self):
        p=self.planner;o=self.original;pin=p.pin
        # Independent first-moment check: gravity of two 18g point masses at
        # the URDF COMs, not a second implementation of inertia scaling.
        mass_model=o.model.copy()
        for i in range(mass_model.njoints):mass_model.inertias[i]=pin.Inertia.Zero()
        tree=et.fromstring(Path(self.cfg['collection_urdf']).read_bytes())
        for name in ['gripper_left','gripper_right']:
            com=np.fromstring(tree.find(f"link[@name='{name}']/inertial/origin").get('xyz'),sep=' ')
            frame=mass_model.frames[mass_model.getFrameId(name)]
            mass_model.appendBodyToJoint(frame.parentJoint,pin.Inertia(.018,com,np.zeros((3,3))),frame.placement)
        for q in [np.array([.1,1.4,1.5,-.8,.2,-.3]),np.array([-.2,1.7,1.1,-.6,-.3,.2])]:
            expected=pin.computeGeneralizedGravity(mass_model,mass_model.createData(),q)*p.gravity['tau_scale']
            np.testing.assert_allclose(o.gravity_torque(q)-p.gravity_torque(q),expected,rtol=0,atol=1e-11)
            np.testing.assert_allclose(o.nominal_fk(q),p.nominal_fk(q),rtol=0,atol=1e-12)
            self.assertGreater(np.max(np.abs(expected)),.01)

    def test_reconfigure_is_idempotent_and_original_urdf_unchanged(self):
        path=Path(self.cfg['collection_urdf']);before=hashlib.sha256(path.read_bytes()).hexdigest()
        p=CollectionPlanner(self.cfg);q=np.array([.1,1.4,1.5,-.8,.2,-.3]);tau=p.gravity_torque(q)
        p._configure_collection_gravity(path)
        np.testing.assert_allclose(p.gravity_torque(q),tau,rtol=0,atol=1e-12)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)

    def test_low_train_and_validation_are_separate_and_retain_raw_angles(self):
        s,c,v=load_settings(ROOT/'config/auto_collection.json','validation',mount='wrist')
        self.assertTrue({tuple(r['xyz_rpy']) for r in self.rows}.isdisjoint(tuple(r['xyz_rpy']) for r in v))
        for rows in [self.rows,v]:
            low=[r for r in rows if r['xyz_rpy'][2]<.14]
            self.assertGreaterEqual(len(low),12)
            self.assertTrue(any(abs(r['xyz_rpy'][4]-np.pi/2)<1e-12 for r in low))
        self.assertEqual(s['target']['marker_m'],.06)
        self.assertFalse(np.any(self.planner.offsets));self.assertNotIn('result',c)
        self.assertEqual(self.planner.robot_metadata['kinematic_model'],'nominal_raw_angles_no_offsets')

    def test_mass_configuration_rejects_negative_or_excess_removal(self):
        for reductions in [dict(gripper_left=-.018),dict(link6=.036),dict(gripper_left=.08),dict(gripper_left=True)]:
            with self.assertRaises(ValueError):CollectionPlanner(dict(self.cfg,collection_gravity_link_mass_reductions_kg=reductions))

    def test_unrelated_external_collection_keeps_prior_config_schema(self):
        _,cfg,_=load_settings(ROOT.parent/'rebot_eye_to_hand/auto_collection_config.json')
        self.assertNotIn('collection_gravity_link_mass_reductions_kg',cfg)

    def test_validation_cannot_be_fitted_as_training(self):
        from test_auto_calibration import synthetic_data
        data,_=synthetic_data();data['acquisition']['split']='validation'
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);path=folder/'dataset.json';path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'train split'):fit_candidate(path,folder/'fit')
            self.assertFalse((folder/'fit').exists())


if __name__=='__main__':unittest.main()

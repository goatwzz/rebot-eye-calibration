"""Actual corrected model, saved images and official SDK paths, offline only."""
from pathlib import Path
import json
import sys
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wrist.click_hover import WristPlanner,load_config
from wrist.validation import load_result,apply_result_correction,evaluate
from eyehand.kinematics import CorrectedKinematics
from eyehand.corner_demo import load_settings
from eyehand.aruco_corners import CornerDetector,corner_goal,plan_corner
from eyehand.hover_control import HoverController
from eyehand import geometry


class PairedCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=WristPlanner(load_config())
        cls.dataset=json.loads((ROOT/'runs/eih_auto_train_01/dataset.json').read_text())
        cls.report=json.loads((ROOT/'runs/eih_auto_train_01/refit_review_20261125/corrected_report.json').read_text())
        cls.ready=cls.p.plan_ready(np.zeros(6))

    def test_camera_fk_matches_fit_model_for_all_74_samples_once(self):
        p=self.p
        fk=CorrectedKinematics(self.dataset['robot'],self.report['kinematic_correction'])
        self.assertGreater(np.linalg.norm(p.offsets),.01)
        for s in self.dataset['samples']:
            raw=np.array(s['robot_reads'][-1]['q_rad'])
            expected=fk.fk(raw)
            np.testing.assert_allclose(p.camera_pose(raw),expected@p.T_gripper_camera,atol=1e-12)
            np.testing.assert_allclose(p.tcp(raw),expected[:3,3],atol=1e-12)

    def test_ready_and_home_keep_raw_motor_coordinates(self):
        p=self.p;r=self.ready;h=p.plan_home(r.goal)
        np.testing.assert_allclose(r.points[0],np.zeros(6),atol=1e-12)
        np.testing.assert_allclose(p.tcp(r.goal),[.3,0,.3],atol=.0002)
        data=p.model.createData();p.pin.framesForwardKinematics(p.model,data,r.goal+p.offsets)
        np.testing.assert_allclose(data.oMf[p.tcp_id].rotation,p.pin.rpy.rpyToMatrix(0.,.7,0.),atol=.001)
        np.testing.assert_array_equal(h.goal,np.zeros(6))
        np.testing.assert_allclose(h.target,p.tcp(np.zeros(6)),atol=1e-12)
        for plan in [r,h]:
            for raw in plan.points:p.validate_raw(raw)

    def test_gravity_uses_corrected_angles_and_110g_exactly_once(self):
        p=self.p
        self.assertEqual(p.gravity['wrist_camera_mass_kg'],.110)
        self.assertEqual(p.gravity['payload_mass_kg'],0.)
        self.assertAlmostEqual(sum(v.mass for v in p.gravity_model.inertias)-sum(v.mass for v in p.model.inertias),.110)
        data=p.gravity_model.createData()
        for raw in [np.zeros(6),self.ready.goal,self.ready.sample(1.5)[0]]:
            expected=p.pin.computeGeneralizedGravity(p.gravity_model,data,raw+p.offsets)*p.gravity['tau_scale']
            np.testing.assert_allclose(p.gravity_torque(raw),expected,atol=1e-10)

    def test_validation_rebuilds_nominal_fk_without_double_correction(self):
        p=self.p;_,info=load_result(p.cfg['result'],geometry,allow_joint_offsets=True)
        corrected=apply_result_correction(self.dataset,info)
        again=apply_result_correction(corrected,info)
        actual=evaluate(corrected,p.T_gripper_camera,np.eye(4),geometry)
        for field,expected in self.report['corrected_training_consistency'].items():
            for stat,value in expected.items():self.assertAlmostEqual(actual['summary'][field][stat],value,places=8)
        self.assertEqual(corrected,again)
        self.assertNotIn('kinematic_correction',self.dataset['robot'])

    def test_corner_demo_keeps_user_60mm_target_and_six_mm_clearance(self):
        p=self.p;s=load_settings(ROOT/'config/aruco_corners.json','wrist')
        self.assertEqual(s['marker_m'],.06);self.assertEqual(s['marker_id'],0)
        self.assertEqual(s['clearance_m'],.006)
        self.assertEqual(p.cfg['camera_fps'],30)
        # Generated 60mm board at a raised plane; RGB-only PnP plus corrected camera pose.
        detector=CornerDetector(s,p.calibration)
        marker=cv2.aruco.generateImageMarker(detector.target.dictionary,0,180)
        obj=np.array([[-.03,.03,0],[.03,.03,0],[.03,-.03,0],[-.03,-.03,0]],float)
        rot=p.pin.rpy.rpyToMatrix(.35,-.25,.15)@np.diag([1.,-1.,-1.])
        uv,_=cv2.projectPoints(obj,cv2.Rodrigues(rot)[0],np.array([.01,-.02,.4]),
                               detector.k,detector.d)
        warp=cv2.getPerspectiveTransform(np.array([[0,0],[179,0],[179,179],[0,179]],np.float32),
                                        uv.reshape(4,2).astype(np.float32))
        mono=cv2.warpPerspective(marker,warp,(1280,720),borderValue=255)
        image=cv2.cvtColor(mono,cv2.COLOR_GRAY2BGR)
        obs=detector.detect(NS(color=image,acquired=time.monotonic()),p.camera_pose(self.ready.goal),self.ready.goal)
        self.assertAlmostEqual(np.linalg.norm(obs.corners_base_m[1]-obs.corners_base_m[0]),.06,places=9)
        for n in range(1,5):
            corner,goal=corner_goal(obs,n,s)
            np.testing.assert_allclose(goal-corner,[0,0,.006],atol=1e-12)

    def test_all_four_raised_corners_plan_with_corrected_ik(self):
        p=self.p;s=load_settings(ROOT/'config/aruco_corners.json','wrist')
        for xyz in [[.38,-.03,.0905],[.44,-.03,.0905],[.44,.03,.0905],[.38,.03,.0905]]:
            plan=plan_corner(p,self.ready.goal,np.array(xyz),s)
            np.testing.assert_allclose(plan.points[0],self.ready.goal,atol=1e-12)
            np.testing.assert_allclose(p.tcp(plan.goal),xyz,atol=.0002)
            robot=NS(arm=Mock(),gripper=Mock(),estop=Mock(),_motor_map={})
            c=HoverController(p,robot);c.enabled=True;c.target=self.ready.goal.copy();c.gripper=np.array([.25])
            c.feedback=Mock();c.feedback.snapshot.return_value=NS(positions=np.r_[self.ready.goal,.25])
            c.plan=plan;c.started=0.;c.stable_since=None;c.ui_seen=0.
            with patch('eyehand.hover_control.time.perf_counter',return_value=0.):c._tick(robot,.002)
            self.assertIsNone(c.fault)
            np.testing.assert_allclose(robot.arm.send_mit.call_args.args[0],self.ready.goal,atol=1e-12)


if __name__=='__main__':unittest.main()

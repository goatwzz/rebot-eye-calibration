"""Offline reference-SDK protocol comparison and smooth transit sampling."""
from pathlib import Path
import sys
import json
import subprocess
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wrist.click_hover import WristPlanner,load_config
from legacy_model_fixture import load_legacy_config as load_config
from eyehand.hover_control import HoverController


class TransitProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=WristPlanner(load_config())
        cls.ready=cls.p.plan_ready(np.zeros(6))

    def controller(self,plan,raw):
        robot=SimpleNamespace(arm=Mock(),gripper=Mock(),estop=Mock(),_motor_map={})
        c=HoverController(self.p,robot);c.enabled=True;c.target=raw.copy();c.gripper=np.array([.25])
        c.feedback=Mock();c.feedback.snapshot.return_value=SimpleNamespace(positions=np.r_[raw,.25])
        c.plan=plan;c.started=0.;c.stable_since=None
        return c,robot

    def test_transit_mit_velocity_and_gains_match_reference_sdk_controller(self):
        from reBotArm_control_py.controllers import RebotArmEndPose
        for plan in (self.ready,self.p.plan_home(self.ready.goal)):
            with self.subTest(kind=plan.kind):
                raw,velocity=plan.sample(plan.duration/2)
                np.testing.assert_array_equal(velocity,np.zeros(6))
                c,robot=self.controller(plan,raw);c.ui_seen=plan.duration/2
                with patch('eyehand.hover_control.time.perf_counter',return_value=plan.duration/2):
                    c._tick(robot,.002)
                self.assertIsNone(c.fault,c.fault)
                ours=robot.arm.send_mit.call_args
                arm=SimpleNamespace(num_joints=6,send_mit=Mock(),
                                    _mit_kp=np.array(self.p.cfg['kp']),_mit_kd=np.array(self.p.cfg['kd']))
                reference_robot=SimpleNamespace(arm_control_mode='mit',has_gripper=False,groups={'arm':arm})
                reference=RebotArmEndPose(reference_robot,use_gravity_ff=False)
                reference._q_target=raw.copy()
                reference._loop_cb(reference_robot,.002)
                expected=arm.send_mit.call_args
                for key in ('vel','kp','kd'):
                    np.testing.assert_array_equal(ours.kwargs[key],expected.kwargs[key])
                np.testing.assert_array_equal(ours.args[0],expected.args[0])
                np.testing.assert_array_equal(robot.gripper.send_mit.call_args.args[0],[.25])

    def test_feedback_confirmed_completion_records_observed_500hz(self):
        self.assertEqual(self.p.cfg['control_rate_hz'],500)
        for plan in (self.ready,self.p.plan_home(self.ready.goal)):
            with self.subTest(kind=plan.kind):
                c,robot=self.controller(plan,plan.goal)
                for i in range(300):
                    now=plan.duration+.01+.002*i;c.ui_seen=now
                    with patch('eyehand.hover_control.time.perf_counter',return_value=now):c._tick(robot,.002)
                    if not c.busy:break
                self.assertIsNone(c.fault,c.fault)
                done=c.take_completion()
                self.assertEqual(done['status'],'HOME_REACHED_HOLD' if plan.kind=='home' else 'SETTLED_MODEL_ONLY')
                self.assertFalse(done['mit_velocity_feedforward'])
                self.assertAlmostEqual(done['control_timing']['observed_callback_rate_hz'],500,places=6)
                self.assertAlmostEqual(done['control_timing']['max_callback_interval_ms'],2,places=6)

    def test_recorded_observation_static_bias_uses_its_own_arrival_tolerance(self):
        log=ROOT/'runs/click_hover/20261123_143644_adacdf49.jsonl'
        rows=[json.loads(line) for line in log.read_text().splitlines()]
        records=[row for row in rows if row.get('phase')=='READY_MOVING'
                 and row.get('status')=='SETTLE_TIMEOUT_HOLD']
        self.assertEqual(len(records),2)
        p,plan=self.p,self.ready
        for record in records:
            measured=np.array(record['raw_q_rad'])
            with self.subTest(error_mm=record['fk_error_mm']):
                # The old 3mm gate reproduces the real timeout without motion.
                with patch.dict(p.cfg,{'ready_position_tolerance_m':.003}):
                    c,robot=self.controller(plan,measured)
                    now=plan.duration+p.cfg['settle_timeout_s']+.01;c.ui_seen=now
                    with patch('eyehand.hover_control.time.perf_counter',return_value=now):c._tick(robot,.002)
                    self.assertEqual(c.take_completion()['status'],'SETTLE_TIMEOUT_HOLD')
                c,robot=self.controller(plan,measured)
                for i in range(301):
                    now=plan.duration+.01+.002*i;c.ui_seen=now
                    with patch('eyehand.hover_control.time.perf_counter',return_value=now):c._tick(robot,.002)
                    self.assertIsNone(c.fault,c.fault)
                    if not c.busy:break
                result=c.take_completion()
                self.assertIsNotNone(result)
                self.assertEqual(result['status'],'SETTLED_MODEL_ONLY')
                self.assertGreater(result['fk_error_mm'],3.)
                self.assertLessEqual(result['fk_error_mm'],5.)
                self.assertEqual(result['ready_position_tolerance_mm'],5.)
                self.assertLessEqual(result['ready_joint_error_deg'],result['ready_joint_tolerance_deg'])
                self.assertIsNone(c.cartesian_settler)
                np.testing.assert_array_equal(c.target,plan.goal)
                self.assertEqual(p.cfg['cartesian_outer_loop']['tolerance_m'],.001)

    def test_observation_requires_both_position_and_joint_arrival(self):
        p,plan=self.p,self.ready
        for mode,delta in [('position',[0,.009,-.009,0,0,0]),
                           ('joint',[0,0,0,0,0,.02])]:
            with self.subTest(mode=mode):
                measured=plan.goal+delta
                position_error=np.linalg.norm(p.tcp(measured)-plan.target)
                joint_error=float(np.max(np.abs(measured-plan.goal)))
                if mode=='position':
                    self.assertLess(joint_error,p.cfg['ready_joint_tolerance_rad'])
                    self.assertGreater(position_error,p.cfg['ready_position_tolerance_m'])
                else:
                    self.assertLess(position_error,p.cfg['ready_position_tolerance_m'])
                    self.assertGreater(joint_error,p.cfg['ready_joint_tolerance_rad'])
                c,robot=self.controller(plan,measured)
                now=plan.duration+p.cfg['settle_timeout_s']+.01;c.ui_seen=now
                with patch('eyehand.hover_control.time.perf_counter',return_value=now):c._tick(robot,.002)
                self.assertIsNone(c.fault,c.fault)
                self.assertEqual(c.take_completion()['status'],'SETTLE_TIMEOUT_HOLD')

    def test_ready_matches_all_reference_checkout_samples_in_isolated_process(self):
        reference=Path(os.environ.get('REBOT_REFERENCE_SDK','/home/zzw/rebot_grasp/sdk/reBotArm_control_py'))
        if not reference.is_dir():self.skipTest('Reference SDK checkout unavailable')
        code = """
import sys,json
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,sys.argv[1])
from reBotArm_control_py.controllers import RebotArmEndPose
class Capture(RebotArmEndPose):
    def _send_loop(self,duration):self._moving=False
raw=np.zeros(7)
r=SimpleNamespace(arm_control_mode='mit',has_gripper=False,groups={'arm':SimpleNamespace(num_joints=6)},get_state=lambda:(raw.copy(),raw.copy(),raw.copy()))
c=Capture(r);c._running=True
assert c.move_to_traj(x=.3,y=0,z=.3,roll=0,pitch=.7,yaw=0,duration=3.)
c._send_thread.join()
print(json.dumps([q.tolist() for q in c._traj]))
"""
        output=subprocess.check_output([sys.executable,'-c',code,str(reference)],text=True,timeout=10)
        points=np.array(json.loads(output))
        self.assertEqual(self.ready.duration,3.)
        np.testing.assert_allclose(self.ready.points,points,atol=1e-10,rtol=0)
        # Execution must preserve every reference waypoint and its hold interval.
        n=len(points)
        for i in range(n):
            np.testing.assert_allclose(self.ready.sample((i+.25)*3/n)[0],points[i],atol=1e-10,rtol=0)

    def test_home_matches_reference_safe_home_targets_and_timing(self):
        start=self.ready.goal
        home=self.p.plan_home(start)
        duration=2*np.abs(start).max()/.5
        self.assertEqual(home.duration,duration)
        count=max(2,int(duration*50))
        s=np.linspace(0,1,count)
        expected=start[None,:]*(1-10*s[:,None]**3+15*s[:,None]**4-6*s[:,None]**5)
        np.testing.assert_allclose(home.points,expected,atol=1e-13)
        for i in range(count):
            np.testing.assert_allclose(home.sample((i+.25)*duration/count)[0],expected[i],atol=1e-13)
        np.testing.assert_array_equal(home.sample(home.duration+1)[0],np.zeros(6))

    def test_discrete_sdk_hold_has_no_spline_overshoot_and_can_slow_down(self):
        from wrist.sdk_transit import SDKTransitPlan
        plan=SDKTransitPlan(self.ready.points,6.,'ready',self.ready.target,'test')
        np.testing.assert_array_equal(plan.sample(3.)[0],self.ready.sample(1.5)[0])
        for t in np.linspace(0,6,1001):
            q=plan.sample(t)[0]
            self.p.validate_raw(q)
        np.testing.assert_array_equal(plan.sample(-1)[0],plan.points[0])
        np.testing.assert_array_equal(plan.sample(7)[0],plan.goal)

    def test_whole_transit_single_sender_and_hold_cancel(self):
        for plan in (self.ready,self.p.plan_home(self.ready.goal)):
            c,robot=self.controller(plan,plan.curve(0))
            for now in np.arange(0,plan.duration,.002):
                # One-cycle ideal plant, no physical motor/CAN access.
                c.feedback.snapshot.return_value=SimpleNamespace(positions=np.r_[c.target,.25])
                c.ui_seen=now
                with patch('eyehand.hover_control.time.perf_counter',return_value=float(now)):c._tick(robot,.002)
                self.assertIsNone(c.fault,c.fault)
            c.hold('USER_HOLD');held=c.target.copy()
            now=plan.duration+.1;c.ui_seen=now;c._last_tick=now-.002
            with patch('eyehand.hover_control.time.perf_counter',return_value=now):c._tick(robot,.002)
            np.testing.assert_array_equal(c.target,held)
            self.assertFalse(c.busy)
            np.testing.assert_array_equal(robot.arm.send_mit.call_args.kwargs['vel'],np.zeros(6))

    def test_existing_cartesian_outer_loop_converges_at_500hz(self):
        from test_cartesian_outer import CartesianOuterTests
        CartesianOuterTests.setUpClass();helper=CartesianOuterTests()
        case=helper.make_case()
        replay=helper.run_plant(case,tau=.5,quantized=True,period=.002)
        done=replay['completion']
        self.assertEqual(done['status'],'SETTLED_MODEL_ONLY')
        self.assertLessEqual(done['fk_error_mm'],1.)
        self.assertAlmostEqual(done['control_timing']['observed_callback_rate_hz'],500,places=6)
        self.assertFalse(done['physical_accuracy_verified'])


if __name__=='__main__':unittest.main()

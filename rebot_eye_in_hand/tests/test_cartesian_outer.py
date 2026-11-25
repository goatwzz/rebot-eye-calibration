"""Actual model + SDK with a biased lag plant. No camera, CAN or motor access."""
from pathlib import Path
import copy
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wrist.click_hover import WristPlanner, load_config
from eyehand.click_hover import MotionPlan
from legacy_model_fixture import load_legacy_config as load_config
from eyehand.hover_control import HoverController
from eyehand.cartesian_settle import CartesianSettler, validate_settings


class CartesianOuterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = [json.loads(x) for x in (ROOT/'runs/click_hover/20261120_212232_75e2ac79.jsonl').read_text().splitlines()]
        cls.record = next(r for r in rows if r.get('phase') == 'HOVER_MOVING')

    def make_case(self):
        p = WristPlanner(load_config())
        goal = np.array(self.record['raw_goal_rad'])
        target = np.array(self.record['target_base_m'])
        bias = goal-np.array(self.record['raw_q_rad'])
        plan = MotionPlan([goal, goal], p.cfg, 0., joint_space=True)
        plan.target = target
        robot = SimpleNamespace(arm=Mock(), gripper=Mock(), estop=Mock(), _motor_map={})
        c = HoverController(p, robot)
        c.enabled = True
        c.target = goal.copy()
        c.gripper = np.array([.25])
        c.feedback = Mock()
        c.feedback.snapshot.return_value = SimpleNamespace(positions=np.r_[goal-bias, .25])
        c.plan, c.started, c.stable_since = plan, 0., None
        return p, c, robot, goal, target, bias

    def tick(self, c, now, raw):
        c.feedback.snapshot.return_value.positions[:6] = raw
        c.ui_seen = now
        with patch('eyehand.hover_control.time.perf_counter', return_value=now):
            c._tick(c.robot, .01)
        self.assertIsNone(c.fault, c.fault)

    def run_plant(self, case, tau=.15, quantized=False, period=.01):
        p,c,robot,goal,target,bias = case
        actual = goal-bias
        commands, velocities, errors = [], [], []
        start = c.plan.duration+.01
        last = goal.copy()
        for i in range(int(15/period)):
            now = start+period*i
            measured = actual.copy()
            if quantized:
                quantum = .0003835069006732022
                measured = quantum*np.round(measured/quantum)
            self.tick(c, now, measured)
            command = robot.arm.send_mit.call_args.args[0].copy()
            commands.append(command)
            velocities.append((command-last)/period)
            errors.append(1000*np.linalg.norm(p.tcp(measured)-target))
            actual += (1-np.exp(-period/tau))*(command-bias-actual)
            last = command
            if not c.busy:
                break
        return dict(actual=actual, commands=np.array(commands), velocities=np.array(velocities),
                    errors=errors, completion=c.take_completion(), last_time=now)

    def test_recorded_bias_converges_with_lag_and_encoder_quantization(self):
        for tau, quantized in ((.05,False),(.15,False),(.5,True)):
            with self.subTest(tau=tau, quantized=quantized):
                case = self.make_case(); p,c,robot,goal,target,bias = case
                replay = self.run_plant(case,tau=tau,quantized=quantized)
                done = replay['completion']
                self.assertEqual(done['status'],'SETTLED_MODEL_ONLY')
                self.assertLessEqual(done['fk_error_mm'],1.)
                self.assertGreater(replay['errors'][0],3.)
                self.assertFalse(done['physical_accuracy_verified'])
                d = done['cartesian_outer_loop']
                self.assertEqual(d['solver'],'official_sdk_position_clik_single_step_integrated')
                self.assertLessEqual(d['peak_joint_command_speed_rad_s'],.02+1e-10)
                self.assertLessEqual(d['peak_tcp_command_speed_m_s'],.003+1e-10)
                self.assertLessEqual(np.abs(replay['commands']-goal).max(),.015+1e-10)
                self.assertLessEqual(np.abs(np.diff(replay['velocities'],axis=0)/.01).max(),.1+1e-7)
                self.assertIsNot(c.cartesian_settler.data,p.data)
                self.assertIsNot(c.cartesian_settler.data,p.camera_data)
                # Position bias remains in the held command after completion.
                held=c.target.copy()
                self.assertGreater(np.linalg.norm(held-goal),.001)
                self.tick(c,replay['last_time']+.01,replay['actual'])
                np.testing.assert_allclose(robot.arm.send_mit.call_args.args[0],held)
                np.testing.assert_array_equal(robot.gripper.send_mit.call_args.args[0],[.25])

    def test_next_trajectory_starts_at_corrected_command_without_a_jump(self):
        case=self.make_case();p,c,robot,goal,target,bias=case
        replay=self.run_plant(case)
        held=c.target.copy()
        np.testing.assert_allclose(c.planning_start(),held)
        next_target=target+np.array([.005,0,0])
        next_plan=p.plan(c.planning_start(),next_target)
        now=replay['last_time']+.01;c.ui_seen=now
        with patch('eyehand.hover_control.time.perf_counter',return_value=now):
            c.move(next_plan)
        self.assertIsNone(c.cartesian_settler)
        self.tick(c,now,replay['actual'])
        np.testing.assert_allclose(robot.arm.send_mit.call_args.args[0],held,atol=1e-12)

    def test_correction_saturation_times_out_without_windup(self):
        case=self.make_case();p,c,robot,goal,target,bias=case
        p.cfg['cartesian_outer_loop']['max_joint_offset_rad']=.001
        replay=self.run_plant(case)
        self.assertEqual(replay['completion']['status'],'SETTLE_TIMEOUT_HOLD')
        self.assertGreater(replay['completion']['fk_error_mm'],1.)
        self.assertLessEqual(np.abs(replay['commands']-goal).max(),.001+1e-12)
        self.assertGreater(replay['completion']['cartesian_outer_loop']['saturation_count'],0)
        self.assertTrue(c.enabled)

    def test_tcp_offset_and_workspace_bounds_block_correction(self):
        for mode in ('tcp_offset','workspace'):
            with self.subTest(mode=mode):
                case=self.make_case();p,c,robot,goal,target,bias=case
                if mode=='tcp_offset':p.cfg['cartesian_outer_loop']['max_tcp_offset_m']=.0005
                else:p.cfg['workspace_max_m'][0]=target[0]+.0001
                replay=self.run_plant(case)
                self.assertEqual(replay['completion']['status'],'SETTLE_TIMEOUT_HOLD')
                self.assertGreater(replay['completion']['cartesian_outer_loop']['saturation_count'],0)
                for command in replay['commands']:
                    tcp=p.tcp(command)
                    self.assertTrue(np.all(tcp<=p.cfg['workspace_max_m']))
                    if mode=='tcp_offset':self.assertLessEqual(np.linalg.norm(tcp-p.tcp(goal)),.0005+1e-12)

    def test_capture_refusal_and_user_hold_stop_integrating(self):
        p,c,robot,goal,target,bias=self.make_case()
        actual=goal.copy();actual[2]-=.04
        self.tick(c,c.plan.duration+.01,actual)
        done=c.take_completion()
        self.assertEqual(done['status'],'SETTLE_TIMEOUT_HOLD')
        self.assertEqual(done['cartesian_outer_loop']['stop_reason'],'capture_radius_exceeded')
        np.testing.assert_allclose(robot.arm.send_mit.call_args.args[0],goal)
        p,c,robot,goal,target,bias=self.make_case()
        start=c.plan.duration+.01
        for i in range(10):self.tick(c,start+i*.01,goal-bias)
        held=c.target.copy();c.hold()
        self.assertIsNone(c.cartesian_settler)
        self.tick(c,start+.1,goal-bias)
        np.testing.assert_allclose(robot.arm.send_mit.call_args.args[0],held)

    def test_stale_feedback_and_bad_sdk_result_never_send_correction(self):
        for fault in ('stale','nan_ik'):
            with self.subTest(fault=fault):
                p,c,robot,goal,target,bias=self.make_case()
                now=c.plan.duration+.01;self.tick(c,now,goal-bias)
                count=robot.arm.send_mit.call_count
                if fault=='stale':c.feedback.snapshot.side_effect=RuntimeError('stale feedback')
                out=SimpleNamespace(q=np.full(6,np.nan),error=float('nan'))
                with patch.object(p.ik,'solve_ik',return_value=out),patch('eyehand.hover_control.time.perf_counter',return_value=now+.01):
                    c._tick(robot,.01)
                self.assertIsNotNone(c.fault)
                self.assertEqual(robot.arm.send_mit.call_count,count)
                robot.estop.assert_called_once()
                self.assertIsNone(c.cartesian_settler)

    def test_ui_timeout_freezes_the_last_corrected_command(self):
        p,c,robot,goal,target,bias=self.make_case()
        start=c.plan.duration+.01
        for i in range(10):self.tick(c,start+i*.01,goal-bias)
        held=c.target.copy();now=start+.1
        c.ui_seen=now-p.cfg['ui_timeout_s']-.01
        with patch('eyehand.hover_control.time.perf_counter',return_value=now):
            c._tick(robot,.01)
        self.assertIsNone(c.fault)
        self.assertIsNone(c.cartesian_settler)
        self.assertEqual(c.take_completion()['status'],'UI_TIMEOUT_HOLD')
        np.testing.assert_allclose(robot.arm.send_mit.call_args.args[0],held)

    def test_disturbance_outside_capture_region_stops_an_active_outer_loop(self):
        p,c,robot,goal,target,bias=self.make_case()
        now=c.plan.duration+.01
        self.tick(c,now,goal-bias);self.tick(c,now+.01,goal-bias)
        held=c.target.copy();disturbed=goal-bias;disturbed[2]-=.04
        self.tick(c,now+.02,disturbed)
        done=c.take_completion()
        self.assertEqual(done['status'],'SETTLE_TIMEOUT_HOLD')
        self.assertEqual(done['cartesian_outer_loop']['stop_reason'],'capture_radius_exceeded')
        np.testing.assert_allclose(robot.arm.send_mit.call_args.args[0],held)

    def test_sdk_feedback_steps_handle_the_external_models_joint_offsets(self):
        from eyehand.click_hover import HoverPlanner,load_config as external_config
        cfg=external_config()
        cfg['cartesian_outer_loop']=validate_settings({'enabled':True},cfg)
        p=HoverPlanner(cfg)
        self.assertGreater(np.linalg.norm(p.offsets),0.)
        goal=np.array(self.record['raw_goal_rad']);target=p.tcp(goal)
        bias=goal-np.array(self.record['raw_q_rad']);actual=goal-bias
        s=CartesianSettler(p,goal,target)
        for i in range(601):
            command=s.step(actual,.01*i)
            actual+=(1-np.exp(-.01/.15))*(command-bias-actual)
        self.assertLess(np.linalg.norm(p.tcp(actual)-target),.001)

    def test_outer_loop_is_not_used_during_trajectory_or_raw_home(self):
        p,c,robot,goal,target,bias=self.make_case()
        self.tick(c,c.plan.duration/2,goal)
        self.assertIsNone(c.cartesian_settler)
        home=p.plan_home(goal);c.plan=home;c.stable_since=None;c._last_tick=0
        actual=np.zeros(6);actual[2]=.012
        now=home.duration+p.cfg['settle_timeout_s']+.01;c._last_tick=now-.01
        self.tick(c,now,actual)
        done=c.take_completion()
        self.assertEqual(done['status'],'HOME_TIMEOUT_HOLD')
        self.assertNotIn('cartesian_outer_loop',done)
        np.testing.assert_array_equal(robot.arm.send_mit.call_args.args[0],np.zeros(6))

    def test_configuration_rejects_excessive_corrections_and_loose_acceptance(self):
        cfg=load_config()
        for bad in ({'max_joint_offset_rad':.1},{'max_tcp_speed_m_s':.1},
                    {'gain_per_s':float('nan')},{'tolerance_m':.01},{'enabled':'yes'}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                validate_settings(bad,cfg)


if __name__=='__main__':unittest.main()

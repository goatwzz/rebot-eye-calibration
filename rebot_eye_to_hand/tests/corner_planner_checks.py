"""Run in a fresh SDK process; never construct camera/robot/CAN objects."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WRIST = '--wrist' in sys.argv
if WRIST:
    sys.argv.remove('--wrist')
    sys.path.insert(0, str(ROOT.parent/'rebot_eye_in_hand'))
    from wrist.click_hover import WristPlanner as Planner, load_config
    CONFIG = ROOT.parent/'rebot_eye_in_hand/config/click_hover.json'
else:
    from eyehand.click_hover import HoverPlanner as Planner, load_config
    CONFIG = ROOT/'click_hover_config.json'
from eyehand.aruco_corners import plan_corner, down_angle_deg
from eyehand.corner_demo import load_settings
from eyehand.hover_control import HoverController
from eyehand.sdk_hover import SDKHoverPlan
from eyehand.cartesian_settle import CartesianTracker


class CornerPlannerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Planner(load_config(CONFIG))
        cls.s = load_settings(ROOT/'aruco_corners_config.json', 'wrist' if WRIST else 'external')
        cls.p.cfg.update(home_position_tolerance_m=.009, home_tolerance_rad=.016)
        cls.raw = cls.p.plan_ready(np.zeros(6)).goal if WRIST else np.zeros(6)
        cls.plans = [plan_corner(cls.p, cls.raw, np.array(goal), cls.s) for goal in
            ([.33, -.03, .122], [.39, -.03, .122], [.39, .03, .122], [.33, .03, .122])]

    def test_all_four_corners_keep_official_samples_downward_pose_and_xyz_outer(self):
        p = self.p
        for plan in self.plans:
            self.assertIsInstance(plan, SDKHoverPlan)
            self.assertTrue(plan.tracking_outer_loop)
            self.assertEqual(p.cfg['cartesian_outer_loop']['axes'], 'xyz')
            np.testing.assert_allclose(plan.points[0], self.raw, atol=1e-10)
            self.assertLess(np.linalg.norm(p.tcp(plan.goal)-plan.target), 2e-4)
            self.assertGreater(down_angle_deg(p, plan.goal), 14.9)
            self.assertEqual(plan.diagnostics['terminal_orientation'], 'official_full_pose_endpoint_IK')
            self.assertEqual(len(plan.points), plan.diagnostics['clik_success_count'])
            self.assertGreater(len(plan.points), 100)
            d, o = plan.diagnostics, p.cfg['cartesian_outer_loop']
            self.assertLessEqual(d['nominal_peak_joint_speed_rad_s']+o['max_joint_speed_rad_s'], p.cfg['max_joint_speed_rad_s'])
            self.assertLessEqual(d['nominal_peak_joint_accel_rad_s2']+o['max_joint_accel_rad_s2'], p.cfg['max_joint_accel_rad_s2'])
            self.assertLessEqual(d['nominal_peak_tcp_speed_m_s']+o['max_tcp_speed_m_s'], p.cfg['max_tcp_speed_m_s'])
            home = p.plan_home(plan.goal)
            np.testing.assert_allclose(home.goal, np.zeros(6), atol=1e-12)
            if WRIST:
                ready = p.plan_ready(plan.goal)
                self.assertLess(np.linalg.norm(p.tcp(ready.goal)-[.3, 0, .3]), .0002)

    def test_new_downward_endpoint_can_settle_xyz_with_feedback_bias(self):
        p, plan = self.p, self.plans[0]
        tracker = CartesianTracker(p, plan.goal, plan.target)
        bias = np.array([0, .003, -.004, .002, -.002, .001])
        actual, command = plan.goal-bias, plan.goal.copy()
        for i in range(1001):
            now = plan.duration+i*.01
            command = tracker.advance(plan.goal, plan.target, actual, now, settling=True)
            actual += .15*(command-bias-actual)
            if tracker.stop_reason:
                self.fail(tracker.stop_reason)
        self.assertLess(np.linalg.norm(p.tcp(actual)-plan.target), .001)

    def test_entire_four_corner_approaches_track_with_lag_and_quantized_feedback(self):
        p = self.p
        bias = np.array([0., .003, -.004, .002, -.002, .001])
        quantum, dt = .0003835069006732131, .002  # Actual 500 Hz sender cadence.
        for number, plan in enumerate(self.plans, 1):
            with self.subTest(corner=number):
                start = plan.sample(0.)[0]
                tracker = CartesianTracker(p, start, plan.reference(0.))
                actual, previous, stable_since = start-bias, start.copy(), None
                peak_command_step = 0.
                for i in range(int((plan.duration+10.)/dt)+1):
                    now = i*dt
                    nominal = plan.sample(now)[0]
                    measured = quantum*np.round(actual/quantum)
                    command = tracker.advance(nominal, plan.reference(now), measured, now,
                                              settling=now >= plan.duration)
                    self.assertIsNone(tracker.stop_reason)
                    peak_command_step = max(peak_command_step, float(np.abs(command-previous).max()))
                    self.assertLessEqual(np.max(np.abs(command-nominal)),
                                         p.cfg['cartesian_outer_loop']['max_joint_offset_rad']+1e-10)
                    previous = command.copy()
                    actual += dt/(.0567+dt)*(command-bias-actual)
                    error = np.linalg.norm(p.tcp(measured)-plan.target)
                    if now >= plan.duration and error <= .001 and tracker.quiet:
                        stable_since = now if stable_since is None else stable_since
                        if now-stable_since >= .5:
                            break
                    else:
                        stable_since = None
                self.assertIsNotNone(stable_since, 'Did not settle within configured timeout')
                self.assertLessEqual(error, .001)
                # SDK targets use zero-order hold; jump/dt at 500 Hz is not a
                # physical velocity. Check nominal sample + bounded outer step.
                allowed_step = (plan.diagnostics['max_adjacent_joint_step_rad']
                                +p.cfg['cartesian_outer_loop']['max_joint_speed_rad_s']*.02+1e-5)
                self.assertLessEqual(peak_command_step, allowed_step)

    def test_historical_home_bias_is_accepted_only_by_new_home_tolerances(self):
        p = self.p
        measured = np.radians([0., .549333, .6592, .483413, -.307627, .13184])
        for is_home, relaxed, expected in ((True, True, 'HOME_REACHED_HOLD'),
                                           (True, False, 'HOME_TIMEOUT_HOLD'),
                                           (False, True, 'SETTLE_TIMEOUT_HOLD')):
            cfg = dict(p.cfg)
            cfg['cartesian_outer_loop'] = dict(enabled=False)
            if not relaxed:
                cfg.pop('home_position_tolerance_m')
                cfg['home_tolerance_rad'] = .01
            robot = NS(arm=Mock(), gripper=Mock(), estop=Mock())
            ctrl = HoverController(p, robot)
            ctrl.cfg = cfg
            ctrl.enabled, ctrl.target, ctrl.gripper = True, np.zeros(6), np.array([.25])
            ctrl.feedback = Mock()
            ctrl.feedback.snapshot.return_value = NS(positions=np.r_[measured, .25])
            ctrl.plan = NS(kind='home' if is_home else 'hover', duration=1., goal=np.zeros(6),
                           target=p.tcp(np.zeros(6)), sample=lambda t: (np.zeros(6), np.zeros(6)))
            ctrl.started, ctrl.ui_seen, ctrl.stable_since = 0., 10., 9.
            with patch('eyehand.hover_control.time.perf_counter', return_value=10.):
                ctrl._tick(robot, .002)
            self.assertIsNone(ctrl.fault)
            complete = ctrl.take_completion()
            self.assertEqual(complete['status'], expected)
            if is_home and relaxed:
                self.assertLess(complete['fk_error_mm'], 9.)
                self.assertEqual(complete['home_position_tolerance_mm'], 9.)
                self.assertFalse(complete['physical_accuracy_verified'])

    def test_corner_home_finishes_sdk_trajectory_without_precision_gate(self):
        p = self.p
        measured = np.array([-.00076699, .00306796, .01227185, .01917476, .01227185, .00191748])
        for elapsed in (.2, 1.5):
            robot = NS(arm=Mock(), gripper=Mock(), estop=Mock())
            ctrl = HoverController(p, robot)
            ctrl.cfg = dict(p.cfg, home_completion_mode='trajectory', cartesian_outer_loop=dict(enabled=False))
            ctrl.enabled, ctrl.target, ctrl.gripper = True, np.zeros(6), np.array([.25])
            ctrl.feedback = Mock()
            ctrl.feedback.snapshot.return_value = NS(positions=np.r_[measured, .25])
            ctrl.plan = NS(kind='home', duration=1., goal=np.zeros(6),
                           target=p.tcp(np.zeros(6)), sample=lambda t: (np.zeros(6), np.zeros(6)))
            ctrl.started, ctrl.ui_seen, ctrl.stable_since = 10.-elapsed, 10., None
            with patch('eyehand.hover_control.time.perf_counter', return_value=10.):
                ctrl._tick(robot, .002)
            self.assertIsNone(ctrl.fault)
            complete = ctrl.take_completion()
            if elapsed < 1.5:
                self.assertIsNone(complete)
                self.assertIsNotNone(ctrl.plan)
            else:
                self.assertEqual(complete['status'], 'HOME_COMMAND_FINISHED_HOLD')
                self.assertGreater(complete['home_joint_error_deg'], 1.)
                self.assertFalse(complete['physical_accuracy_verified'])
                self.assertFalse(ctrl.home_confirmed)
                self.assertIsNone(ctrl.plan)

    def test_cancelled_planning_never_returns_a_plan(self):
        with self.assertRaises(InterruptedError):
            plan_corner(self.p, self.raw, self.plans[0].target, self.s, lambda: True)


if __name__ == '__main__':
    unittest.main()

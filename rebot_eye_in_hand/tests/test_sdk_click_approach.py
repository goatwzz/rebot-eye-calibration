"""Wrist model with the shared public SDK trajectory and mocked MIT transport."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wrist.click_hover import WristPlanner, load_config
from legacy_model_fixture import load_legacy_config as load_config
from eyehand.hover_control import HoverController
from eyehand.cartesian_settle import CartesianTracker
from eyehand.sdk_hover import SDKHoverPlan


class WristSDKClickApproachTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = WristPlanner(load_config())
        cls.ready = cls.p.plan_ready(np.zeros(6))
        cfg = cls.p.cfg
        cls.target = np.array([.4, 0., cfg['table_reference']['table_z_m']
                               +cfg['hover_m']+cfg['height_compensation_m']])
        cls.plan = cls.p.plan(cls.ready.goal, cls.target)

    def test_wrist_uses_the_external_sdk_path_and_speed_budget(self):
        p, plan = self.p, self.plan
        self.assertIsInstance(plan, SDKHoverPlan)
        self.assertEqual(p.cfg['planning_mode'], 'sdk_cartesian')
        self.assertEqual(p.cfg['control_rate_hz'], 500)
        self.assertEqual(p.cfg['cartesian_outer_loop']['axes'], 'xyz')
        self.assertTrue(plan.tracking_outer_loop)
        self.assertFalse(plan.mit_velocity_feedforward)
        np.testing.assert_array_equal(p.offsets, np.zeros(6))
        np.testing.assert_allclose(plan.points[0], self.ready.goal, atol=1e-12)
        self.assertLess(np.linalg.norm(p.tcp(plan.goal)-self.target), .0002)
        d, c, o = plan.diagnostics, p.cfg, p.cfg['cartesian_outer_loop']
        self.assertEqual(d['clik_success_count'], len(plan.points))
        self.assertLessEqual(d['nominal_peak_joint_speed_rad_s']+o['max_joint_speed_rad_s'], c['max_joint_speed_rad_s'])
        self.assertLessEqual(d['nominal_peak_joint_accel_rad_s2']+o['max_joint_accel_rad_s2'], c['max_joint_accel_rad_s2'])
        self.assertLessEqual(d['nominal_peak_tcp_speed_m_s']+o['max_tcp_speed_m_s'], c['max_tcp_speed_m_s'])

    def test_xyz_outer_tracks_during_motion_and_settles_with_original_wrist_bias(self):
        p, plan = self.p, self.plan
        rows = [json.loads(line) for line in
                (ROOT/'runs/click_hover/20261120_212232_75e2ac79.jsonl').read_text().splitlines()]
        record = next(row for row in rows if row.get('phase') == 'HOVER_MOVING')
        bias = np.array(record['raw_goal_rad'])-record['raw_q_rad']
        robot = SimpleNamespace(arm=Mock(), gripper=Mock(), estop=Mock(), _motor_map={})
        controller = HoverController(p, robot)
        controller.enabled = True
        controller.target = self.ready.goal.copy()
        controller.gripper = np.array([.25])
        controller.feedback = Mock()
        controller.feedback.snapshot.return_value = SimpleNamespace(positions=np.r_[self.ready.goal, .25])
        controller.plan, controller.started, controller.stable_since = plan, 0., None
        actual = self.ready.goal.copy(); dt = .002; tracked_before_endpoint = False
        for i in range(int((plan.duration+11)/dt)):
            now = i*dt
            measured = .0003835069006732131*np.round(actual/.0003835069006732131)
            controller.feedback.snapshot.return_value.positions[:6] = measured
            controller.ui_seen = now
            with patch('eyehand.hover_control.time.perf_counter', return_value=now):
                controller._tick(robot, dt)
            self.assertIsNone(controller.fault, controller.fault)
            np.testing.assert_array_equal(robot.arm.send_mit.call_args.kwargs['vel'], np.zeros(6))
            np.testing.assert_array_equal(robot.gripper.send_mit.call_args.args[0], [.25])
            actual += (1-np.exp(-dt/.08))*(controller.target-bias-actual)
            if now < plan.duration and isinstance(controller.cartesian_settler, CartesianTracker):
                tracked_before_endpoint = True
            if not controller.busy:
                break
        result = controller.take_completion()
        self.assertTrue(tracked_before_endpoint)
        self.assertIsNotNone(result)
        self.assertEqual(result['status'], 'SETTLED_MODEL_ONLY', result)
        self.assertLessEqual(result['fk_error_mm'], 1.)
        self.assertTrue(result['cartesian_outer_loop']['correction_stopped'])
        self.assertTrue(result['cartesian_outer_loop']['z_feedback_enabled'])
        self.assertGreater(result['cartesian_outer_loop']['trajectory_updates'], 0)
        self.assertGreater(result['cartesian_outer_loop']['settling_updates'], 0)
        self.assertFalse(result['physical_accuracy_verified'])
        held = controller.target.copy()
        returned = p.plan_ready(controller.planning_start())
        np.testing.assert_allclose(returned.points[0], held, atol=1e-12)
        self.assertLess(np.linalg.norm(p.tcp(returned.goal)-[.3, 0., .3]), .0002)
        self.assertFalse(getattr(returned, 'tracking_outer_loop', False))
        pin = p.pin; data = p.model.createData()
        pin.framesForwardKinematics(p.model, data, returned.goal)
        self.assertLess(np.linalg.norm(data.oMf[p.tcp_id].rotation-pin.rpy.rpyToMatrix(0., .7, 0.)), .001)


if __name__ == '__main__':
    unittest.main()

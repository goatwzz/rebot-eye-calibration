"""Completion telemetry uses fresh feedback for success and timeout; no hardware."""
import json
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pinocchio as pin

from eyehand.hover_control import HoverController


class CompletionTelemetryTests(unittest.TestCase):
    def test_success_and_timeout_keep_feedback_and_continue_holding(self):
        for kind in ('hover', 'home'):
            for timed_out in (False, True):
                with self.subTest(kind=kind, timed_out=timed_out):
                    model = pin.buildSampleModelManipulator()
                    q = np.array([0., .3, -.2, 0., .1, 0.])
                    data = model.createData()
                    pin.forwardKinematics(model, data, q)
                    pin.updateFramePlacements(model, data)
                    tcp = data.oMf[-1].translation.copy()
                    delta = np.array([.004, -.003, .002]) if timed_out else np.zeros(3)
                    goal = q + (.01 if timed_out else 0.)
                    cfg = dict(feedback_max_age_s=.15, ui_timeout_s=.5,
                               settle_tolerance_m=.003, home_tolerance_rad=.005,
                               settle_timeout_s=.25, tracking_limit_rad=.12,
                               kp=[50.]*6, kd=[4.]*6)
                    planner = SimpleNamespace(model=model, offsets=np.zeros(6),
                                              tcp_id=len(model.frames)-1, cfg=cfg,
                                              validate_feedback=lambda value: value,
                                              validate_raw=lambda value: value,
                                              gravity_torque=lambda value: np.zeros(6))
                    robot = SimpleNamespace(arm=Mock(), gripper=Mock(), estop=Mock())
                    controller = HoverController(planner, robot)
                    controller.enabled = True
                    controller.target = q.copy()
                    controller.gripper = np.array([.25])
                    controller.feedback = Mock()
                    controller.feedback.snapshot.return_value = SimpleNamespace(positions=np.r_[q, .25])
                    controller.plan = SimpleNamespace(kind=kind, duration=1., goal=goal,
                                                      target=tcp-delta,
                                                      sample=lambda elapsed: (goal.copy(), np.zeros(6)))
                    controller.started, controller.ui_seen = 8., 10.
                    controller.stable_since = 9.
                    with patch('eyehand.hover_control.time.perf_counter', return_value=10.):
                        controller._tick(robot, .01)
                    self.assertIsNone(controller.fault)
                    record = controller.take_completion()
                    expected = ('HOME_TIMEOUT_HOLD' if kind == 'home' else 'SETTLE_TIMEOUT_HOLD') if timed_out else (
                        'HOME_REACHED_HOLD' if kind == 'home' else 'SETTLED_MODEL_ONLY')
                    self.assertEqual(record['status'], expected)
                    np.testing.assert_allclose(record['raw_q_rad'], q)
                    np.testing.assert_allclose(record['raw_goal_rad'], goal)
                    np.testing.assert_allclose(record['joint_feedback_minus_goal_deg'], np.degrees(q-goal))
                    np.testing.assert_allclose(record['tcp_base_m'], tcp)
                    np.testing.assert_allclose(record['target_base_m'], tcp-delta)
                    np.testing.assert_allclose(record['fk_error_xyz_mm'], 1000*delta, atol=1e-10)
                    self.assertAlmostEqual(record['fk_error_mm'], 1000*np.linalg.norm(delta))
                    self.assertEqual(record['elapsed_s'], 2.)
                    self.assertEqual(record['planned_duration_s'], 1.)
                    self.assertFalse(record['physical_accuracy_verified'])
                    if kind == 'home':
                        self.assertAlmostEqual(record['home_joint_error_deg'], float(np.max(np.abs(np.degrees(q-goal)))))
                    json.dumps(record, allow_nan=False)
                    self.assertTrue(controller.enabled)
                    self.assertIsNone(controller.plan)
                    robot.estop.assert_not_called()
                    np.testing.assert_allclose(robot.arm.send_mit.call_args.args[0], goal)
                    np.testing.assert_allclose(robot.gripper.send_mit.call_args.args[0], [.25])
                    # The subsequent tick keeps sending the same joint goal.
                    with patch('eyehand.hover_control.time.perf_counter', return_value=10.01):
                        controller._tick(robot, .01)
                    self.assertIsNone(controller.fault)
                    np.testing.assert_allclose(robot.arm.send_mit.call_args.args[0], goal)
                    self.assertIsNone(controller.take_completion())


if __name__ == '__main__':
    unittest.main()

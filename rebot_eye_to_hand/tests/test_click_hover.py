from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from eyehand.click_hover import ROOT, depth_at_pixel, hover_target, load_config
from eyehand.hover_control import HoverController
from eyehand.paths import artifact_path


class ClickGeometryTests(unittest.TestCase):
    def test_above_is_base_z_after_camera_transform(self):
        T = np.array([[0,0,1,.5], [1,0,0,.1], [0,1,0,.2], [0,0,0,1.]])
        surface, target = hover_target(T, [.1,.2,.6])
        np.testing.assert_allclose(surface, [1.1,.2,.4])
        np.testing.assert_allclose(target, [1.1,.2,.5])

    def test_depth_patch_accepts_solid_surface_but_rejects_holes_and_edges(self):
        depth = np.full((30,30), .7)
        self.assertAlmostEqual(depth_at_pixel(depth,15,15)[0], .7)
        depth[15,15] = 0
        with self.assertRaises(ValueError): depth_at_pixel(depth,15,15)
        depth[:] = .7
        depth[:,16:] = 1.
        with self.assertRaisesRegex(ValueError, 'discontinuity'): depth_at_pixel(depth,15,15)
        with self.assertRaises(ValueError): depth_at_pixel(depth,0,0)
        depth[:] = np.nan
        with self.assertRaises(ValueError): depth_at_pixel(depth,15,15)

    def test_relocation_remaps_only_project_artifacts(self):
        self.assertEqual(artifact_path('/home/zzw/robot_sim/rebot_eye_to_hand/runs/train_03/a.png'), ROOT/'runs/train_03/a.png')
        other = Path('/home/zzw/reBotArm_control_py/config/rebotarm_rs.yaml')
        self.assertEqual(artifact_path(other),other)

    def test_mesh_distal_tip_report_and_config_agree(self):
        from scripts.analyze_tcp_tip import analyze
        report = json.loads((ROOT/'reports/tcp_from_mesh.json').read_text())
        recomputed = analyze(report['urdf'])
        np.testing.assert_allclose(recomputed['recommended_tcp_offset_m'], load_config()['tcp_offset_m'])
        # Independent wrist/tool geometry: the motor lies behind both finger
        # meshes, so the distal end is their maximum X, not the wide rear rail.
        self.assertAlmostEqual(recomputed['wrist_motor_center_gripper_end_m'][0], -.16621, places=7)
        np.testing.assert_allclose(recomputed['fingertip_outward_axis_gripper_end'], [1,0,0],atol=1e-5)
        np.testing.assert_array_equal(recomputed['recommended_tcp_offset_m'], [0,0,0])
        self.assertAlmostEqual(recomputed['fingers'][0]['distal_x_m'], 0., places=6)
        self.assertLess(recomputed['fingers'][0]['bounds_gripper_end_m'][0][0], -.088)
        self.assertFalse(report['physical_tcp_verified'])

    def test_real_model_planning_in_isolated_sdk_process(self):
        # Other tests deliberately use the legacy SDK. Isolate modern SDK imports.
        result = subprocess.run([sys.executable, str(ROOT/'tests/hover_planner_checks.py')],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


class ControlFaultTests(unittest.TestCase):
    def make_controller(self):
        planner = SimpleNamespace(cfg=load_config(), gravity_torque=lambda q: np.zeros(6),
                                  validate_raw=lambda q: q, validate_feedback=lambda q:q)
        robot = SimpleNamespace(arm=Mock(),gripper=Mock(),estop=Mock(),stop_control_loop=Mock(),
                                disconnect=Mock(),_motor_map={})
        controller = HoverController(planner,robot)
        controller.enabled = controller.connected = True
        controller.target=np.array([.1,.2,.3,.1,.2,.3])
        controller.gripper=np.array([.25])
        controller.feedback=Mock()
        controller.feedback.snapshot.return_value=SimpleNamespace(positions=np.r_[controller.target,.25])
        return controller,robot

    def test_first_command_holds_actual_arm_and_gripper_not_zero(self):
        ctrl,robot=self.make_controller()
        ctrl._tick(robot,.01)
        np.testing.assert_allclose(robot.arm.send_mit.call_args.args[0],ctrl.target)
        np.testing.assert_allclose(robot.gripper.send_mit.call_args.args[0],[.25])
        self.assertTrue(robot.arm.send_mit.call_args.kwargs['strict'])
        robot.estop.assert_not_called()

    def test_stale_feedback_disables_without_sending_motion(self):
        ctrl,robot=self.make_controller()
        ctrl.feedback.snapshot.side_effect=RuntimeError('stale feedback')
        ctrl._tick(robot,.01)
        robot.estop.assert_called_once()
        robot.arm.send_mit.assert_not_called()
        self.assertIn('stale feedback',ctrl.fault)
        ctrl._tick(robot,.01)
        robot.estop.assert_called_once()

    def test_tracking_fault_latches_and_cancels_motion(self):
        ctrl,robot=self.make_controller()
        ctrl.feedback.snapshot.return_value.positions[:6] += .2
        ctrl._tick(robot,.01)
        robot.estop.assert_called_once()
        self.assertIn('tracking',ctrl.fault)

    def test_send_fault_disables_and_does_not_send_gripper(self):
        ctrl,robot=self.make_controller()
        ctrl.robot.arm.send_mit.side_effect=RuntimeError('send error')
        ctrl._tick(robot,.01)
        robot.estop.assert_called_once()
        robot.gripper.send_mit.assert_not_called()

    def test_hold_cancels_plan_and_close_never_homes(self):
        ctrl,robot=self.make_controller()
        ctrl.plan=object()
        ctrl.hold()
        self.assertIsNone(ctrl.plan)
        saved=ctrl.target.copy()
        ctrl.close()
        np.testing.assert_allclose(ctrl.target,saved)
        robot.disconnect.assert_called_once()
        robot.arm.send_mit.assert_not_called()
        robot.arm.safe_home.assert_not_called()

    def test_partial_startup_failure_requests_disable(self):
        ctrl,robot=self.make_controller()
        ctrl._start_impl=Mock(side_effect=RuntimeError('partial enable'))
        with self.assertRaises(RuntimeError): ctrl.start()
        robot.estop.assert_called_once()
        self.assertFalse(ctrl.enabled)


if __name__=='__main__':unittest.main()

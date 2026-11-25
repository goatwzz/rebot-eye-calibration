"""Real calibrated model, official SDK, and simulated plants; no hardware I/O."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyehand.click_hover import HoverPlanner, load_config
from eyehand.hover_control import HoverController
from eyehand.cartesian_settle import CartesianTracker


class SDKHoverChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = HoverPlanner(load_config())
        rows = [json.loads(x) for x in (ROOT/'runs/click_hover/20261122_230809_91e5fde6.jsonl').read_text().splitlines()]
        cls.click = next(r for r in rows if r['event'] == 'click')
        cls.start = np.array(cls.click['raw_start_rad'])
        cls.target = np.array(cls.click['tcp_goal_base_m'])
        cls.plan = cls.p.plan(cls.start, cls.target)

    def controller(self, plan=None, raw=None, planner=None):
        plan = self.plan if plan is None else plan
        raw = self.start.copy() if raw is None else raw.copy()
        robot = SimpleNamespace(arm=Mock(), gripper=Mock(), estop=Mock(), _motor_map={})
        c = HoverController(self.p if planner is None else planner, robot)
        c.enabled = True; c.target = raw.copy(); c.gripper = np.array([.25])
        c.feedback = Mock()
        c.feedback.snapshot.return_value = SimpleNamespace(positions=np.r_[raw, .25])
        c.plan, c.started, c.stable_since = plan, 0., None
        return c

    def tick(self, c, now, raw, dt=.002):
        c.feedback.snapshot.return_value.positions[:6] = raw
        c.ui_seen = now
        with patch('eyehand.hover_control.time.perf_counter', return_value=now):
            c._tick(c.robot, dt)
        self.assertIsNone(c.fault, c.fault)

    def test_every_sample_matches_the_public_sdk_sampler_and_clik(self):
        from reBotArm_control_py.trajectory import (
            plan_cartesian_geodesic_trajectory, track_trajectory, TrajPlanParams, TrajProfile)
        from reBotArm_control_py.trajectory.clik_tracker import IKParams
        p, plan = self.p, self.plan
        data = p.model.createData()
        def pose(raw):
            p.pin.framesForwardKinematics(p.model, data, raw+p.offsets)
            return data.oMf[p.tcp_id].copy()
        # Endpoint selection uses position-only SDK IK, not a downward pose.
        out = p.ik.solve_ik(p.model, data, p.tcp_id, p.pin.SE3(np.eye(3), self.target),
            self.start+p.offsets, p.ik.IKParams(max_iter=1000, tolerance=1e-7),
            controlled_joints=6, position_only=True)
        cart = plan_cartesian_geodesic_trajectory(pose(self.start), pose(out.q-p.offsets),
            plan.duration, TrajPlanParams(dt=.01, profile=TrajProfile.MIN_JERK))
        reference = track_trajectory(p.model, p.tcp_id, cart.trajectory, self.start+p.offsets,
                                    IKParams(tolerance=1e-7), null_gain=0.)
        self.assertTrue(all(x.ik_success for x in reference))
        np.testing.assert_allclose(plan.points, np.array([x.q-p.offsets for x in reference]), atol=1e-12, rtol=0)
        for i in range(len(plan.points)):
            t = (i+.25)*plan.duration/len(plan.points)
            np.testing.assert_array_equal(plan.sample(t)[0], plan.points[i])
            np.testing.assert_array_equal(plan.sample(t)[1], np.zeros(6))
            np.testing.assert_allclose(plan.reference(t), p.tcp(plan.points[i]), atol=1e-12)
        self.assertLess(np.linalg.norm(p.tcp(plan.goal)-self.target), .0002)
        self.assertFalse(plan.mit_velocity_feedforward)
        self.assertEqual(p.cfg['control_rate_hz'], 500)

    def test_nominal_path_reserves_the_outer_correction_budget(self):
        d = self.plan.diagnostics; c = self.p.cfg; o = c['cartesian_outer_loop']
        self.assertLessEqual(d['nominal_peak_joint_speed_rad_s']+o['max_joint_speed_rad_s'], c['max_joint_speed_rad_s'])
        self.assertLessEqual(d['nominal_peak_joint_accel_rad_s2']+o['max_joint_accel_rad_s2'], c['max_joint_accel_rad_s2'])
        self.assertLessEqual(d['nominal_peak_tcp_speed_m_s']+o['max_tcp_speed_m_s'], c['max_tcp_speed_m_s'])
        self.assertEqual(c['height_compensation_m'], 0.0)
        self.assertEqual(c['cartesian_outer_loop']['axes'], 'xyz')

    def test_outer_tracks_the_moving_reference_not_the_final_click(self):
        c = self.controller()
        self.tick(c, 0., self.start)
        self.assertIsInstance(c.cartesian_settler, CartesianTracker)
        np.testing.assert_allclose(c.cartesian_settler.target, self.p.tcp(self.start), atol=1e-12)
        self.assertGreater(np.linalg.norm(c.cartesian_settler.target-self.target), .1)
        for t in np.arange(.002, 1., .002):
            self.tick(c, float(t), self.plan.sample(t)[0])
        self.assertGreater(c.cartesian_settler.trajectory_updates, 50)
        self.assertEqual(c.cartesian_settler.settling_updates, 0)
        self.assertLess(np.linalg.norm(c.cartesian_settler.command-c.cartesian_settler.nominal), 1e-10)
        np.testing.assert_array_equal(c.robot.arm.send_mit.call_args.kwargs['vel'], np.zeros(6))

    def test_lag_and_quantized_feedback_converge_with_smooth_final_hold(self):
        # Recorded pre-outer static tracking bias, plus a first-order lag.
        old = [json.loads(x) for x in (ROOT/'runs/click_hover/20261122_225255_1d1be4bf.jsonl').read_text().splitlines()]
        done = next(r for r in old if r.get('status') == 'SETTLED_MODEL_ONLY')
        bias = np.array(done['raw_goal_rad'])-done['raw_q_rad']
        c = self.controller(); actual = self.start.copy(); dt = .002
        peak_xy = 0.; stopped_at = None
        for i in range(int((self.plan.duration+11)/dt)):
            now = i*dt
            measured = .0003835069006732131*np.round(actual/.0003835069006732131)
            self.tick(c, now, measured, dt)
            command = c.target.copy()
            np.testing.assert_array_equal(c.robot.arm.send_mit.call_args.kwargs['vel'], np.zeros(6))
            actual += (1-np.exp(-dt/.08))*(command-bias-actual)
            if now < self.plan.duration:
                peak_xy = max(peak_xy, np.linalg.norm((self.p.tcp(measured)-self.plan.reference(now))[:2]))
            if not c.busy:
                stopped_at = now; break
        result = c.take_completion()
        self.assertIsNotNone(result)
        self.assertEqual(result['status'], 'SETTLED_MODEL_ONLY', result)
        self.assertLessEqual(result['fk_error_xy_mm'], 1.)
        self.assertLessEqual(result['fk_error_mm'], 1.)
        self.assertTrue(result['cartesian_outer_loop']['correction_stopped'])
        self.assertGreater(result['cartesian_outer_loop']['trajectory_updates'], 0)
        self.assertGreater(result['cartesian_outer_loop']['settling_updates'], 0)
        self.assertFalse(result['physical_accuracy_verified'])
        self.assertTrue(result['cartesian_outer_loop']['z_feedback_enabled'])
        traces = c.take_motion_traces()
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0][1].shape[1], 34)
        samples = traces[0][1]
        steps = np.diff(samples[:, 0])
        bias_history = samples[:, 8:14]-samples[:, 2:8]
        bias_velocity = np.diff(bias_history, axis=0)/steps[:, None]
        transport_peak = float(np.abs(bias_velocity).max())
        self.assertLessEqual(transport_peak, self.p.cfg['cartesian_outer_loop']['max_joint_speed_rad_s']+1e-8)
        command_z_correction = []
        for row in samples[::10]:
            correction = self.p.tcp(row[8:14])-self.p.tcp(row[2:8])
            command_z_correction.append(abs(correction[2]))
            self.assertLessEqual(np.linalg.norm(correction), .008+1e-10)
        held = c.target.copy()
        self.tick(c, stopped_at+dt, actual)
        np.testing.assert_array_equal(c.target, held)
        # Next click begins exactly at the held command (bias not discarded).
        next_plan = self.p.plan(c.planning_start(), self.target+[.003, 0, 0])
        now = stopped_at+2*dt; c.ui_seen = now
        with patch('eyehand.hover_control.time.perf_counter', return_value=now): c.move(next_plan)
        self.tick(c, now, actual)
        np.testing.assert_allclose(c.target, held, atol=1e-12)
        output = dict(synthetic_plant=True, hardware_accessed=False,
            physical_accuracy_verified=False, lag_s=.08, quantized_feedback=True,
            peak_tracking_xy_error_mm=peak_xy*1000, completion=result)
        output.update(peak_total_correction_speed_rad_s=transport_peak,
                      peak_command_z_correction_mm=max(command_z_correction)*1000)
        Path('/tmp/rebot_sdk_hover/replay_report.json').write_text(json.dumps(output, indent=2))

    def test_z_error_is_corrected_and_home_uses_existing_sdk_transit(self):
        # Construct a purely vertical feedback error with the SDK position IK.
        p = self.p; nominal = self.plan.goal
        target = p.tcp(nominal)
        out = p.ik.solve_ik(p.model, p.model.createData(), p.tcp_id,
            p.pin.SE3(np.eye(3), target+[0, 0, .006]), nominal+p.offsets,
            p.ik.IKParams(tolerance=1e-9), controlled_joints=6, position_only=True)
        self.assertTrue(out.success)
        s = CartesianTracker(p, nominal, target)
        for i in range(100): s.advance(nominal, target, out.q-p.offsets, .01*i, settling=True)
        self.assertLess(p.tcp(s.command)[2], target[2]-.0001)
        self.assertTrue(s.diagnostics()['z_feedback_enabled'])
        home = p.plan_home(nominal); c = self.controller(home, nominal)
        self.tick(c, 0., nominal)
        self.assertIsNone(c.cartesian_settler)
        self.assertFalse(home.mit_velocity_feedforward)

    def recorded_transport_failure(self):
        log = ROOT/'runs/click_hover/20261123_141006_1020e1fd.jsonl'
        events = [json.loads(line) for line in log.read_text().splitlines()]
        planner = HoverPlanner(load_config(overrides=events[0]['config']))
        click = next(row for row in events if row['event'] == 'click')
        done = next(row for row in events if row['event'] == 'completion')
        trace = np.load(log.with_name(log.stem+'_approach_7346607a.npz'))
        plan = planner.plan(np.array(click['raw_start_rad']), np.array(click['tcp_goal_base_m']))
        return planner, click, done, trace, plan

    def test_recorded_bound_saturated_bias_transports_without_false_stop(self):
        p, click, done, trace, plan = self.recorded_transport_failure()
        previous_nominal = trace['nominal_q_rad'][-1]
        previous_command = trace['command_q_rad'][-1]
        nominal = plan.sample(done['elapsed_s'])[0]
        carried_seed = nominal+previous_command-previous_nominal
        # The recorded third joint crossed below zero by 0.75 microradians.
        self.assertLess(carried_seed[2], p.raw_lower[2])
        tracker = CartesianTracker(p, previous_nominal, p.tcp(previous_nominal))
        tracker.command = previous_command.copy()
        expected = p.tcp(nominal)+p.tcp(previous_command)-p.tcp(previous_nominal)
        command = tracker.advance(nominal, plan.reference(done['elapsed_s']),
                                  np.array(done['raw_q_rad']), done['elapsed_s'])
        self.assertIsNone(tracker.stop_reason)
        p.validate_raw(command)
        np.testing.assert_array_equal(tracker.nominal, nominal)
        self.assertLess(np.linalg.norm(p.tcp(command)-expected), 1e-8)
        self.assertLess(np.max(np.abs(command-previous_command)), .0001)

    def test_recorded_startup_bias_completes_faster_approach_in_simulation(self):
        p, click, done, trace, plan = self.recorded_transport_failure()
        start = np.array(click['raw_start_rad'])
        # Reproduce the positive joint errors that saturated joint3 at startup.
        bias = np.array(done['raw_q_rad'])-np.array(done['raw_command_rad'])
        c = self.controller(plan, start, planner=p)
        actual = start.copy(); dt = .002
        for i in range(int((plan.duration+11)/dt)):
            now = i*dt
            measured = .0003835069006732131*np.round(actual/.0003835069006732131)
            self.tick(c, now, measured, dt)
            actual += (1-np.exp(-dt/.08))*(c.target+bias-actual)
            if not c.busy:
                break
        result = c.take_completion()
        self.assertIsNotNone(result)
        self.assertEqual(result['status'], 'SETTLED_MODEL_ONLY', result)
        self.assertLessEqual(result['fk_error_mm'], 1.)
        self.assertIsNone(result['cartesian_outer_loop']['stop_reason'])
        self.assertTrue(result['cartesian_outer_loop']['correction_stopped'])
        self.assertFalse(result['physical_accuracy_verified'])
        self.assertLess(plan.duration, 20.)

    def test_fixed_z_compensation_is_applied_once_before_sdk_planning(self):
        from eyehand.table_reference import table_hover_target, effective_hover_m
        p = self.p; cfg = p.cfg
        table_z = cfg['table_reference']['table_z_m']
        surface_base = np.r_[self.target[:2], table_z+.12]
        camera_point = (np.linalg.inv(p.T_base_camera)@np.r_[surface_base, 1.])[:3]
        surface, target = table_hover_target(p.T_base_camera, camera_point,
                                             effective_hover_m(cfg), table_z_m=table_z)
        np.testing.assert_allclose(surface, surface_base, atol=1e-12)
        self.assertAlmostEqual(target[2], table_z+.098, places=12)
        # Even a raised clicked surface does not replace the saved table Z.
        self.assertGreater(surface[2], target[2])
        plan = p.plan(self.start, target)
        self.assertAlmostEqual(plan.target[2], target[2], places=12)
        self.assertLess(abs(p.tcp(plan.goal)[2]-target[2]), .0002)
        self.assertTrue(plan.diagnostics['z_outer_feedback'])

    def test_cancel_fault_and_capture_refusal_freeze_or_disable(self):
        c = self.controller(); self.tick(c, 0., self.start)
        self.tick(c, .01, self.start)
        held = c.target.copy(); c.hold('USER_HOLD'); self.tick(c, .012, self.start)
        np.testing.assert_array_equal(c.target, held)
        self.assertIsNone(c.cartesian_settler)
        c = self.controller(); self.tick(c, 0., self.start)
        count = c.robot.arm.send_mit.call_count
        c.feedback.snapshot.side_effect = RuntimeError('stale feedback')
        with patch('eyehand.hover_control.time.perf_counter', return_value=.002): c._tick(c.robot, .002)
        self.assertIsNotNone(c.fault)
        self.assertEqual(c.robot.arm.send_mit.call_count, count)
        c.robot.estop.assert_called_once()
        c = self.controller(); self.tick(c, 0., self.start)
        held = c.target.copy(); disturbed = self.start.copy(); disturbed[0] += .05
        self.tick(c, .01, disturbed)
        result = c.take_completion()
        self.assertEqual(result['status'], 'TRACKING_REFUSED_HOLD')
        self.assertEqual(result['cartesian_outer_loop']['stop_reason'], 'capture_radius_exceeded')
        np.testing.assert_array_equal(c.target, held)

    def test_trace_files_preserve_reference_feedback_and_timing_columns(self):
        import tempfile
        from scripts.click_hover_eth import save_motion_traces
        rows = np.arange(102, dtype=float).reshape(3, 34)
        controller = SimpleNamespace(take_motion_traces=lambda: [('USER_HOLD', rows)])
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/'session.jsonl'
            save_motion_traces(controller, log)
            event = json.loads(log.read_text())
            self.assertEqual(event['event'], 'approach_trace')
            data = np.load(event['path'])
            np.testing.assert_array_equal(data['nominal_q_rad'], rows[:, 2:8])
            np.testing.assert_array_equal(data['command_q_rad'], rows[:, 8:14])
            np.testing.assert_array_equal(data['feedback_q_rad'], rows[:, 14:20])
            np.testing.assert_array_equal(data['reference_tcp_m'], rows[:, 20:23])
            np.testing.assert_array_equal(data['send_batch_duration_s'], rows[:, 33])


if __name__ == '__main__': unittest.main()

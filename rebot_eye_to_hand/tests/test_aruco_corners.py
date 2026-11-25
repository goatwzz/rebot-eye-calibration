"""PnP geometry, reset interlocks and RGB adapter; no camera or CAN access."""
from dataclasses import replace
from pathlib import Path
from contextlib import ExitStack
import tempfile
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from eyehand.aruco_corners import CornerDetector, CornerObservation, StableCorners, corner_goal, confirm_observation
from eyehand.corner_demo import RGBCamera, load_settings
from eyehand import corner_demo as app
from eyehand.corner_flow import CornerFlow
from eyehand.hover_runtime import DisplayFrame, RGBDFramePending

ROOT = Path(__file__).resolve().parents[1]


def settings(mount='external'):
    return load_settings(ROOT/'aruco_corners_config.json', mount, clearance_mm=2)


def observation(at=101., epoch=1):
    return CornerObservation(at, np.zeros(6), epoch,
        np.array([[.3, -.03, .12], [.36, -.03, .121], [.36, .03, .122], [.3, .03, .121]]),
        np.array([[100, 100], [200, 100], [200, 200], [100, 200]]), np.eye(4), .1, 1.)


class GeometryTests(unittest.TestCase):
    def test_known_size_pnp_recovers_xyz_and_canonical_order_for_both_mounts(self):
        k = np.array([[900., 0, 640.], [0, 900., 360.], [0, 0, 1.]])
        obj = np.array([[-.03, .03, 0], [.03, .03, 0], [.03, -.03, 0], [-.03, -.03, 0]])
        # Rotate in image too: canonical marker order must not be image-sorted.
        r = cv2.Rodrigues(np.array([2.7, .3, .7]))[0]
        t = np.array([.04, -.05, .55])
        uv = cv2.projectPoints(obj, cv2.Rodrigues(r)[0], t, k, np.zeros(5))[0].reshape(4, 2)
        for mount in ('external', 'wrist'):
            detector = CornerDetector(settings(mount), dict(K=k, D=np.zeros(5)))
            detector.target.detector = NS(detectMarkers=lambda gray: ([uv[None].astype(np.float32)], np.array([[0]]), []))
            base_camera = np.eye(4)
            base_camera[:3, :3] = cv2.Rodrigues(np.array([.2, -.4, .5]))[0]
            base_camera[:3, 3] = [.2, .1, .3]
            result = detector.detect(DisplayFrame(np.zeros((720, 1280, 3), np.uint8), None, 101.), base_camera, np.zeros(6), 1)
            expected = (obj @ r.T+t) @ base_camera[:3, :3].T+base_camera[:3, 3]
            np.testing.assert_allclose(result.corners_base_m, expected, atol=2e-6)
            np.testing.assert_allclose(result.corners_pixel, uv, atol=3e-5)
            np.testing.assert_allclose(np.linalg.norm(np.roll(result.corners_base_m, -1, axis=0)-result.corners_base_m, axis=1), .06)
            for n in range(1, 5):
                point, target = corner_goal(result, n, settings())
                np.testing.assert_allclose(point, expected[n-1], atol=2e-6)
                np.testing.assert_allclose(target-point, [0, 0, .002], atol=1e-12)
            self.assertFalse(result.event()['depth_used'])

    def test_real_detector_accepts_printed_60mm_marker(self):
        k = np.array([[900., 0, 640.], [0, 900., 360.], [0, 0, 1.]])
        detector = CornerDetector(settings(), dict(K=k, D=np.zeros(5)))
        obj = np.array([[-.03, .03, 0], [.03, .03, 0], [.03, -.03, 0], [-.03, -.03, 0]])
        rv, tv = np.array([2.7, .2, .1]), np.array([.04, -.03, .4])
        uv = cv2.projectPoints(obj, rv, tv, k, np.zeros(5))[0].reshape(4, 2)
        marker = cv2.aruco.generateImageMarker(detector.target.dictionary, 0, 400)
        homography = cv2.getPerspectiveTransform(np.float32([[0, 0], [399, 0], [399, 399], [0, 399]]), uv.astype(np.float32))
        image = cv2.warpPerspective(marker, homography, (1280, 720), borderValue=255)
        result = detector.detect(DisplayFrame(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None, 101.), np.eye(4))
        expected = obj @ cv2.Rodrigues(rv)[0].T+tv
        self.assertLess(np.max(np.linalg.norm(result.corners_base_m-expected, axis=1)), .003)
        detector.settings['marker_id'] = 1
        detector.target.config['id'] = 1
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            detector.detect(DisplayFrame(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None, 102.), np.eye(4))

    def test_stability_requires_new_frames_same_epoch_and_small_corner_span(self):
        window = StableCorners(3, .003)
        row = observation()
        self.assertIsNone(window.update(row))
        self.assertIsNone(window.update(row))
        self.assertEqual(len(window.rows), 1)
        self.assertIsNone(window.update(replace(row, acquired=102.)))
        self.assertIsNotNone(window.update(replace(row, acquired=103.)))
        self.assertIsNone(window.update(replace(row, acquired=104., epoch=2)))
        for t in (105., 106.):
            self.assertIsNone(window.update(replace(row, acquired=t, epoch=2,
                corners_base_m=row.corners_base_m+(.01 if t == 105. else 0))))

    def test_stale_or_moved_observation_cannot_start_a_motion(self):
        controller = NS(busy=False, motion_epoch=1, snapshot=lambda: np.zeros(7))
        row = observation()
        confirm_observation(row, controller, .3, .002, now=101.1)
        for changed in (replace(row, acquired=100.), replace(row, epoch=0),
                        replace(row, raw_q=np.ones(6)*.01), replace(row, raw_q=None)):
            with self.assertRaises(ValueError):
                confirm_observation(changed, controller, .3, .002, now=101.1)
        controller.busy = True
        with self.assertRaises(ValueError):
            confirm_observation(row, controller, .3, .002, now=101.1)


class FakeController:
    def __init__(self):
        self.motion_epoch, self.busy, self.complete = 0, False, None
        self.moves = []
    def hold(self, reason):
        self.motion_epoch += 1
        self.busy = False
        self.complete = dict(status=reason)
    def snapshot(self):
        return np.zeros(7)
    def planning_start(self):
        return np.zeros(6)
    def take_completion(self):
        result, self.complete = self.complete, None
        return result
    def move(self, plan, expected_epoch=None):
        if expected_epoch != self.motion_epoch:
            raise RuntimeError('Cancelled epoch')
        self.moves.append(plan.kind)
        self.busy = True
    def finish(self, status):
        self.busy, self.complete = False, dict(status=status)


class FlowTests(unittest.TestCase):
    def make_flow(self, mount):
        ctrl = FakeController()
        def plan(kind):
            return NS(kind=kind, target=np.array([.3, 0., .2]), goal=np.zeros(6), duration=1., diagnostics={})
        planner = NS(cfg=dict(camera_max_age_s=.3), plan_home=Mock(return_value=plan('home')),
                     plan_ready=Mock(return_value=plan('ready')))
        flow = CornerFlow(planner, ctrl, settings(mount), Mock(), clock=lambda: 101.)
        return ctrl, flow, plan('hover')

    def finish_plan(self, flow):
        flow.pump()
        flow.worker.thread.join(timeout=2.)
        flow.pump()

    def test_each_number_requires_completed_reset_and_new_observation(self):
        for mount in ('external', 'wrist'):
            ctrl, flow, hover = self.make_flow(mount)
            reset = 'home' if mount == 'external' else 'ready'
            for number in (1, 2, 3, 4):
                self.assertTrue(flow.reset(number))
                self.assertFalse(flow.reset(4))
                self.assertFalse(flow.observe(observation()))
                self.finish_plan(flow)
                self.assertEqual(ctrl.moves[-1], reset)
                self.assertFalse(flow.reset(4))
                ctrl.finish('HOME_REACHED_HOLD' if mount == 'external' else 'SETTLED_MODEL_ONLY')
                flow.pump()
                self.assertEqual(flow.phase, 'WAIT_MARKER')
                self.assertFalse(flow.observe(observation(at=100., epoch=ctrl.motion_epoch)))
                self.assertTrue(flow.observe(observation(epoch=ctrl.motion_epoch)))
                with patch('eyehand.corner_flow.plan_corner', return_value=hover):
                    self.finish_plan(flow)
                self.assertEqual(ctrl.moves[-1], 'hover')
                ctrl.finish('SETTLED_MODEL_ONLY'); flow.pump()
                self.assertEqual(flow.phase, 'CORNER_HOLD')
            self.assertEqual(ctrl.moves, [reset, 'hover']*4)

    def test_home_command_completion_allows_next_observation_and_exit(self):
        ctrl, flow, _ = self.make_flow('external')
        flow.reset(1); self.finish_plan(flow)
        ctrl.finish('HOME_COMMAND_FINISHED_HOLD'); flow.pump()
        self.assertEqual(flow.phase, 'WAIT_MARKER')
        self.assertTrue(flow.observe(observation(epoch=ctrl.motion_epoch)))
        flow.exit(); self.finish_plan(flow)
        ctrl.finish('HOME_COMMAND_FINISHED_HOLD'); flow.pump()
        self.assertTrue(flow.done)
        self.assertFalse(flow.home_confirmed)

    def test_failed_reset_or_hold_never_approaches(self):
        for status in ('HOME_TIMEOUT_HOLD', 'SETTLE_TIMEOUT_HOLD', 'UI_TIMEOUT_HOLD'):
            ctrl, flow, _ = self.make_flow('external')
            flow.reset(1); self.finish_plan(flow)
            ctrl.finish(status); flow.pump()
            self.assertFalse(flow.observe(observation(epoch=ctrl.motion_epoch)))
            self.assertEqual(ctrl.moves, ['home'])
            self.assertIsNone(flow.selected)
        ctrl, flow, _ = self.make_flow('wrist')
        flow.reset(1); flow.hold(); flow.pump()
        self.assertFalse(flow.observe(observation(epoch=ctrl.motion_epoch)))
        self.assertEqual(ctrl.moves, [])

    def test_cancelled_async_result_is_never_executed(self):
        ctrl, flow, _ = self.make_flow('wrist')
        flow.reset(1); flow.pump()
        flow.worker.thread.join(timeout=2.)
        flow.hold(); flow.pump()
        self.assertEqual(ctrl.moves, [])

    def test_exit_always_plans_raw_home_and_requires_confirmation(self):
        ctrl, flow, _ = self.make_flow('wrist')
        flow.exit(); self.finish_plan(flow)
        self.assertEqual(ctrl.moves, ['home'])
        self.assertFalse(flow.done)
        ctrl.finish('HOME_REACHED_HOLD'); flow.pump()
        self.assertTrue(flow.done and flow.home_confirmed)

    def test_camera_exit_discards_completed_corner_plan_before_sending(self):
        ctrl, flow, hover = self.make_flow('external')
        flow.reset(1); self.finish_plan(flow)
        ctrl.finish('HOME_REACHED_HOLD'); flow.pump()
        flow.observe(observation(epoch=ctrl.motion_epoch))
        with patch('eyehand.corner_flow.plan_corner', return_value=hover):
            flow.pump(); flow.worker.thread.join(timeout=2.)
        flow.exit('CAMERA_ERROR: USB disconnected')
        self.finish_plan(flow)
        self.assertEqual(ctrl.moves, ['home', 'home'])
        self.assertEqual(flow.phase, 'HOME_MOVING')


class CameraTests(unittest.TestCase):
    def test_rgb_only_existing_camera_no_depth_and_no_duplicate_freshness(self):
        cfg = dict(serial='test', camera_fps=15)
        cal = dict(camera=dict(serial='test'), image_size=[1280, 720], K=np.eye(3), D=np.zeros(5))
        cam = Mock(metadata=dict(serial='test'), width=1280, height=720, k=np.eye(3), d=np.zeros(5),
                   last_frame_metadata=dict(frame_number=1))
        cam.read.return_value = np.zeros((720, 1280, 3), np.uint8)
        with patch('eyehand.corner_demo.Camera', return_value=cam) as constructor:
            adapter = RGBCamera(cfg, cal)
            constructor.assert_called_once_with('realsense', serial='test', fps=15, intrinsics=None)
            adapter.warmup = 0
            image, depth = adapter.read()
            self.assertIsNone(depth)
            with self.assertRaises(RGBDFramePending):
                adapter.read()
            cam.read.side_effect = RuntimeError("Frame didn't arrive within 200")
            with self.assertRaises(RGBDFramePending):
                adapter.read()
            adapter.close(); cam.close.assert_called_once()


class DiagnosticRateTests(unittest.TestCase):
    def test_temporary_rate_override_preserves_age_and_motion_limits(self):
        for mount in ('external', 'wrist'):
            cfg = dict(control_rate_hz=500, feedback_max_age_s=.15, max_tcp_speed_m_s=.06)
            opts = dict(base_config='unused', home_position_tolerance_m=.009, home_tolerance_rad=.016)
            loader, planner = Mock(return_value=cfg), Mock(return_value='planner')
            with patch.object(sys, 'argv', ['demo', '--execute', '--control-rate-hz', '200']), \
                 patch.object(app, 'load_settings', return_value=opts), patch.object(app, 'run', return_value=0) as run:
                self.assertEqual(app.main(mount, 'unused', loader, planner), 0)
            self.assertEqual(cfg['control_rate_hz'], 200)
            self.assertEqual(cfg['feedback_max_age_s'], .15)
            self.assertEqual(cfg['max_tcp_speed_m_s'], .06)
            run.assert_called_once_with(cfg, 'planner', opts, True)

    def test_no_override_preserves_existing_rate(self):
        cfg = dict(control_rate_hz=500)
        opts = dict(base_config='unused', home_position_tolerance_m=.009, home_tolerance_rad=.016)
        with patch.object(sys, 'argv', ['demo']), patch.object(app, 'load_settings', return_value=opts), \
             patch.object(app, 'run', return_value=0):
            app.main('external', 'unused', Mock(return_value=cfg), Mock())
        self.assertEqual(cfg['control_rate_hz'], 500)


class UIInterlockTests(unittest.TestCase):
    def fake_run(self, folder, mount='external', *, execute=True, camera_error=False, hold_key=False,
                 close_window_error=False, startup_close=False, startup_x=False,
                 camera_lost=False, trace_error=False, late_startup_exit=False):
        config = dict(serial='test', camera_max_age_s=.3)
        opts = settings(mount); opts['log_dir'] = folder
        planner = NS(calibration={}, result_sha256='test', offsets=np.zeros(6))
        ctrl = Mock(busy=False)
        ctrl.feedback_checkpoints = []
        ctrl.failure_diagnostics = None
        ctrl.feedback_diagnostics.return_value = {}
        ctrl.snapshot.return_value = np.zeros(7)
        ctrl.take_motion_traces.return_value = []
        flow = Mock(done=False, exiting=False, home_confirmed=True, phase='WAIT_KEY')
        order = []
        flow.hold.side_effect = lambda: order.append('hold')
        flow.exit.side_effect = lambda *args: order.append('exit')
        def pump():
            order.append('pump')
            flow.done = True
        flow.pump.side_effect = pump
        gui = Mock(WINDOW_AUTOSIZE=1, WND_PROP_VISIBLE=1, FONT_HERSHEY_SIMPLEX=0, error=cv2.error)
        gui.waitKey.side_effect = ([ord('x')] if startup_x else
                                  [0, ord('h') if hold_key else 0 if camera_lost else ord('q')])
        gui.getWindowProperty.return_value = 1
        if close_window_error:
            gui.getWindowProperty.side_effect = [1, cv2.error('NULL guiReceiver')]
        elif startup_close:
            gui.getWindowProperty.side_effect = cv2.error('NULL guiReceiver')
        frame = DisplayFrame(np.zeros((720, 1280, 3), np.uint8), None, 101.)
        frames = Mock()
        frames.latest.return_value = frame
        if camera_lost:
            frames.latest.side_effect = [frame, RuntimeError('RGB camera lost')]
        if trace_error:
            flow.home_confirmed = False
        sys.path.insert(0, str(ROOT/'scripts'))
        import click_hover_eth as helpers
        with ExitStack() as stack:
            stack.enter_context(patch.object(app, 'cv2', gui))
            camera = stack.enter_context(patch.object(app, 'RGBCamera'))
            if camera_error:
                camera.side_effect = RuntimeError('camera failed')
            stack.enter_context(patch.object(app, 'LatestFrames', return_value=frames))
            constructor = stack.enter_context(patch.object(app, 'HoverController', return_value=ctrl))
            if late_startup_exit:
                ctrl.enabled = False
                ctrl.start.side_effect = lambda: constructor.call_args.kwargs['before_enable']()
            stack.enter_context(patch.object(app, 'CornerDetector'))
            stack.enter_context(patch.object(app, 'CornerFlow', return_value=flow))
            release = stack.enter_context(patch.object(app, 'release_supported'))
            release.side_effect = lambda *args: order.append('release')
            ctrl.close.side_effect = lambda: order.append('close')
            if trace_error:
                stack.enter_context(patch.object(helpers, 'save_motion_traces', side_effect=OSError('disk full')))
            code = app.run(config, planner, opts, execute)
        return code, constructor, ctrl, flow, order

    def test_stop_key_precedes_async_result_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            code, _, ctrl, flow, order = self.fake_run(folder, hold_key=True)
        self.assertEqual(code, 0)
        self.assertEqual(order, ['hold', 'pump', 'close'])
        ctrl.close.assert_called_once()

    def test_startup_wrist_resets_external_holds_and_preview_has_no_robot(self):
        for mount in ('wrist', 'external'):
            with tempfile.TemporaryDirectory() as folder:
                code, _, ctrl, flow, _ = self.fake_run(folder, mount)
            self.assertEqual(code, 0)
            ctrl.start.assert_called_once()
            self.assertEqual(flow.reset.call_count, int(mount == 'wrist'))
            flow.exit.assert_called_once()
        with tempfile.TemporaryDirectory() as folder:
            code, constructor, ctrl, _, _ = self.fake_run(folder, execute=False)
        self.assertEqual(code, 0)
        constructor.assert_not_called(); ctrl.start.assert_not_called()

    def test_camera_startup_failure_prevents_robot_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            code, constructor, ctrl, _, _ = self.fake_run(folder, camera_error=True)
        self.assertEqual(code, 1)
        constructor.assert_not_called(); ctrl.start.assert_not_called()

    def test_native_window_destroy_error_requests_normal_home_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            code, _, ctrl, flow, _ = self.fake_run(folder, close_window_error=True)
        self.assertEqual(code, 0)
        flow.exit.assert_called_once()
        flow.fail.assert_not_called()

    def test_startup_window_close_and_x_never_enable_or_return_ready(self):
        for mode in ('startup_close', 'startup_x'):
            with tempfile.TemporaryDirectory() as folder:
                code, constructor, ctrl, flow, _ = self.fake_run(folder, 'wrist', **{mode: True})
            self.assertEqual(code, 0)
            constructor.assert_not_called(); ctrl.start.assert_not_called(); flow.reset.assert_not_called()

    def test_camera_failure_is_handled_before_a_finished_plan_is_pumped(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, _, flow, order = self.fake_run(folder, camera_lost=True)
        self.assertEqual(order[:2], ['exit', 'pump'])
        self.assertIn('CAMERA_ERROR:', flow.exit.call_args.args[0])

    def test_trace_write_failure_cannot_skip_supported_release(self):
        with tempfile.TemporaryDirectory() as folder:
            code, _, ctrl, flow, order = self.fake_run(folder, trace_error=True)
        self.assertEqual(code, 1)
        self.assertEqual(order[-2:], ['release', 'close'])
        ctrl.close.assert_called_once()

    def test_exit_queued_during_sdk_setup_is_checked_before_motor_enable(self):
        for kwargs in ({}, dict(hold_key=True), dict(close_window_error=True)):
            with tempfile.TemporaryDirectory() as folder:
                code, constructor, ctrl, flow, _ = self.fake_run(folder, 'wrist', late_startup_exit=True, **kwargs)
            self.assertEqual(code, 0)
            self.assertIn('before_enable', constructor.call_args.kwargs)
            flow.reset.assert_not_called(); flow.pump.assert_not_called(); flow.fail.assert_not_called()
            ctrl.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()

"""SDK target producer isolation, stale-result handling and real-model replay; no I/O."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concurrent.futures import Future, ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from eyehand.click_hover import HoverPlanner, load_config
from eyehand.cartesian_settle import CartesianTracker
from eyehand.hover_control import HoverController
from eyehand.target_worker import BufferedCartesianTracker


class DeferredExecutor:
    def __init__(self):
        self.jobs = []
    def submit(self, fn, *args):
        result = Future()
        self.jobs.append((result, fn, args))
        return result
    def finish(self):
        result, fn, args = self.jobs.pop(0)
        if result.set_running_or_notify_cancel():
            try:
                result.set_result(fn(*args))
            except Exception as exc:
                result.set_exception(exc)


class TargetWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = HoverPlanner(load_config())
        cls.q = np.array([0., 1., 1., -.8, 0., 0.])
        cls.ref = cls.p.tcp(cls.q)

    def test_unchanged_sdk_tracker_result_and_no_blocking_wait(self):
        p, q, ref = self.p, self.q, self.ref
        executor = DeferredExecutor()
        worker = BufferedCartesianTracker(p, q, ref, executor)
        original = CartesianTracker(p, q, ref)
        raw = q+np.array([0., .003, -.004, .002, 0., 0.])
        previous = q.copy()
        for i in range(20):
            now = .012*i
            expected = original.advance(q, ref, raw, now, settling=True)
            # Pending work cannot block the sender or prematurely change its target.
            np.testing.assert_allclose(worker.advance(q, ref, raw, now, settling=True), previous)
            self.assertFalse(worker.future.done())
            executor.finish()
            np.testing.assert_allclose(worker.advance(q, ref, raw, now+.002, settling=True), expected,
                                       atol=1e-12, rtol=0)
            previous = expected
        self.assertEqual(worker.accepted, 20)
        self.assertLessEqual(worker.diagnostics()['peak_target_result_age_ms'], 2.01)

    def test_timeout_never_applies_late_result_or_continues_submitting(self):
        executor = DeferredExecutor()
        worker = BufferedCartesianTracker(self.p, self.q, self.ref, executor)
        worker.advance(self.q, self.ref, self.q, 0.)
        future = worker.future
        np.testing.assert_array_equal(worker.advance(self.q+.001, self.ref, self.q, .051), self.q)
        self.assertEqual(worker.stop_reason, 'target_worker_timeout')
        self.assertTrue(future.cancelled())
        executor.finish()
        self.assertEqual(worker.accepted, 0)
        self.assertEqual(len(executor.jobs), 0)

    def test_cancelled_running_result_cannot_resume_a_cancelled_motion(self):
        executor = DeferredExecutor()
        worker = BufferedCartesianTracker(self.p, self.q, self.ref, executor)
        worker.advance(self.q, self.ref, self.q, 0.)
        future = worker.future
        self.assertTrue(future.set_running_or_notify_cancel())
        worker.cancel()
        future.set_result({'invalid': 'late result must never be read'})
        np.testing.assert_array_equal(worker.advance(self.q+.001, self.ref, self.q, .002), self.q)
        self.assertEqual(worker.accepted, 0)
        self.assertEqual(worker.stop_reason, 'target_worker_cancelled')

    def test_solver_exception_is_propagated_to_existing_control_fault_path(self):
        executor = DeferredExecutor()
        def failed(*args):
            raise ValueError('SDK IK failure')
        worker = BufferedCartesianTracker(self.p, self.q, self.ref, executor, factory=failed)
        worker.advance(self.q, self.ref, self.q, 0.)
        executor.finish()
        with self.assertRaisesRegex(ValueError, 'SDK IK failure'):
            worker.advance(self.q, self.ref, self.q, .002)

    def test_controller_cancel_invalidates_pending_target(self):
        c = HoverController(self.p, NS(arm=Mock(), gripper=Mock()))
        executor = DeferredExecutor()
        worker = BufferedCartesianTracker(self.p, self.q, self.ref, executor)
        worker.advance(self.q, self.ref, self.q, 0.)
        c.cartesian_settler = worker
        c.target = self.q.copy()
        c.hold()
        self.assertTrue(worker.closed)
        self.assertTrue(worker.future.cancelled())
        np.testing.assert_array_equal(c.target, self.q)
        self.assertIsNone(c.plan)

    def test_settle_comparison_keeps_the_sdk_path_and_defers_xyz_correction(self):
        import copy
        p = copy.copy(self.p)
        p.cfg = {**p.cfg, 'cartesian_outer_loop': {**p.cfg['cartesian_outer_loop'], 'during_motion': False}}
        plan = p.plan(self.q, self.ref+[.002, .001, -.002])
        robot = NS(arm=Mock(), gripper=Mock())
        c = HoverController(p, robot)
        c.target, c.gripper, c.enabled = self.q.copy(), np.array([.25]), True
        c.feedback = Mock()
        c.feedback.snapshot.return_value = NS(positions=np.r_[plan.sample(.5)[0], .25])
        c.plan, c.started, c.stable_since, c.ui_seen = plan, 0., None, .5
        c.target_executor = DeferredExecutor()
        with patch('eyehand.hover_control.time.perf_counter', return_value=.5):
            c._tick(robot, .002)
        self.assertIsNone(c.fault)
        self.assertIsNone(c.cartesian_settler)
        np.testing.assert_allclose(c.target, plan.sample(.5)[0])
        self.assertEqual(len(c.trace_samples), 1)
        now = plan.duration+.002
        c._last_tick, c.ui_seen = now-.002, now
        c.feedback.snapshot.return_value.positions[:6] = plan.goal
        with patch('eyehand.hover_control.time.perf_counter', return_value=now):
            c._tick(robot, .002)
        self.assertIsNone(c.fault)
        self.assertIsInstance(c.cartesian_settler, BufferedCartesianTracker)
        self.assertEqual(c.cartesian_settler.axes, 'xyz')
        c.hold()

    def test_real_worker_converges_with_one_sender_and_quantized_feedback(self):
        p, q = self.p, self.q
        plan = p.plan(q, p.tcp(q)+[.002, .001, -.002])
        robot = NS(arm=Mock(), gripper=Mock())
        c = HoverController(p, robot)
        c.enabled, c.target, c.gripper = True, q.copy(), np.array([.25])
        c.feedback = Mock()
        actual = q.copy()
        c.feedback.snapshot.return_value = NS(positions=np.r_[actual, .25])
        c.plan, c.started, c.stable_since = plan, time.perf_counter(), None
        bias = np.array([0., .003, -.004, .002, -.002, .001])
        quantum = .0003835069006732131
        sender_threads = set()
        robot.arm.send_mit.side_effect = lambda *args, **kw: sender_threads.add(threading.get_ident())
        with ThreadPoolExecutor(max_workers=1) as executor:
            c.target_executor = executor
            last = time.perf_counter()
            while time.perf_counter()-c.started < plan.duration+10.:
                now = time.perf_counter()
                actual += (1-np.exp(-(now-last)/.08))*(c.target-bias-actual)
                last = now
                measured = quantum*np.round(actual/quantum)
                c.feedback.snapshot.return_value.positions = np.r_[measured, .25]
                c.ui_seen = now
                c._tick(robot, .002)
                self.assertIsNone(c.fault, c.fault)
                if c.plan is None:
                    break
                time.sleep(.002)
            self.assertEqual(c.completion['status'], 'SETTLED_MODEL_ONLY', c.completion)
            self.assertLessEqual(c.completion['fk_error_mm'], 1.)
            d = c.completion['cartesian_outer_loop']
            self.assertEqual(d['target_producer'], 'separate_single_worker_no_motor_io')
            self.assertGreater(d['accepted_targets'], 20)
            self.assertEqual(sender_threads, {threading.get_ident()})
            np.testing.assert_array_equal(robot.arm.send_mit.call_args.kwargs['vel'], np.zeros(6))
            held = c.target.copy()
            c.ui_seen = time.perf_counter()
            c._tick(robot, .002)
            np.testing.assert_array_equal(c.target, held)
        c.target_executor = None


if __name__ == '__main__':
    unittest.main()

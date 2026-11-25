from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from eyehand.hover_runtime import ClickGate, DisplayFrame, LatestFrames, PlanWorker, release_supported, RGBDFramePending
import test_click_hover as fixtures


class RuntimeTests(unittest.TestCase):
    def test_click_during_motion_is_discarded_not_queued(self):
        moving=True
        gate=ClickGate(lambda:moving)
        self.assertFalse(gate.offer(1,2,object()))
        moving=False
        self.assertIsNone(gate.take())
        frame=object()
        self.assertTrue(gate.offer(1,2,frame))
        self.assertFalse(gate.offer(3,4,frame))
        self.assertEqual(gate.take(),(1,2,frame))

    def test_cancelled_worker_cannot_launch_later_or_share_pinocchio_data(self):
        worker=PlanWorker()
        entered,finish=threading.Event(),threading.Event()
        def job(cancelled):
            entered.set()
            finish.wait(1.)
            return 'old-plan'
        worker.submit(job,{'pixel':[1,2]})
        self.assertTrue(entered.wait(1.))
        worker.cancel()
        with self.assertRaises(RuntimeError):worker.submit(lambda c:'new-plan',{})
        finish.set();worker.thread.join(1.)
        self.assertIsNone(worker.poll())
        worker.submit(lambda c:'new-plan',{'pixel':[3,4]})
        worker.thread.join(1.)
        self.assertEqual(worker.poll()[1],'new-plan')

    def test_camera_worker_failure_and_stale_frame_are_visible(self):
        camera=Mock()
        camera.read.side_effect=RuntimeError('USB unplugged')
        frames=LatestFrames(camera);frames.start();frames.thread.join(1.)
        with self.assertRaisesRegex(RuntimeError,'USB unplugged'):frames.latest(.3)
        frames.close();camera.close.assert_called_once()
        frames=LatestFrames(Mock())
        frames.frame=DisplayFrame(None,None,time.monotonic()-1.)
        with self.assertRaisesRegex(RuntimeError,'expired'):frames.latest(.3)

    def test_transient_rgbd_errors_do_not_kill_reader_or_refresh_frame_age(self):
        stop=threading.Event()
        class Camera:
            calls=0
            def read(self):
                self.calls+=1
                if self.calls==1:raise RGBDFramePending('skew=235 ms')
                if self.calls==2:return 'rgb','depth'
                stop.wait(.01)
                raise RGBDFramePending('waiting for next new pair')
            def close(self):stop.set()
        camera=Camera();frames=LatestFrames(camera);frames.start()
        deadline=time.monotonic()+1.
        while frames.frame is None and time.monotonic()<deadline:time.sleep(.001)
        self.assertIsNone(frames.error)
        frame=frames.latest(.3)
        self.assertEqual(frame.color,'rgb')
        acquired=frame.acquired
        # A retry never republishes cached pixels with a new acquisition time.
        time.sleep(.02)
        self.assertEqual(frames.frame.acquired,acquired)
        with patch('eyehand.hover_runtime.time.monotonic',return_value=acquired+1):
            with self.assertRaisesRegex(RuntimeError,'expired'):frames.latest(.3)
        frames.close()

    def test_release_wait_detects_control_fault_without_enter_key(self):
        controller=SimpleNamespace(enabled=True,fault=None,heartbeat=Mock(side_effect=RuntimeError('control loop died')))
        with self.assertRaisesRegex(RuntimeError,'died'):
            release_supported(controller,input_fn=lambda _: 'release')

    def test_release_eof_returns_to_shutdown(self):
        controller=SimpleNamespace(enabled=True,fault=None,heartbeat=Mock())
        def eof(_):raise EOFError
        release_supported(controller,input_fn=eof)
        controller.heartbeat.assert_called()


class ControllerReviewTests(unittest.TestCase):
    def make_controller(self):
        return fixtures.ControlFaultTests().make_controller()

    def test_nan_gripper_feedback_never_reaches_motor_send(self):
        ctrl,robot=self.make_controller()
        ctrl.feedback.snapshot.return_value.positions[6]=float('nan')
        ctrl._tick(robot,.01)
        robot.arm.send_mit.assert_not_called()
        robot.gripper.send_mit.assert_not_called()
        self.assertIn('finite',ctrl.fault)

    def test_ui_timeout_cancels_active_motion_and_old_click_epoch(self):
        ctrl,robot=self.make_controller()
        original=ctrl.target.copy()
        epoch=ctrl.motion_epoch
        ctrl.plan=SimpleNamespace(sample=Mock())
        ctrl.ui_seen=time.perf_counter()-1.
        ctrl._tick(robot,.01)
        self.assertIsNone(ctrl.plan)
        self.assertEqual(ctrl.take_completion()['status'],'UI_TIMEOUT_HOLD')
        np.testing.assert_allclose(robot.arm.send_mit.call_args.args[0],original)
        ctrl.heartbeat()
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            ctrl.move(SimpleNamespace(curve=lambda x:original),expected_epoch=epoch)

    def test_hold_invalidates_already_completed_ik_result(self):
        ctrl,_=self.make_controller()
        epoch=ctrl.motion_epoch
        ctrl.hold()
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            ctrl.move(SimpleNamespace(curve=lambda x:ctrl.target),expected_epoch=epoch)

    def test_disable_attempts_every_motor_after_one_failure(self):
        ctrl,robot=self.make_controller()
        robot._motor_map={str(i):Mock() for i in range(7)}
        robot._motor_map['0'].disable.side_effect=RuntimeError('CAN failure')
        ctrl._disable_motors()
        for motor in robot._motor_map.values():motor.disable.assert_called_once()
        self.assertEqual(len(ctrl.disable_errors),1)

    def test_close_closes_bus_even_if_sdk_disconnect_raises(self):
        ctrl,robot=self.make_controller()
        robot._ctrl_map={'robstride':Mock()}
        robot.disconnect.side_effect=RuntimeError('disconnect failure')
        with self.assertRaisesRegex(RuntimeError,'disconnect failure'):ctrl.close()
        robot._ctrl_map['robstride'].close.assert_called_once()

    def test_dead_sdk_thread_is_detected_during_hold(self):
        ctrl,robot=self.make_controller()
        robot._ctrl_thread=threading.Thread(target=lambda:None)
        with self.assertRaisesRegex(RuntimeError,'stopped unexpectedly'):ctrl.heartbeat()
        robot.estop.assert_called_once()

    def test_broken_stdout_cannot_skip_fault_disable(self):
        ctrl,robot=self.make_controller()
        ctrl.feedback.snapshot.side_effect=RuntimeError('stale')
        with patch('builtins.print',side_effect=BrokenPipeError('closed stdout')):
            ctrl._tick(robot,.01)
        robot.estop.assert_called_once()
        self.assertEqual(ctrl.fault,'stale')

    def test_completion_is_consumed_once(self):
        ctrl,_=self.make_controller()
        ctrl.completion={'status':'done'}
        self.assertEqual(ctrl.take_completion(),{'status':'done'})
        self.assertIsNone(ctrl.take_completion())


if __name__=='__main__':unittest.main()

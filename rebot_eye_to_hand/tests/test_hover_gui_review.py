from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from eyehand.click_hover import load_config, ROOT
from eyehand.hover_runtime import DisplayFrame, RGBDFramePending
from scripts import click_hover_eth as app
import eyehand.hover_control as control_module


class GUIReviewTests(unittest.TestCase):
    def run_fake(self, directory, *, frame_failure=False, camera_close_failure=False,
                 release_failure=False, camera_start_failure=False, exit_key=ord('x'), key_sequence=None):
        cfg=load_config();cfg['log_dir']=directory
        frame=DisplayFrame(np.zeros((720,1280,3),dtype=np.uint8),np.ones((720,1280)),time.monotonic())
        frames=Mock()
        frames.latest.side_effect=[frame,RuntimeError('USB lost')] if frame_failure else None
        frames.latest.return_value=frame
        if camera_close_failure:frames.close.side_effect=RuntimeError('camera stop failed')
        planner=SimpleNamespace(calibration={},result_sha256='offline-test',cfg=cfg)
        controller=Mock()
        controller.busy=False;controller.fault=None;controller.enabled=True
        controller.motion_kind=None;controller.motion_epoch=1
        controller.snapshot.return_value=np.r_[np.zeros(6),.25]
        controller.planning_start.return_value=np.zeros(6)
        controller.take_completion.return_value=None
        controller.take_motion_traces.return_value=[]
        gui=Mock(EVENT_LBUTTONDOWN=1,WINDOW_AUTOSIZE=1,WND_PROP_VISIBLE=1,FONT_HERSHEY_SIMPLEX=1,MARKER_CROSS=1)
        gui.waitKey.side_effect=key_sequence or [0,exit_key,ord('x')]
        gui.getWindowProperty.return_value=1
        worker=Mock(active=False)
        worker.poll.return_value=None
        worker.results.empty.return_value=True
        with ExitStack() as stack:
            stack.enter_context(patch.object(app,'cv2',gui))
            stack.enter_context(patch.object(app,'load_config',return_value=cfg))
            stack.enter_context(patch.object(app,'HoverPlanner',return_value=planner))
            camera_constructor=stack.enter_context(patch.object(app,'AlignedRGBD'))
            if camera_start_failure:camera_constructor.side_effect=RuntimeError('camera startup failed')
            stack.enter_context(patch.object(app,'LatestFrames',return_value=frames))
            stack.enter_context(patch.object(app,'PlanWorker',return_value=worker))
            robot_constructor=stack.enter_context(patch.object(control_module,'HoverController',return_value=controller))
            release=stack.enter_context(patch.object(app,'release_supported'))
            if release_failure:release.side_effect=OSError('terminal failed')
            stack.enter_context(patch('sys.argv',['click_hover_eth.py','--execute']))
            if frame_failure:
                with self.assertRaisesRegex(RuntimeError,'USB lost'):app.main()
            elif camera_start_failure:
                with self.assertRaisesRegex(RuntimeError,'camera startup failed'):app.main()
            elif release_failure:
                with self.assertRaisesRegex(OSError,'terminal failed'):app.main()
            else:
                self.assertEqual(app.main(),1 if camera_close_failure else 0)
        return controller,frames,worker,robot_constructor

    def test_startup_only_holds_and_does_not_request_observation_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            ctrl,_,worker,_=self.run_fake(directory,exit_key=ord('x'))
        ctrl.start.assert_called_once()
        ctrl.move.assert_not_called()
        worker.submit.assert_not_called()

    def test_camera_failure_before_start_cannot_construct_robot(self):
        with tempfile.TemporaryDirectory() as directory:
            ctrl,frames,worker,constructor=self.run_fake(directory,camera_start_failure=True)
        constructor.assert_not_called();ctrl.start.assert_not_called()

    def test_usb_and_camera_cleanup_failure_still_hold_and_close_robot(self):
        with tempfile.TemporaryDirectory() as directory:
            ctrl,frames,worker,_=self.run_fake(directory,frame_failure=True,camera_close_failure=True)
        ctrl.hold.assert_called();ctrl.close.assert_called_once();worker.cancel.assert_called()

    def test_release_terminal_exception_still_closes_robot(self):
        with tempfile.TemporaryDirectory() as directory:
            ctrl,frames,worker,_=self.run_fake(directory,release_failure=True)
        ctrl.close.assert_called_once();frames.close.assert_called_once()

    def test_stop_key_precedes_completed_plan_and_cancels_job(self):
        with tempfile.TemporaryDirectory() as directory:
            ctrl,_,worker,_=self.run_fake(directory,exit_key=ord('h'))
        ctrl.move.assert_not_called()
        ctrl.hold.assert_called()
        worker.poll.assert_not_called()
        self.assertGreaterEqual(worker.cancel.call_count,2)

    def test_q_and_escape_schedule_home_instead_of_exit(self):
        for key in (ord('q'),27):
            with tempfile.TemporaryDirectory() as directory:
                ctrl,_,worker,_=self.run_fake(directory,key_sequence=[0,key,0,ord('x')])
            worker.submit.assert_called_once()
            self.assertEqual(worker.submit.call_args.args[1]['kind'],'home')
            self.assertTrue(any(call.args==('HOME_REQUESTED_HOLD',) for call in ctrl.hold.call_args_list))
            ctrl.move.assert_not_called()  # Fake worker has not returned a validated plan.

    def test_h_cancels_pending_home_request(self):
        with tempfile.TemporaryDirectory() as directory:
            ctrl,_,worker,_=self.run_fake(directory,key_sequence=[0,ord('q'),ord('h'),ord('x')])
        worker.submit.assert_not_called()
        ctrl.move.assert_not_called()



class RGBDReviewTests(unittest.TestCase):
    def fake_camera(self):
        camera=object.__new__(app.AlignedRGBD)
        camera.pipeline=Mock()
        camera.align=Mock()
        color,depth=Mock(),Mock()
        color.get_timestamp.return_value=100.;depth.get_timestamp.return_value=101.
        color.get_frame_number.return_value=1;depth.get_frame_number.return_value=1
        color.get_frame_timestamp_domain.return_value='hardware';depth.get_frame_timestamp_domain.return_value='hardware'
        color.get_data.return_value=np.zeros((720,1280,3),np.uint8)
        depth.get_data.return_value=np.full((720,1280),500,np.uint16)
        frames=Mock()
        frames.get_color_frame.return_value=color;frames.get_depth_frame.return_value=depth
        camera.align.process.return_value=frames
        camera.pipeline.wait_for_frames.return_value=frames
        camera.scale=.001;camera.last_numbers=None;camera.last_timestamps=None
        return camera,color,depth

    def test_aligned_depth_scale_and_frozen_frame_detection(self):
        camera,_,_=self.fake_camera()
        rgb,depth=camera.read()
        np.testing.assert_allclose(depth,.5)
        with self.assertRaisesRegex(RuntimeError,'Repeated'):camera.read()

    def test_timestamp_domain_skew_and_nan_fail(self):
        for kind in ('domain','skew','nan'):
            camera,color,depth=self.fake_camera()
            if kind=='domain':depth.get_frame_timestamp_domain.return_value='system'
            if kind=='skew':depth.get_timestamp.return_value=150.
            if kind=='nan':depth.get_timestamp.return_value=float('nan')
            with self.assertRaises(RuntimeError):camera.read()

    def test_bad_startup_pair_is_rejected_then_next_good_pair_recovers(self):
        camera,color,depth=self.fake_camera()
        depth.get_timestamp.return_value=335.
        with self.assertRaisesRegex(RGBDFramePending,'235.0 ms'):camera.read()
        self.assertIsNone(camera.last_numbers)
        depth.get_timestamp.return_value=100.024
        color.get_frame_number.return_value=3;depth.get_frame_number.return_value=1
        rgb,d=camera.read()
        self.assertEqual(rgb.shape,(720,1280,3))
        self.assertEqual(camera.last_numbers,(3,1))

    def test_sdk_frame_timeout_is_retriable_but_unexpected_error_is_fatal(self):
        camera,_,_=self.fake_camera()
        camera.pipeline.wait_for_frames.side_effect=RuntimeError("Frame didn't arrive within 200")
        with self.assertRaises(RGBDFramePending):camera.read()
        camera.pipeline.wait_for_frames.side_effect=RuntimeError('USB backend disconnected')
        with self.assertRaisesRegex(RuntimeError,'USB backend disconnected'):camera.read()

    def test_warmup_drops_valid_pairs_without_treating_them_as_failure(self):
        camera,color,depth=self.fake_camera();camera.warmup_remaining=2
        for number in (1,2):
            color.get_frame_number.return_value=number;depth.get_frame_number.return_value=number
            color.get_timestamp.return_value=number*33.;depth.get_timestamp.return_value=number*33.+.1
            with self.assertRaisesRegex(RGBDFramePending,'warmup'):camera.read()
        color.get_frame_number.return_value=3;depth.get_frame_number.return_value=3
        color.get_timestamp.return_value=99.;depth.get_timestamp.return_value=99.1
        self.assertEqual(camera.read()[1].shape,(720,1280))

    def test_rgb_intrinsics_are_taken_from_selected_result(self):
        calibration=json.loads((ROOT.parent/'examples/legacy_click/result.json').read_text())['provenance']
        cfg=load_config()
        rs=Mock()
        pipeline=rs.pipeline.return_value
        profile=pipeline.start.return_value
        intrinsic=SimpleNamespace(fx=999.,fy=909.,ppx=656.,ppy=380.,width=1280,height=720,coeffs=[0.]*5)
        profile.get_stream.return_value.as_video_stream_profile.return_value.get_intrinsics.return_value=intrinsic
        profile.get_device.return_value.get_info.return_value=cfg['serial']
        with patch.dict('sys.modules',{'pyrealsense2':rs}):
            with self.assertRaisesRegex(ValueError,'intrinsics differ'):app.AlignedRGBD(cfg,calibration)
        pipeline.stop.assert_called_once()


if __name__=='__main__':unittest.main()

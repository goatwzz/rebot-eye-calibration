"""Table height policy, persistence, and real GUI orchestration with mock I/O."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch
import numpy as np
from eyehand.click_hover import load_config
from eyehand.hover_runtime import DisplayFrame
from eyehand.table_reference import (table_hover_target, save_table_reference,
    matching_table_reference, validate_table_settings, effective_hover_m)
from scripts import click_hover_eth as app
import eyehand.hover_control as control_module


class TableGeometryTests(unittest.TestCase):
    def test_raised_surfaces_keep_same_table_relative_target_height(self):
        for z in (0., .04, .08):
            surface,target=table_hover_target(np.eye(4),[.3,.1,max(z,.001)],.088,table_z_m=0.)
            np.testing.assert_array_equal(surface[:2],target[:2])
            self.assertAlmostEqual(target[2],.088)
            self.assertAlmostEqual(target[2]-surface[2],.088-max(z,.001))
        for z in (.088,.1):
            surface,target=table_hover_target(np.eye(4),[.3,.1,z],.088,table_z_m=0.)
            np.testing.assert_array_equal(target[:2],surface[:2])
            self.assertAlmostEqual(target[2],.088)

    def test_signed_compensation_lowers_only_target_and_preserves_surface(self):
        for offset,expected in ((-.01,.078),(-.005,.083),(0.,.088)):
            cfg={'hover_m':.088,'height_compensation_m':offset}
            height=effective_hover_m(cfg)
            self.assertAlmostEqual(height,expected)
            surface,target=table_hover_target(np.eye(4),[.3,.1,.04],height,table_z_m=0.)
            np.testing.assert_array_equal(surface,[.3,.1,.04])
            np.testing.assert_array_equal(target[:2],surface[:2])
            self.assertAlmostEqual(target[2],expected)
        for offset in (float('nan'),float('inf'),-.088,-.1):
            with self.assertRaises(ValueError):effective_hover_m({'hover_m':.088,'height_compensation_m':offset})
        # A surface above the compensated plane still only determines XY.
        surface,target=table_hover_target(np.eye(4),[.3,.1,.08],effective_hover_m({'hover_m':.088,'height_compensation_m':-.01}),table_z_m=0.)
        self.assertAlmostEqual(target[2],.078)
        self.assertGreater(surface[2],target[2])

    def test_table_height_is_applied_after_rotated_camera_transform(self):
        T=np.array([[0,0,1,.1],[1,0,0,.2],[0,1,0,-.1],[0,0,0,1.]])
        surface,target=table_hover_target(T,[.1,.12,.5],.088,table_z_m=.01)
        np.testing.assert_allclose(surface,[.6,.3,.02])
        np.testing.assert_allclose(target,[.6,.3,.098])
        for z in (None,float('nan'),float('inf')):
            with self.assertRaises(ValueError):table_hover_target(T,[.1,.12,.5],.088,table_z_m=z)

    def test_reference_identity_height_validation_and_persistent_save(self):
        cfg=load_config();cfg['table_reference']=dict(table_z_m=0.,camera_serial=cfg['serial'],calibration_sha256='A')
        validate_table_settings(cfg)
        self.assertIsNotNone(matching_table_reference(cfg,'A'))
        self.assertIsNone(matching_table_reference(cfg,'B'))
        cfg['table_reference']['camera_serial']='other'
        self.assertIsNone(matching_table_reference(cfg,'A'))
        for z in (float('nan'),1.,None):
            cfg['table_reference']['table_z_m']=z
            with self.assertRaises(ValueError):validate_table_settings(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json';raw={'hover_m':.088,'custom':'preserve'};path.write_text(json.dumps(raw))
            cfg={'_path':str(path)};ref={'table_z_m':0.,'table_reference_id':'saved'}
            save_table_reference(cfg,ref)
            self.assertEqual(json.loads(path.read_text()),{**raw,'table_reference':ref})
            self.assertEqual(cfg['table_reference'],ref)
            with patch.object(Path,'replace',side_effect=OSError('disk blocked')):
                with self.assertRaisesRegex(RuntimeError,'not saved'):save_table_reference(cfg,{'table_z_m':.01})
            self.assertEqual(cfg['table_reference'],ref)
            self.assertEqual(list(Path(tmp).iterdir()),[path])


class TableGUITests(unittest.TestCase):
    def run_case(self, *, reused=False, reselect=False, above=False, mismatch=False, compensation=0.):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=load_config();cfg['hover_m']=.088;cfg['height_compensation_m']=compensation;cfg['log_dir']=tmp;cfg['_path']=str(Path(tmp)/'config.json')
            cfg['table_reference']=None
            if reused or reselect or mismatch:
                cfg['table_reference']=dict(table_z_m=.5,table_reference_id='previous',camera_serial=cfg['serial'],
                                            calibration_sha256='old' if mismatch else 'test')
            Path(cfg['_path']).write_text(json.dumps(cfg))
            plan=SimpleNamespace(goal=np.zeros(6),duration=.1,path_mode='joint')
            K=[[500.,0,640.],[0,500.,360.],[0,0,1.]]
            planner=SimpleNamespace(cfg=cfg,result_sha256='test',T_base_camera=np.eye(4),
                result={'T_camera_base':np.eye(4)},calibration={'K':K,'D':[0.]*5},plan=Mock(return_value=plan))
            controller=Mock(busy=False,fault=None,enabled=True,motion_kind=None,motion_epoch=0)
            controller.snapshot.return_value=np.r_[np.zeros(6),.25]
            controller.planning_start.return_value=np.zeros(6);controller.take_completion.return_value=None
            controller.take_motion_traces.return_value=[]
            frame=DisplayFrame(np.zeros((720,1280,3),np.uint8),np.full((720,1280),.5),time.monotonic())
            frames=Mock();frames.latest.return_value=frame
            class Worker:
                active=False
                outcome=None
                def __init__(self):self.results=SimpleNamespace(empty=lambda:self.outcome is None)
                def submit(self,fn,context):self.outcome=(context,fn(lambda:False),None)
                def poll(self):value,self.outcome=self.outcome,None;return value
                def cancel(self):self.outcome=None
            worker=Worker();camera=Mock();events=[];callbacks={}
            if reselect:actions=iter(['t','click','click','idle','x']);heights=iter([.51,.54]);expected=[.598]
            elif reused:actions=iter(['click','idle','click','idle','x']);heights=iter([.54,.57]);expected=[.588+compensation,.588+compensation]
            elif above:actions=iter(['click','click','idle','x']);heights=iter([.5,.60]);expected=[.588+compensation]
            else:actions=iter(['click','click','idle','click','idle','x']);heights=iter([.5,.54,.57]);expected=[.588+compensation,.588+compensation]
            camera.point.side_effect=lambda *a:np.array([.3,0.,next(heights)])
            gui=Mock(EVENT_LBUTTONDOWN=1,WINDOW_AUTOSIZE=1,WND_PROP_VISIBLE=1,FONT_HERSHEY_SIMPLEX=1)
            gui.Rodrigues.side_effect=app.cv2.Rodrigues
            gui.projectPoints.side_effect=app.cv2.projectPoints
            gui.getWindowProperty.return_value=1
            def waitkey(*a):
                if 'mouse' not in callbacks:return 255
                action=next(actions)
                if action=='click':callbacks['mouse'](1,600,400,None,None);return 255
                return {'t':ord('t'),'x':ord('x'),'idle':255}[action]
            gui.waitKey.side_effect=waitkey
            with ExitStack() as stack:
                for target,name,value in [(app,'cv2',gui),(app,'load_config',lambda *a:cfg),(app,'HoverPlanner',lambda *a:planner),
                    (app,'AlignedRGBD',lambda *a:camera),(app,'LatestFrames',lambda *a:frames),
                    (app,'PlanWorker',lambda:worker),(control_module,'HoverController',lambda *a,**kw:controller),
                    (app,'release_supported',Mock()),(app,'append_log',lambda path,event:events.append(event)),
                    (app,'draw_clicked_point',Mock())]:stack.enter_context(patch.object(target,name,value))
                gui.setMouseCallback.side_effect=lambda name,fn:callbacks.update(mouse=fn)
                stack.enter_context(patch('sys.argv',['click_hover_eth.py','--execute']))
                self.assertEqual(app.main(),0)
            goals=[call.args[1] for call in planner.plan.call_args_list]
            np.testing.assert_allclose([g[2] for g in goals],expected)
            self.assertEqual(controller.move.call_count,len(expected))
            reference_events=[e for e in events if e.get('event')=='table_reference']
            self.assertEqual(len(reference_events),0 if reused else 1)
            for event in reference_events:self.assertFalse(event['motion_requested'])
            for event in events:
                if event.get('event')=='click':
                    self.assertEqual(event['height_reference'],'table')
                    self.assertAlmostEqual(event['height_above_table_m'],.088+compensation)
                    self.assertAlmostEqual(event['height_compensation_m'],compensation)
                    self.assertAlmostEqual(event['requested_hover_m'],.088)
                    self.assertAlmostEqual(event['hover_base_m'][2],event['table_z_m']+.088+compensation)
            saved=json.loads(Path(cfg['_path']).read_text())['table_reference']
            self.assertAlmostEqual(saved['table_z_m'],.51 if reselect else .5)
            self.assertEqual(saved['calibration_sha256'],'test')
            controller.start.assert_called_once();controller.close.assert_called_once()
            if above:
                self.assertFalse(any(e.get('event')=='rejected' for e in events))
                self.assertLess(next(e for e in events if e.get('event')=='click')['height_offset_m'],0.)

    def test_first_table_click_records_only_then_raised_targets_have_fixed_z(self):self.run_case()
    def test_compensated_commands_keep_saved_reference_and_lower_targets(self):self.run_case(reused=True,compensation=-.01)
    def test_restart_reuses_saved_table_without_a_table_click(self):self.run_case(reused=True)
    def test_t_replaces_table_height_without_moving_on_reference_click(self):self.run_case(reselect=True)
    def test_object_above_hover_plane_keeps_fixed_z_and_submits_motion(self):self.run_case(above=True)
    def test_calibration_change_requires_fresh_table_reference(self):self.run_case(mismatch=True)


if __name__=='__main__':unittest.main()

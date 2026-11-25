"""Offline wrist geometry and startup/exit orchestration. Never opens hardware."""
from pathlib import Path
import contextlib
import io
import json
import queue
import sys
import unittest
from unittest.mock import Mock
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wrist.click_hover import WristPlanner,load_config,freeze_click,save_table_reference
from wrist.demo_flow import DemoFlow,StationaryWindow,WristFrame,click_pose
from legacy_model_fixture import load_legacy_config as load_config
from eyehand.click_hover import MotionPlan


class CapturedWorker:
    def __init__(self):
        self.active=False;self.results=queue.Queue();self.outcome=None;self.submissions=[];self.generation=0
    def cancel(self):
        self.generation+=1;self.outcome=None
    def poll(self):
        out,self.outcome=self.outcome,None;return out
    def submit(self,fn,context):
        self.submissions.append((fn,context));self.active=True
    def finish(self,plan=None,error=None):
        self.active=False;self.outcome=(self.submissions[-1][1],plan,error)


class FakeController:
    def __init__(self):
        self.motion_epoch=0;self.busy=False;self.q=np.zeros(7);self.complete=None;self.moves=[];self.enabled=True
    def hold(self,reason):
        self.motion_epoch+=1;self.busy=False
    def snapshot(self):return self.q.copy()
    def planning_start(self):return self.q[:6].copy()
    def take_completion(self):
        c,self.complete=self.complete,None;return c
    def take_motion_traces(self):return []
    def move(self,plan,expected_epoch):
        if expected_epoch!=self.motion_epoch:raise RuntimeError('stale')
        self.moves.append(plan);self.busy=True


class WristGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=load_config();cls.planner=WristPlanner(cls.cfg)
        cls.ready=cls.planner.plan_ready(np.zeros(6))

    def test_reference_actual_ready_pose_and_zero_offsets(self):
        self.assertEqual(self.cfg['ready_pose'],dict(x=.3,y=0.,z=.3,roll=0.,pitch=.7,yaw=0.,duration=3.))
        np.testing.assert_array_equal(self.planner.offsets,np.zeros(6))
        self.assertEqual(self.cfg['base_compensation_m'],[0.,0.,0.])
        self.assertLess(np.linalg.norm(self.planner.tcp(self.ready.goal)-[.3,0,.3]),.0002)

    def test_ready_full_orientation_matches_reference(self):
        pin=self.planner.pin;d=self.planner.model.createData()
        pin.forwardKinematics(self.planner.model,d,self.ready.goal);pin.updateFramePlacements(self.planner.model,d)
        expected=pin.rpy.rpyToMatrix(0.,.7,0.)
        self.assertLess(np.linalg.norm(d.oMf[self.planner.tcp_id].rotation-expected),.001)

    def test_target_is_fixed_in_base_for_moving_camera(self):
        p=self.planner;point=np.array([.4,0,0]);q1=self.ready.goal;q2=q1.copy();q2[0]+=.15
        T1,T2=p.camera_pose(q1),p.camera_pose(q2)
        c1=(np.linalg.inv(T1)@np.r_[point,1])[:3];c2=(np.linalg.inv(T2)@np.r_[point,1])[:3]
        b1,h1=freeze_click(p,c1,q1,.1,table_z_m=0.);b2,h2=freeze_click(p,c2,q2,.1,table_z_m=0.)
        np.testing.assert_allclose(b1,point,atol=1e-12);np.testing.assert_allclose(b2,point,atol=1e-12)
        np.testing.assert_allclose(h1,h2,atol=1e-12)
        self.assertFalse(np.allclose(c1,c2))
        self.assertNotEqual(p.project_base(h1,q1),p.project_base(h1,q2))

    def test_elevated_object_keeps_fixed_table_height_and_clicked_xy(self):
        p=self.planner;q=self.ready.goal;T=p.camera_pose(q)
        table_z=.00518327030359772
        for point in ([.4,0.,table_z],[.3643584225521316,-.15612388719900505,.0866997051579117]):
            camera=(np.linalg.inv(T)@np.r_[point,1])[:3]
            surface,target=freeze_click(p,camera,q,.1,table_z_m=table_z)
            np.testing.assert_allclose(surface,point,atol=1e-12)
            np.testing.assert_allclose(target[:2],point[:2],atol=1e-12)
            self.assertAlmostEqual(target[2],.10518327030359772)
            # Freeze in base: another camera pose sees the same fixed target.
            other=q.copy();other[0]+=.1
            camera2=(np.linalg.inv(p.camera_pose(other))@np.r_[point,1])[:3]
            _,target2=freeze_click(p,camera2,other,.1,table_z_m=table_z)
            np.testing.assert_allclose(target2,target,atol=1e-12)

    def test_missing_reference_is_rejected_and_elevated_surface_keeps_table_z(self):
        p=self.planner;q=self.ready.goal
        camera=(np.linalg.inv(p.camera_pose(q))@np.array([.4,0,.2,1]))[:3]
        for table in (None,float('nan'),float('inf')):
            with self.assertRaises(ValueError):freeze_click(p,camera,q,.1,table_z_m=table)
        surface,target=freeze_click(p,camera,q,.1,table_z_m=0.)
        np.testing.assert_allclose(surface,[.4,0,.2],atol=1e-12)
        np.testing.assert_allclose(target,[.4,0,.1],atol=1e-12)

    def test_actual_model_hover_then_raw_home(self):
        p=self.planner;hover=p.plan(self.ready.goal,np.array([.4,0,.1]));home=p.plan_home(hover.goal)
        self.assertLess(np.linalg.norm(p.tcp(hover.goal)-[.4,0,.1]),.0002)
        np.testing.assert_array_equal(home.goal,np.zeros(6))
        self.assertEqual(home.kind,'home')

    def test_ready_sdk_samples_stay_inside_joint_limits(self):
        p=self.planner
        for t in self.ready.check_grid:p.validate_raw(self.ready.curve(t))
        self.assertEqual(self.ready.duration,3.*self.cfg['transit']['time_scale'])
        self.assertEqual(len(self.ready.points),301)
        self.assertLessEqual(np.abs(np.diff(self.ready.points,axis=0)).max(),self.cfg['tracking_limit_rad'])
        self.assertFalse(self.ready.diagnostics['hover_speed_limits_applied'])

    def test_cancel_ready_before_any_planning(self):
        with self.assertRaises(InterruptedError):self.planner.plan_ready(np.zeros(6),cancelled=lambda:True)


class ClickAndExitTests(unittest.TestCase):
    def make_flow(self):
        c=FakeController();w=CapturedWorker();p=Mock();p.cfg={'ready_pose':{}};events=[]
        return DemoFlow(p,c,events.append,worker=w),c,w,events

    def fakeplan(self,kind):
        return type('Plan',(),{'goal':np.zeros(6),'duration':3.,'kind':kind})()

    def finish_move(self, flow, controller, worker, kind, status='SETTLED_MODEL_ONLY'):
        flow.pump();worker.finish(self.fakeplan(kind));flow.pump()
        controller.busy=False;controller.complete={'status':status};flow.pump()

    def test_every_hover_requires_a_completed_key_requested_return_before_next_click(self):
        f,c,w,e=self.make_flow();f.ready();self.finish_move(f,c,w,'ready')
        for cycle in range(2):
            self.assertTrue(f.can_click)
            f.hover([.4,0,.1],{});self.finish_move(f,c,w,'hover')
            self.assertEqual(f.phase,'HOVER_HOLD_PRESS_R');self.assertFalse(f.can_click)
            with self.assertRaises(ValueError):f.hover([.41,0,.1],{})
            f.hold();self.assertFalse(f.can_click)
            self.assertTrue(f.ready());f.pump()
            self.assertFalse(f.ready())  # Key repeat cannot restart an active return.
            w.finish(self.fakeplan('ready'));f.pump()
            self.assertFalse(f.can_click)
            c.busy=False;c.complete={'status':'SETTLED_MODEL_ONLY'};f.pump()
            self.assertTrue(f.can_click)
        self.assertEqual([context['kind'] for _,context in w.submissions],
                         ['ready','hover','ready','hover','ready'])

    def test_tracking_stop_and_failed_ready_cannot_unlock_clicks(self):
        for stopped in ('TRACKING_REFUSED_HOLD','UI_TIMEOUT_HOLD','SETTLE_TIMEOUT_HOLD'):
            with self.subTest(stopped=stopped):
                f,c,w,e=self.make_flow();f.phase='HOVER_MOVING'
                c.complete={'status':stopped};f.pump();self.assertFalse(f.can_click)
                f.ready();self.finish_move(f,c,w,'ready',status='SETTLE_TIMEOUT_HOLD')
                self.assertFalse(f.can_click)
                f.ready();f.pump();w.finish(error=ValueError('unreachable return'));f.pump()
                self.assertFalse(f.can_click)
                f.ready();self.finish_move(f,c,w,'ready');self.assertTrue(f.can_click)

    def test_zero_ready_wait_click_hover_then_home(self):
        f,c,w,e=self.make_flow();f.ready();self.assertFalse(f.can_click);f.pump()
        self.assertEqual(w.submissions[-1][1]['kind'],'ready')
        w.finish(self.fakeplan('ready'));f.pump();self.assertEqual(f.phase,'READY_MOVING');self.assertFalse(f.can_click)
        c.busy=False;c.complete={'status':'SETTLED_MODEL_ONLY'};f.pump();self.assertTrue(f.can_click)
        f.hover([.4,0,.1],{'surface_base_m':[.4,0,0]});f.pump();w.finish(self.fakeplan('hover'));f.pump()
        f.exit();self.assertFalse(f.can_click);f.pump();self.assertEqual(w.submissions[-1][1]['kind'],'home')
        w.finish(self.fakeplan('home'));f.pump();self.assertEqual(f.phase,'HOME_MOVING');self.assertFalse(f.done)
        c.complete={'status':'HOME_REACHED_HOLD'};c.busy=False;f.pump();self.assertTrue(f.done);self.assertTrue(f.home_confirmed)

    def test_exit_discards_a_pending_click_before_home(self):
        f,c,w,e=self.make_flow();f.phase='WAIT_CLICK';f.hover([.4,0,.1],{});f.pump();self.assertTrue(w.active)
        f.exit();self.assertTrue(f.exiting)
        # An already cancelled worker must finish/drain before home gets submitted.
        w.active=False;f.pump();self.assertEqual(w.submissions[-1][1]['kind'],'home')
        self.assertEqual(c.moves,[])

    def test_home_timeout_is_not_success(self):
        f,c,w,e=self.make_flow();f.exiting=True;f.phase='HOME_MOVING';c.complete={'status':'HOME_TIMEOUT_HOLD'}
        f.pump();self.assertTrue(f.done);self.assertFalse(f.home_confirmed);self.assertTrue(c.enabled)

    def test_home_plan_refusal_holds_for_supported_release(self):
        f,c,w,e=self.make_flow();f.exit();f.pump();w.finish(error=ValueError('bad path'));f.pump()
        self.assertTrue(f.done);self.assertFalse(f.home_confirmed);self.assertTrue(c.enabled);self.assertEqual(c.moves,[])

    def test_second_interrupt_or_hold_cancels_return(self):
        f,c,w,e=self.make_flow();f.exit();f.pump();f.hold()
        self.assertTrue(f.done);self.assertFalse(f.home_confirmed);self.assertIsNone(f.pending)

    def test_fault_does_not_start_home(self):
        f,c,w,e=self.make_flow();f.fail('stale feedback');f.pump()
        self.assertTrue(f.done);self.assertFalse(f.home_confirmed);self.assertEqual(w.submissions,[])

    def test_stale_image_changed_q_and_cancelled_epoch_rejected(self):
        c=FakeController();frame=WristFrame(None,None,10.,np.zeros(6),0)
        np.testing.assert_array_equal(click_pose(frame,c,.3,.002,now=10.1),np.zeros(6))
        with self.assertRaises(ValueError):click_pose(frame,c,.3,.002,now=10.4)
        c.q[1]=.003
        with self.assertRaises(ValueError):click_pose(frame,c,.3,.002,now=10.1)
        c.q[1]=0;c.motion_epoch=1
        with self.assertRaises(ValueError):click_pose(frame,c,.3,.002,now=10.1)

    def test_stationary_time_does_not_accumulate_over_drift(self):
        s=StationaryWindow(.5,.002);q=np.zeros(6)
        self.assertFalse(s.update(q,0));self.assertTrue(s.update(q,.6))
        q[1]=.003;self.assertFalse(s.update(q,.7));self.assertFalse(s.update(q,.8))


class GuiLifecycleTests(unittest.TestCase):
    def _run_gui_case(self, reselect=False, elevated_target=False, reuse_table=False,
                      mismatched_table=False, repeat_hover=False):
        import importlib.util
        import time
        from types import SimpleNamespace
        from unittest.mock import patch
        spec=importlib.util.spec_from_file_location('wrist_click_entry_test',ROOT/'scripts/click_hover_eih.py')
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        planner=WristPlanner(load_config());ready_plan=planner.plan_ready(np.zeros(6))
        import tempfile,shutil
        planner.cfg['log_dir']=tempfile.mkdtemp(prefix='wrist_gui_test_')
        self.addCleanup(shutil.rmtree,planner.cfg['log_dir'])
        planner.cfg['table_reference']=None
        if reuse_table or mismatched_table:
            planner.cfg['table_reference']={'table_z_m':.02,'table_reference_id':'saved-table',
                'camera_serial':planner.cfg['serial'],
                'calibration_sha256':'different' if mismatched_table else planner.result_sha256}
        # GUI reference selection must write only a temporary test configuration.
        config_path=Path(planner.cfg['log_dir'])/'click_hover.json'
        planner.cfg['_path']=str(config_path)
        config_path.write_text(json.dumps(planner.cfg))
        current_flow=[];callbacks={};made=[]
        class ImmediateWorker(CapturedWorker):
            def submit(self,fn,context):
                self.submissions.append((fn,context));kind=context['kind']
                goal=ready_plan.goal.copy() if kind!='home' else np.zeros(6)
                plan=SimpleNamespace(kind=kind,goal=goal,duration=.01)
                self.outcome=(context,plan,None)
        class Robot(FakeController):
            def __init__(self,*a,**kw):super().__init__();made.append(self);self.closed=False;self.fault=None
            def start(self):pass
            def heartbeat(self):pass
            def move(self,plan,expected_epoch):
                super().move(plan,expected_epoch);self.q[:6]=plan.goal
                self.complete={'status':'HOME_REACHED_HOLD' if plan.kind=='home' else 'SETTLED_MODEL_ONLY'}
            def take_completion(self):
                r=super().take_completion()
                if r:self.busy=False
                return r
            def close(self):self.closed=True;self.enabled=False
        class Camera:
            def __init__(self,*a):self.click_count=0
            def point(self,*a):
                self.click_count+=1
                if reuse_table:z=.0815
                elif self.click_count==1:z=0.
                elif reselect and self.click_count==2:z=.02
                elif elevated_target and self.click_count==2:z=.2
                else:z=.0815
                return (np.linalg.inv(planner.camera_pose(ready_plan.goal))@np.array([.4,0,z,1]))[:3]
            def close(self):pass
        class Frames:
            def __init__(self,c):pass
            def start(self):pass
            def latest(self,*a):
                return SimpleNamespace(color=np.zeros((720,1280,3),np.uint8),depth=np.full((720,1280),.45),acquired=time.monotonic())
            def close(self):pass
        def flow_factory(p,c,emit):
            f=DemoFlow(p,c,emit,worker=ImmediateWorker());current_flow.append(f);return f
        clicked=[0]
        blocked_clicks=[0];key_calls=[0]
        reselect_requested=[False]
        def waitkey(*a):
            key_calls[0]+=1
            if key_calls[0]>200:raise AssertionError('GUI did not complete the click/return cycle')
            if current_flow and current_flow[0].phase=='HOVER_HOLD_PRESS_R':
                if repeat_hover and clicked[0]<2:
                    callbacks['mouse'](entry.cv2.EVENT_LBUTTONDOWN,601,401,None,None)
                    blocked_clicks[0]+=1
                    return ord('r')
                return ord('q')
            if current_flow and current_flow[0].can_click:
                if reselect and clicked[0]==1 and not reselect_requested[0]:
                    reselect_requested[0]=True
                    return ord('t')
                count=2 if repeat_hover else (1 if reuse_table else (3 if reselect else 2))
                if clicked[0]<count:
                    # The first click records table height; it must not submit motion.
                    if clicked[0]>=1:
                        expected=['ready','hover','ready'] if repeat_hover else ['ready']
                        self.assertEqual([p.kind for p in made[0].moves],expected)
                    clicked[0]+=1;callbacks['mouse'](entry.cv2.EVENT_LBUTTONDOWN,600,400,None,None)
                elif any(p.kind=='hover' for p in made[0].moves):return ord('q')
            return 255
        with patch.object(entry,'AlignedRGBD',Camera),patch.object(entry,'LatestFrames',Frames),patch.object(entry,'HoverController',Robot),patch.object(entry,'DemoFlow',flow_factory),patch.object(entry,'append_log') as log_write,patch.object(entry,'release_supported') as release,patch.object(entry.StationaryWindow,'update',return_value=True),patch.object(entry.cv2,'namedWindow'),patch.object(entry.cv2,'imshow'),patch.object(entry.cv2,'destroyAllWindows'),patch.object(entry.cv2,'getWindowProperty',return_value=1),patch.object(entry.cv2,'setMouseCallback',side_effect=lambda w,fn:callbacks.update(mouse=fn)),patch.object(entry.cv2,'waitKey',side_effect=waitkey),contextlib.redirect_stdout(io.StringIO()):
            result=entry.run(planner.cfg,planner,True)
        self.assertEqual(result,0)
        expected=['ready','hover','ready','hover','home'] if repeat_hover else ['ready','hover','home']
        self.assertEqual([p.kind for p in made[0].moves],expected)
        if repeat_hover:
            self.assertEqual(blocked_clicks[0],1)
            blocked=[call.args[1] for call in log_write.call_args_list
                     if call.args[1].get('event')=='click_blocked']
            self.assertEqual(len(blocked),1)
            self.assertEqual(blocked[0]['reason'],'return_ready_required')
            self.assertEqual(blocked[0]['phase'],'HOVER_HOLD_PRESS_R')
        hover_context=next(context for fn,context in current_flow[0].worker.submissions if context['kind']=='hover')
        table_z=.02 if reselect or reuse_table else 0.
        self.assertAlmostEqual(hover_context['event']['hover_base_m'][2],table_z+planner.cfg['hover_m']+planner.cfg['height_compensation_m'])
        clicked_surface_z=.2 if elevated_target else .0815
        self.assertAlmostEqual(hover_context['event']['surface_base_m'][2],clicked_surface_z)
        self.assertAlmostEqual(hover_context['event']['height_offset_m'],table_z+planner.cfg['hover_m']+planner.cfg['height_compensation_m']-clicked_surface_z)
        self.assertEqual(hover_context['event']['height_reference'],'table')
        # A fresh process/config load retains the selected table height.
        persisted=load_config(config_path)['table_reference']
        self.assertAlmostEqual(persisted['table_z_m'],table_z)
        self.assertEqual(persisted['calibration_sha256'],planner.result_sha256)
        if reuse_table:self.assertEqual(persisted['table_reference_id'],'saved-table')
        self.assertTrue(made[0].closed);release.assert_not_called()

    def test_real_run_ready_table_click_hover_home_close_without_release(self):
        self._run_gui_case()

    def test_reselect_table_click_records_without_moving(self):
        self._run_gui_case(reselect=True)

    def test_surface_above_hover_plane_submits_fixed_table_hover(self):
        self._run_gui_case(elevated_target=True)

    def test_saved_table_first_click_is_motion_target(self):
        self._run_gui_case(reuse_table=True)

    def test_mouse_clicks_after_hover_are_ignored_until_r_returns_ready(self):
        self._run_gui_case(reuse_table=True,repeat_hover=True)

    def test_different_calibration_requires_new_reference(self):
        self._run_gui_case(mismatched_table=True)


class TablePersistenceTests(unittest.TestCase):
    def test_saving_reference_preserves_other_config_settings(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'click_hover.json'
            raw={'hover_m':.1,'custom_setting':'preserved','table_reference':None}
            path.write_text(json.dumps(raw))
            cfg={'_path':str(path)}
            reference={'table_z_m':-.0001374148658709884,'table_reference_id':'test'}
            save_table_reference(cfg,reference)
            saved=json.loads(path.read_text())
            self.assertEqual(saved,{**raw,'table_reference':reference})
            self.assertEqual(cfg['table_reference'],reference)
            self.assertEqual(list(Path(folder).iterdir()),[path])


if __name__=='__main__':unittest.main()

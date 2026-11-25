"""Real SDK planning plus fake feedback lifecycle checks; no devices opened."""
from pathlib import Path
import sys
import tempfile
import threading
import time
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation
from eyehand.auto_collection import (CollectionPlanner,StationaryWindow,collection_plan,
    coverage,load_settings,sample_from_window,apply_sampling_region)
from eyehand.auto_collection_runtime import AutoSequence,RGBFrame
import eyehand.auto_collection_runtime as runtime
from eyehand.kinematics import CorrectedKinematics

ROOT=Path(__file__).resolve().parents[1]


class CollectionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings,cls.cfg,cls.poses=load_settings(ROOT/'auto_collection_config.json')
        cls.cfg['result']='/definitely/not/a/previous_calibration.json'
        cls.p=CollectionPlanner(cls.cfg)

    def test_nominal_fk_and_no_previous_extrinsic(self):
        self.assertTrue(np.all(self.p.offsets==0))
        q=np.array([.1,2.,1.9,-1.4,.1,-.2])
        original=CorrectedKinematics(self.p.robot_metadata)
        np.testing.assert_allclose(self.p.nominal_fk(q),original.fk(q),atol=1e-12)
        self.assertFalse(hasattr(self.p,'T_base_camera'))

    def resume_fixture(self,path):
        rows=[dict(row,source_index=i) for i,row in enumerate(self.poses[:2])]
        image=np.zeros((4,4,3),np.uint8)
        runtime.cv2.imwrite(str(path/'saved.png'),image)
        dataset=dict(dataset_id='previous-id',synthetic=False,image_size=[4,4],
            camera=dict(serial='camera'),K=np.eye(3).tolist(),D=[0.]*5,
            robot=self.p.robot_metadata,acquisition=dict(split='train'),
            samples=[dict(id='saved-id',image='saved.png',pose_index=0,requested_pose=rows[0])])
        runtime.save(path/'dataset.json',dataset)
        runtime.save(path/'session.json',dict(settings=self.settings,control_config=self.p.cfg))
        runtime.save(path/'collection_end.json',dict(cleanup_pending=False))
        runtime.save(path/'planned_poses.json',dict(accepted=[dict(index=i,**row) for i,row in enumerate(rows)]))
        return rows

    def test_resume_pose_number_is_not_sample_count_and_clone_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';source.mkdir()
            output=Path(directory)/'new';output.mkdir()
            rows=self.resume_fixture(source)
            original=(source/'dataset.json').read_bytes()
            resume=runtime.load_resume(source,self.p,self.settings,rows,2)
            self.assertEqual(resume['start_index'],1)
            self.assertEqual(runtime.load_resume(source,self.p,self.settings,rows)['start_index'],1)
            result=runtime.clone_resume(resume,output,self.p.robot_metadata)
            self.assertEqual(result['samples'],resume['dataset']['samples'])
            self.assertNotEqual(result['dataset_id'],'previous-id')
            self.assertEqual(result['acquisition']['resumed_from']['start_pose'],2)
            self.assertEqual((source/'dataset.json').read_bytes(),original)
            self.assertEqual((source/'saved.png').read_bytes(),(output/'saved.png').read_bytes())
            with self.assertRaisesRegex(ValueError,'Start pose'):
                runtime.load_resume(source,self.p,self.settings,rows,1)

    def test_resume_rejects_incompatible_config_plan_or_missing_image_before_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory);rows=self.resume_fixture(source)
            with self.assertRaisesRegex(ValueError,'settings changed'):
                runtime.load_resume(source,self.p,dict(self.settings,fps=15),rows)
            changed=[dict(rows[0],group='changed'),rows[1]]
            with self.assertRaisesRegex(ValueError,'sequence changed'):
                runtime.load_resume(source,self.p,self.settings,changed)
            (source/'saved.png').unlink()
            with self.assertRaisesRegex(ValueError,'unreadable saved image'):
                runtime.load_resume(source,self.p,self.settings,rows)

    def test_resume_camera_checked_before_enabling_and_source_change_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';source.mkdir()
            output=Path(directory)/'new';output.mkdir()
            rows=self.resume_fixture(source)
            resume=runtime.load_resume(source,self.p,self.settings,rows)
            camera=SimpleNamespace(metadata=dict(serial='camera'),width=4,height=4,
                                   k=np.eye(3),d=np.zeros(5))
            runtime.check_resume_camera(resume['dataset'],camera)
            camera.k=np.eye(3)*2
            with self.assertRaisesRegex(ValueError,'intrinsics changed'):
                runtime.check_resume_camera(resume['dataset'],camera)
            (source/'dataset.json').write_bytes((source/'dataset.json').read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'Source dataset changed'):
                runtime.clone_resume(resume,output,self.p.robot_metadata)

    def test_mass_model_removed_camera_retained_16g_exactly_once(self):
        self.assertEqual(self.p.gravity['wrist_camera_mass_kg'],0.)
        self.assertEqual(self.p.gravity['payload_mass_kg'],.016)
        difference=sum(i.mass for i in self.p.gravity_model.inertias)-sum(i.mass for i in self.p.model.inertias)
        self.assertAlmostEqual(self.p.gravity_exclusion['removed_mass_kg'],.160,places=10)
        self.assertEqual(self.p.gravity_exclusion['links'],['camera_mount_link','camera_link'])
        self.assertAlmostEqual(difference,-.160+.016,places=10)
        self.assertIs(self.p.sdk_law.model,self.p.gravity_model)
        self.assertAlmostEqual(self.p.robot_metadata['gravity_parameters']['removed_urdf_mass_kg'],.160)

    def test_excluded_gravity_matches_camera_free_sdk_reference_values(self):
        # Reference vectors independently computed with the camera-free SDK URDF
        # plus the same 16 g target, not by the exclusion implementation.
        references=[
            ([.1,1.4,1.5,-.8,.2,-.3],[0.,-4.710224750169308,5.95991763177121,1.6925013936446278,.11281781527394495,-.0005730091374141784]),
            ([-.2,1.7,1.1,-.6,-.3,.2],[0.,-6.830307706402713,5.6885647765317975,.9309821652268182,-.24104869489556505,-.00024058125613732437])]
        for q,expected in references:
            np.testing.assert_allclose(self.p.gravity_torque(q),expected,rtol=0,atol=1e-10)

    def test_gravity_exclusion_leaves_original_urdf_and_fk_unchanged(self):
        import hashlib
        path=Path(self.cfg['collection_urdf']);before=hashlib.sha256(path.read_bytes()).hexdigest()
        legacy=CollectionPlanner(dict(self.cfg,collection_gravity_excluded_urdf_links=[]))
        for q in [np.array([.1,1.4,1.5,-.8,.2,-.3]),np.array([-.2,1.7,1.1,-.6,-.3,.2])]:
            np.testing.assert_allclose(self.p.nominal_fk(q),legacy.nominal_fk(q),atol=1e-12)
            self.assertGreater(np.max(np.abs(self.p.gravity_torque(q)-legacy.gravity_torque(q))),.1)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)

    def test_reconfiguring_excluded_gravity_never_stacks_payload(self):
        planner=CollectionPlanner(dict(self.cfg))
        q=np.array([.1,1.4,1.5,-.8,.2,-.3]);before=planner.gravity_torque(q)
        planner._configure_collection_gravity(Path(self.cfg['collection_urdf']))
        np.testing.assert_allclose(planner.gravity_torque(q),before,atol=1e-12)

    def test_excluded_camera_mass_cannot_be_added_back_as_point_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'gravity.json'
            gravity=json.loads(Path(self.cfg['gravity_config']).read_text());gravity['wrist_camera_mass_kg']=.16
            path.write_text(json.dumps(gravity))
            with self.assertRaisesRegex(ValueError,'already has wrist-camera frames|zero added wrist-camera mass'):
                CollectionPlanner(dict(self.cfg,gravity_config=str(path)))

    def test_visible_validation_config_avoids_unobservable_low_group(self):
        settings,cfg,poses=load_settings(ROOT/'auto_validation_config.json','validation')
        self.assertEqual(len(poses),24)
        self.assertEqual(cfg['collection_gravity_excluded_urdf_links'],['camera_mount_link','camera_link'])
        self.assertIsNone(settings['sampling_region'])
        self.assertTrue(all(row['xyz_rpy'][2]>=.23 for row in poses))
        self.assertTrue(all('low' not in row['group'] for row in poses))
        self.assertEqual(sum('near_horizontal' in row['group'] for row in poses),6)
        with self.assertRaisesRegex(ValueError,'Need 1..150'):
            load_settings(ROOT/'auto_validation_config.json','train')

    def test_official_full_path_and_limits(self):
        plan=collection_plan(self.p,np.zeros(6),self.poses[0],self.settings)
        self.assertGreater(len(plan.points),100)
        self.assertFalse(plan.tracking_outer_loop)
        self.assertEqual(plan.kind,'calibration')
        np.testing.assert_array_equal(plan.goal,plan.points[-1])
        np.testing.assert_allclose(plan.target,self.poses[0]['xyz_rpy'][:3],atol=1e-8)
        dt=plan.duration/len(plan.points)
        self.assertLessEqual(np.abs(np.diff(plan.points,axis=0)/dt).max(),self.cfg['max_joint_speed_rad_s'])
        self.assertTrue(all(self.p.tcp(q)[2]>=self.settings['min_tcp_z_m'] for q in plan.points))

    def test_time_stretch_updates_recorded_sample_cadence(self):
        settings=dict(self.settings,min_move_duration_s=15.)
        plan=collection_plan(self.p,np.zeros(6),self.poses[0],settings)
        self.assertGreaterEqual(plan.duration,15.)
        self.assertAlmostEqual(plan.diagnostics['target_update_hz'],len(plan.points)/plan.duration)
        self.assertAlmostEqual(plan.diagnostics['replay_sample_interval_s'],plan.duration/len(plan.points))

    def rows(self,drift=0.):
        rows=[]
        for i in range(25):
            g=np.eye(4);g[0,3]=.3+drift*i/24
            uv=np.array([[100,100],[200,100],[200,200],[100,200]],float)
            obs=dict(T_camera_target=np.eye(4).tolist(),image_points=uv.tolist(),
                object_points=np.zeros((4,3)).tolist(),K=np.eye(3).tolist(),D=[0.]*5,
                pnp_rms_px=.1,ippe_second_solution_gap_px=1.)
            frame=RGBFrame(np.zeros((4,4,3),np.uint8),dict(frame_number=i),i/30.)
            rows.append((frame,g,dict(q_rad=[0.]*6),obs))
        return rows

    def test_stationary_window_requires_distinct_frames_and_duration(self):
        w=StationaryWindow(self.settings)
        row=self.rows()[0]
        for _ in range(30):self.assertFalse(w.add(*row))
        self.assertEqual(len(w.rows),1)
        for r in self.rows()[1:]:ready=w.add(*r)
        self.assertTrue(ready)
        sample=sample_from_window(w,dict(samples=[]))
        self.assertGreaterEqual(sample['stationary_duration_s'],self.settings['stationary_s'])
        self.assertGreaterEqual(len(sample['robot_reads']),5)
        self.assertEqual(sample['robot_reads'][-1]['q_rad'],[0.]*6)

    def test_moving_feedback_and_duplicate_pose_rejected(self):
        w=StationaryWindow(self.settings)
        for r in self.rows(drift=.004):ready=w.add(*r)
        self.assertFalse(ready)
        self.assertEqual(w.reason,'Robot not stationary')
        w.clear()
        for r in self.rows():w.add(*r)
        sample=sample_from_window(w,dict(samples=[]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            sample_from_window(w,dict(samples=[sample]))

    def test_insufficient_coverage_not_passed(self):
        self.assertEqual(coverage(dict(samples=[]))['status'],'REVIEW_COVERAGE_OR_COLLECT_MORE')
        result=coverage(dict(samples=[],acquisition=dict(split='validation')))
        self.assertEqual(result['minimum_samples'],15)
        self.assertEqual(result['status'],'REVIEW_COVERAGE_OR_COLLECT_MORE')

    def test_missing_low_pose_group_prevents_ready_status(self):
        samples=[]
        for i in range(40):
            g=np.eye(4);g[:3,:3]=Rotation.from_euler('xyz',[.3*np.sin(i),.3*np.cos(i),0]).as_matrix()
            samples.append(dict(T_base_ee=g.tolist(),robot_reads=[dict(q_rad=[0.]*6)],
                                requested_pose=dict(group='volume')))
        result=coverage(dict(samples=samples,acquisition=dict(split='train',
            planned_groups=dict(volume=40,downward_low_task=10))))
        self.assertEqual(result['status'],'REVIEW_MISSING_POSE_GROUPS')
        self.assertIn('downward_low_task',result['undersampled_groups'])

    def test_frame_gaps_restart_stationary_window(self):
        w=StationaryWindow(self.settings)
        rows=self.rows()
        for row in rows[:10]:w.add(*row)
        frame,g,meta,obs=rows[10]
        late=RGBFrame(frame.color,frame.metadata,frame.acquired+.2)
        self.assertFalse(w.add(late,g,meta,obs))
        self.assertEqual(len(w.rows),1)
        with self.assertRaisesRegex(ValueError,'stationary'):sample_from_window(w,dict(samples=[]))

    def test_unstable_pnp_pose_or_gripper_rejected_even_with_fixed_pixels(self):
        for mode in ('vision','gripper'):
            w=StationaryWindow(self.settings)
            for i,(f,g,meta,obs) in enumerate(self.rows()):
                meta=dict(meta,gripper_q_rad=[i*.001 if mode=='gripper' else .25])
                if mode=='vision':
                    c=np.eye(4);c[2,3]=i*.001;obs=dict(obs,T_camera_target=c.tolist())
                ready=w.add(f,g,meta,obs)
            self.assertFalse(ready,mode)

    def test_ignored_table_settings_cannot_block_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            config=json.loads((ROOT/'auto_collection_config.json').read_text())
            base=json.loads((ROOT/'click_hover_config.json').read_text())
            base['table_reference']=dict(table_z_m='invalid for click demo')
            for key in ('sdk_root','result','gravity_config','log_dir'):
                base[key]=str((ROOT/base[key]).resolve())
            config['urdf']=str((ROOT/config['urdf']).resolve())
            bad=Path(directory)/'base.json';bad.write_text(json.dumps(base))
            config['base_config']=str(bad);config['gravity_config']=str(ROOT/'gravity_collection.json')
            path=Path(directory)/'auto.json';path.write_text(json.dumps(config))
            _,cfg,_=load_settings(path)
            self.assertNotIn('table_reference',cfg)

    def test_home_path_uses_collection_height_floor(self):
        p=self.p
        plan=collection_plan(p,np.zeros(6),self.poses[0],self.settings)
        home=p.plan_home(plan.goal)
        self.assertTrue(all(p.tcp(q)[2]>=self.settings['min_tcp_z_m']-1e-6 for q in home.points))
        raised=dict(p.cfg);raised['collection_min_tcp_z_m']=.4
        with patch.object(p,'cfg',raised):
            with self.assertRaisesRegex(ValueError,'minimum TCP height'):p.plan_home(plan.goal)

    def test_cancel_before_ik_does_not_start_sdk_solver(self):
        with patch.object(self.p.ik,'solve_ik') as solve:
            with self.assertRaises(InterruptedError):
                collection_plan(self.p,np.zeros(6),self.poses[0],self.settings,lambda:True)
            solve.assert_not_called()

    def test_failed_json_commit_preserves_previous_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);data=dict(samples=[])
            runtime.save(output/'dataset.json',data)
            sample=dict(id='new',image='new.png')
            with patch.object(runtime,'save',side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError,'disk full'):
                    runtime.persist_sample(output,data,sample,np.zeros((4,4,3),np.uint8))
            self.assertEqual(data['samples'],[])
            self.assertEqual(json.loads((output/'dataset.json').read_text())['samples'],[])
            self.assertFalse((output/'new.png').exists())

    def test_interrupt_after_commit_keeps_referenced_image(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);data=dict(samples=[]);runtime.save(output/'dataset.json',data)
            original_save=runtime.save
            def commit_then_interrupt(*args):original_save(*args);raise KeyboardInterrupt
            sample=dict(id='new',image='new.png')
            with patch.object(runtime,'save',side_effect=commit_then_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    runtime.persist_sample(output,data,sample,np.zeros((4,4,3),np.uint8))
            self.assertEqual(data['samples'],[sample])
            self.assertTrue((output/'new.png').exists())

    def test_split_and_30hz_identity(self):
        s,c,poses=load_settings(ROOT/'auto_collection_config.json','validation')
        self.assertEqual(s['fps'],30)
        self.assertEqual(c['serial'],'250122070127')
        self.assertEqual(len(poses),28)
        self.assertTrue(set(tuple(p['xyz_rpy']) for p in poses).isdisjoint(
            tuple(p['xyz_rpy']) for p in self.poses))

    def test_forward_region_preserves_source_and_does_not_accumulate(self):
        self.assertEqual(self.poses[0]['source_xyz_rpy'][:3],[.28,0.,.30])
        self.assertEqual(self.poses[0]['xyz_rpy'][:3],[.41,0.,.30])
        mapped=apply_sampling_region(self.poses,self.settings['sampling_region'])
        self.assertEqual([r['xyz_rpy'] for r in mapped],[r['xyz_rpy'] for r in self.poses])
        low=next(r for r in mapped if r['group']=='downward_low_task')
        self.assertEqual(low['sampling_region_kind'],'low_downward')
        self.assertAlmostEqual(low['xyz_rpy'][0]-low['source_xyz_rpy'][0],.02)
        self.assertTrue(all(abs(r['xyz_rpy'][1])<=.09 and r['xyz_rpy'][2]<=.30 for r in mapped))

    def test_sampling_region_validation_and_optional_legacy_mode(self):
        bad=dict(self.settings['sampling_region'],normal_x_range_m=[.46,.41])
        with self.assertRaisesRegex(ValueError,'Reversed'):apply_sampling_region(self.poses,bad)
        rows=[dict(xyz_rpy=[.28,0,.30,0,.5,0],group='test')]
        unchanged=apply_sampling_region(rows,None)
        self.assertEqual(unchanged,rows)
        self.assertIsNot(unchanged[0]['xyz_rpy'],rows[0]['xyz_rpy'])

    def test_repeated_capture_failure_pauses_without_launching_next_pose(self):
        controller=Mock();controller.motion_epoch=1;controller.take_completion.return_value=None
        seq=AutoSequence(self.p,controller,self.poses,self.settings,Mock())
        seq.worker=Mock();seq.worker.active=False;seq.worker.poll.return_value=None
        seq._submit=Mock()
        for _ in range(3):
            seq.phase='CAPTURING';seq.deadline=0;seq.pump()
        self.assertEqual(seq.phase,'PAUSED');self.assertEqual(seq.index,3)
        self.assertEqual(seq._submit.call_count,2)
        controller.hold.assert_called_once()
        self.assertEqual(len(seq.skipped),3)
        seq.resume();self.assertEqual(seq.capture_failures,0)
        self.assertEqual(seq._submit.call_count,3)

    def test_urdf_snapshot_is_self_contained_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata=runtime.snapshot_robot(self.p,Path(directory))
            snapshot=Path(metadata['urdf'])
            self.assertTrue(snapshot.exists())
            np.testing.assert_allclose(CorrectedKinematics(metadata).fk(np.zeros(6)),self.p.nominal_fk(np.zeros(6)))
            snapshot.write_text('changed')
            with self.assertRaisesRegex(ValueError,'differs'):runtime.snapshot_robot(self.p,Path(directory))

    def test_camera_is_closed_even_if_worker_never_started(self):
        camera=Mock();frames=runtime.CollectionFrames(camera)
        frames.close();camera.close.assert_called_once()

    def test_warmup_exit_never_constructs_robot_and_records_cancel(self):
        camera=Mock(metadata={},width=1280,height=720)
        frames=Mock();frames.stats.return_value=dict(unique_received_frames=0,observed_receive_hz=None)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runtime,'Camera',return_value=camera),\
                 patch.object(runtime,'CollectionFrames',return_value=frames),\
                 patch.object(runtime,'HoverController') as robot,\
                 patch.object(runtime.cv2,'namedWindow'),patch.object(runtime.cv2,'imshow'),\
                 patch.object(runtime.cv2,'destroyAllWindows'),\
                 patch.object(runtime.cv2,'waitKey',return_value=ord('q')),\
                 patch.object(runtime.cv2,'getWindowProperty',return_value=1):
                status=runtime.run(self.p,self.settings,[self.poses[0]],Path(directory),True)
            self.assertEqual(status,0);robot.assert_not_called();frames.close.assert_called_once()
            end=json.loads((Path(directory)/'collection_end.json').read_text())
            self.assertEqual(end['status'],'CANCELLED_BEFORE_ENABLE')
            self.assertFalse(end['robot_open_attempted'])

    def test_cancelled_planning_cannot_move(self):
        controller=Mock();controller.motion_epoch=1
        controller.planning_start.return_value=np.zeros(6)
        controller.take_completion.return_value=None
        seq=AutoSequence(self.p,controller,[self.poses[0]],self.settings,Mock())
        seq.worker=Mock();seq.worker.active=False;seq.worker.poll.return_value=None
        seq.resume();self.assertEqual(seq.phase,'PLANNING')
        seq.pause();controller.motion_epoch=2
        seq.worker.poll.return_value=((1,'pose',0),Mock(),None)
        seq.pump();controller.move.assert_not_called()

    def test_failed_home_keeps_holding(self):
        controller=Mock();controller.motion_epoch=1
        controller.take_completion.return_value=dict(status='HOME_TIMEOUT_HOLD')
        seq=AutoSequence(self.p,controller,[self.poses[0]],self.settings,Mock())
        seq.phase='HOMING';seq.worker=Mock();seq.worker.poll.return_value=None
        seq.pump()
        self.assertEqual(seq.phase,'PAUSED');self.assertFalse(seq.home_ok)

    def test_resumed_runtime_starts_at_next_pose_keeps_old_sample_and_homes(self):
        status,data,end,moves=self.fake_session(resumed=True)
        self.assertEqual(status,0)
        self.assertTrue(end['home_verified'])
        self.assertEqual(moves,['calibration','home'])

    def fake_session(self,window_failure=False,cleanup_log_failure=False,closed_home_failure=False,
                     cancel_before_enable=False,report_failure=False,resumed=False):
        obs=self.rows()[0][3]
        class FakeCamera:
            def __init__(self,*a,**kw):
                self.metadata=dict(serial='250122070127');self.width=1280;self.height=720
                self.k=np.eye(3);self.d=np.zeros(5);self.number=0;self.closed=False
                cameras.append(self)
            def read(self,**kw):
                time.sleep(1/30);self.number+=1
                self.last_frame_metadata=dict(frame_number=self.number,host_received_unix_s=time.time())
                return np.zeros((720,1280,3),np.uint8)
            def close(self):self.closed=True
        class FakeController:
            def __init__(self,p,**kw):
                self.io=threading.RLock();self.q=np.r_[np.zeros(6),.25]
                self.feedback=SimpleNamespace(times=np.full(7,time.perf_counter()),sequences=np.ones(7,int))
                self.enabled=False;self.fault=None;self.motion_epoch=0;self.completion=None
                self.failure_diagnostics=dict(reason='injected') if cleanup_log_failure else None
                self.moves=[];controllers.append(self)
            def start(self):
                if cancel_before_enable:raise runtime.CollectionCancelled()
                self.enabled=True
            def snapshot(self):
                self.feedback.times[:]=time.perf_counter();self.feedback.sequences+=1
                return self.q.copy()
            def planning_start(self):return self.q[:6].copy()
            def heartbeat(self):pass
            def hold(self,*a):self.motion_epoch+=1
            def move(self,plan,**kw):
                self.moves.append(plan.kind);self.q[:6]=plan.goal
                status=('HOME_TIMEOUT_HOLD' if closed_home_failure else 'HOME_REACHED_HOLD') if plan.kind=='home' else 'SETTLED_MODEL_ONLY'
                self.completion=dict(status=status)
            def take_completion(self):
                c=self.completion;self.completion=None;return c
            def close(self):self.enabled=False
        cameras=[];controllers=[]
        detector=Mock();detector.detect.return_value=obs
        original_open=Path.open;original_save=runtime.save
        def opened(path,*args,**kwargs):
            if cleanup_log_failure and path.name=='events.jsonl' and cameras and cameras[0].closed:
                raise OSError('injected cleanup log failure')
            return original_open(path,*args,**kwargs)
        def saved(path,*args):
            if report_failure and path.name=='coverage_report.json':raise OSError('injected report failure')
            return original_save(path,*args)
        def visible(*args):
            return 0 if closed_home_failure and controllers and controllers[0].moves else 1
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);rows=[self.poses[0]];resume=None
            if resumed:
                source=output/'source';source.mkdir()
                rows=self.resume_fixture(source)
                data=json.loads((source/'dataset.json').read_text())
                data.update(image_size=[1280,720],camera=dict(serial='250122070127'))
                data['samples'][0].update(T_base_ee=self.p.nominal_fk(np.zeros(6)).tolist(),
                    T_camera_target=obs['T_camera_target'],robot_reads=[dict(q_rad=[0.]*6,gripper_q_rad=[.25])])
                runtime.save(source/'dataset.json',data)
                runtime.cv2.imwrite(str(source/'saved.png'),np.zeros((720,1280,3),np.uint8))
                resume=runtime.load_resume(source,self.p,self.settings,rows,2)
                output=output/'continued';output.mkdir()
            with patch.object(runtime,'Camera',FakeCamera),patch.object(runtime,'HoverController',FakeController),\
                 patch.object(runtime,'Target',return_value=detector),patch.object(runtime,'release_supported') as release,\
                 patch.object(runtime.cv2,'namedWindow'),patch.object(runtime.cv2,'imshow'),\
                 patch.object(runtime.cv2,'destroyAllWindows',side_effect=RuntimeError('injected Qt failure') if window_failure else None),\
                 patch.object(runtime.cv2,'getWindowProperty',side_effect=visible),\
                 patch.object(Path,'open',opened),patch.object(runtime,'save',side_effect=saved),\
                 patch.object(runtime.cv2,'waitKey',side_effect=lambda *a:(time.sleep(.002) or 255)):
                status=runtime.run(self.p,self.settings,rows,output,True,resume=resume)
            data=json.loads((output/'dataset.json').read_text())
            end=json.loads((output/'collection_end.json').read_text())
            if resumed:
                self.assertEqual([row['pose_index'] for row in data['samples']],[0,1])
                self.assertEqual(data['samples'][0]['id'],'saved-id')
                self.assertEqual(end['sample_count'],2)
                self.assertEqual(len(json.loads((source/'dataset.json').read_text())['samples']),1)
                events=[json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
                self.assertEqual([row['index'] for row in events if row['event']=='move' and row['kind']=='pose'],[1])
            for sample in data['samples']:
                fk=CorrectedKinematics(data['robot'])
                np.testing.assert_allclose(fk.fk(sample['robot_reads'][-1]['q_rad']),sample['T_base_ee'],atol=1e-12)
                self.assertEqual(sample['robot_reads'][-1]['gripper_q_rad'],[.25])
                self.assertTrue((output/sample['image']).exists())
            self.assertFalse(controllers[0].enabled)
            self.assertTrue(cameras[0].closed)
            self.assertTrue(release.called)
            self.assertFalse(end['cleanup_pending'])
            return status,data,end,controllers[0].moves

    def test_full_capture_save_home_cleanup_with_fake_devices(self):
        status,data,end,moves=self.fake_session()
        self.assertEqual(status,0);self.assertEqual(len(data['samples']),1)
        self.assertTrue(end['home_verified']);self.assertEqual(moves,['calibration','home'])

    def test_qt_or_log_failure_cannot_bypass_robot_close(self):
        for kwargs in (dict(window_failure=True),dict(cleanup_log_failure=True),dict(report_failure=True)):
            status,data,end,moves=self.fake_session(**kwargs)
            self.assertEqual(status,1);self.assertEqual(len(data['samples']),1)
            self.assertTrue(end['home_verified']);self.assertTrue(end['cleanup_errors'])

    def test_closed_window_and_failed_home_reaches_release_instead_of_hanging(self):
        status,data,end,moves=self.fake_session(closed_home_failure=True)
        self.assertEqual(status,1);self.assertFalse(end['home_verified'])
        self.assertEqual(moves,['calibration','home'])

    def test_cancel_before_enable_saves_end_status_without_moving(self):
        status,data,end,moves=self.fake_session(cancel_before_enable=True)
        self.assertEqual(status,0);self.assertEqual(moves,[])
        self.assertEqual(end['status'],'CANCELLED_BEFORE_ENABLE')


if __name__=='__main__':unittest.main()

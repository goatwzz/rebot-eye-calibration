"""Automatic external-camera collection UI; one existing SDK motor sender."""
from dataclasses import dataclass
from collections import Counter
from datetime import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import uuid
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from .auto_collection import (CollectionPlanner,StationaryWindow,collection_plan,
    coverage,load_settings,preflight,sample_from_window)
from .cli import save
from .corner_demo import window_event
from .hover_control import HoverController
from .hover_runtime import PlanWorker,release_supported
from .vision import Camera,Target

ROOT=Path(__file__).resolve().parents[1]


def load_resume(source,p,settings,rows,start_pose=None):
    """Validate a completed on-disk session before any device access.

    Continuation always writes a new directory; the source is never modified.
    Pose numbering is the full accepted sequence, not the saved-sample count.
    """
    source=Path(source).resolve()
    dataset_bytes=(source/'dataset.json').read_bytes()
    dataset=json.loads(dataset_bytes)
    session=json.loads((source/'session.json').read_text())
    end=json.loads((source/'collection_end.json').read_text())
    if end.get('cleanup_pending',True):raise ValueError('Previous session cleanup is still pending')
    if dataset.get('synthetic') is not False or not dataset.get('samples'):
        raise ValueError('Resume requires existing real samples')
    previous={k:v for k,v in session['settings'].items() if k!='_path'}
    current={k:v for k,v in settings.items() if k!='_path'}
    if previous!=current or session['control_config']!=p.cfg:
        raise ValueError('Collection/control settings changed; cannot resume this dataset')
    if (dataset['robot']['urdf_sha256']!=p.robot_metadata['urdf_sha256'] or
            dataset['robot']['gravity_parameters']!=p.robot_metadata['gravity_parameters']):
        raise ValueError('Nominal robot model or payload changed')
    plan=json.loads((source/'planned_poses.json').read_text())['accepted']
    if len(plan)!=len(rows) or any(old['index']!=row['source_index'] or
            old['group']!=row['group'] or old['xyz_rpy']!=row['xyz_rpy']
            for old,row in zip(plan,rows)):
        raise ValueError('Accepted pose sequence changed; numbering no longer matches')
    ids=set();images=set();indices=set()
    for sample in dataset['samples']:
        i=sample['pose_index'];name=sample['image']
        if (type(i) is not int or not 0<=i<len(rows) or i in indices or
                sample['requested_pose']!=rows[i]):
            raise ValueError('Saved pose identity does not match the accepted sequence')
        if (sample['id'] in ids or name in images or Path(name).name!=name):
            raise ValueError('Invalid or duplicate saved image/sample identity')
        image=cv2.imread(str(source/name))
        if image is None or list(image.shape[1::-1])!=dataset['image_size']:
            raise ValueError('Missing or unreadable saved image: '+name)
        ids.add(sample['id']);images.add(name);indices.add(i)
    start_index=max(indices)+1 if start_pose is None else start_pose-1
    if not max(indices)<start_index<len(rows):
        raise ValueError('Start pose must be after all saved poses and inside the sequence')
    return dict(source=str(source),dataset=dataset,start_index=start_index,
                dataset_sha256=hashlib.sha256(dataset_bytes).hexdigest())


def clone_resume(resume,output,robot_metadata):
    source=Path(resume['source'])
    if hashlib.sha256((source/'dataset.json').read_bytes()).hexdigest()!=resume['dataset_sha256']:
        raise ValueError('Source dataset changed during resume preparation')
    dataset=json.loads(json.dumps(resume['dataset']))
    for sample in dataset['samples']:
        destination=output/sample['image']
        with (source/sample['image']).open('rb') as src,destination.open('xb') as dst:
            shutil.copyfileobj(src,dst)
    parent=dict(path=str(source),dataset_id=dataset['dataset_id'],
                dataset_sha256=resume['dataset_sha256'],inherited_samples=len(dataset['samples']),
                start_pose=resume['start_index']+1)
    dataset.update(dataset_id=str(uuid.uuid4()),robot=robot_metadata)
    dataset['acquisition']['resumed_from']=parent
    save(output/'dataset.json',dataset)
    return dataset


def check_resume_camera(dataset,camera):
    if (dataset['camera']['serial']!=camera.metadata['serial'] or
            dataset['image_size']!=[camera.width,camera.height] or
            not np.allclose(dataset['K'],camera.k,rtol=0,atol=1e-6) or
            not np.allclose(dataset['D'],camera.d,rtol=0,atol=1e-9)):
        raise ValueError('Camera identity/profile/intrinsics changed; do not mix calibration data')


class CollectionCancelled(Exception):
    """Normal operator exit before any automatic motion."""


def snapshot_robot(p,output):
    """Keep the exact nominal URDF beside the data even if SDK files change."""
    source=Path(p.robot_metadata.get('source_urdf',p.robot_metadata['urdf']))
    data=source.read_bytes();digest=hashlib.sha256(data).hexdigest()
    if digest!=p.robot_metadata['urdf_sha256']:
        raise ValueError('URDF changed after planner initialization')
    snapshot=output/'robot_nominal.urdf'
    if snapshot.exists():
        if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=digest:
            raise ValueError('Saved nominal URDF differs from collection model')
    else:
        with snapshot.open('xb') as f:f.write(data)
    return dict(p.robot_metadata,urdf=str(snapshot.resolve()),source_urdf=str(source.resolve()))


def persist_sample(output,dataset,sample,color):
    """Publish the PNG before atomically committing JSON; retain prior data on I/O failure."""
    final=output/sample['image'];temporary=output/('.'+sample['image']+'.partial.png')
    if final.exists():raise FileExistsError('Sample image already exists')
    image_published=False
    try:
        if not cv2.imwrite(str(temporary),color):raise OSError('Could not save image')
        temporary.replace(final);image_published=True
        samples=[*dataset['samples'],sample]
        save(output/'dataset.json',dict(dataset,samples=samples))
        dataset['samples']=samples
    except BaseException:
        # An interrupt can arrive immediately AFTER JSON's atomic rename.
        # Check the committed file before removing an image it may reference.
        committed=None
        try:
            recorded=json.loads((output/'dataset.json').read_text())
            committed=any(row['id']==sample['id'] for row in recorded['samples'])
            if committed:dataset['samples']=recorded['samples']
        except (OSError,ValueError,KeyError):pass
        if image_published and committed is False:
            try:final.unlink(missing_ok=True)
            except OSError:pass
        raise
    finally:
        try:temporary.unlink(missing_ok=True)
        except OSError:pass


@dataclass(frozen=True)
class RGBFrame:
    color: object
    metadata: dict
    acquired: float


class CollectionFrames:
    """Single camera owner; publish RGB and metadata together, drain at 30 Hz."""
    def __init__(self,camera):
        self.camera=camera;self.frame=None;self.error=None
        self.lock=threading.Lock();self.stop=threading.Event()
        self.frame_count=0;self.first_received=None;self.last_received=None
        self.thread=threading.Thread(target=self._run,daemon=True,name='calibration-rgb')

    def start(self):
        self.thread.start()

    def _run(self):
        last=-1;warmup=20
        try:
            while not self.stop.is_set():
                try:
                    image=self.camera.read(timeout_ms=200)
                except RuntimeError as exc:
                    if "Frame didn't arrive within" in str(exc):continue
                    raise
                meta=dict(self.camera.last_frame_metadata)
                if meta['frame_number']<=last:continue
                last=meta['frame_number']
                if warmup:warmup-=1;continue
                frame=RGBFrame(image,meta,time.monotonic())
                with self.lock:
                    self.frame=frame;self.frame_count+=1
                    self.first_received=self.first_received or frame.acquired
                    self.last_received=frame.acquired
        except Exception as exc:
            if not self.stop.is_set():
                with self.lock:self.error=str(exc)

    def latest(self,max_age=.3):
        with self.lock:frame,error=self.frame,self.error
        if error:raise RuntimeError('RGB acquisition failed: '+error)
        if frame and not 0<=time.monotonic()-frame.acquired<=max_age:
            raise RuntimeError('RGB frame expired; hold, no automatic next motion')
        return frame

    def close(self):
        self.stop.set();started=self.thread.ident is not None
        if started:self.thread.join(timeout=1.)
        try:self.camera.close()
        finally:
            if started:self.thread.join(timeout=1.)
        if self.thread.is_alive():raise RuntimeError('RGB worker failed to stop')

    def stats(self):
        with self.lock:
            elapsed=(self.last_received-self.first_received) if self.frame_count>1 else None
            return dict(unique_received_frames=self.frame_count,
                observed_receive_hz=(self.frame_count-1)/elapsed if elapsed and elapsed>0 else None)


class AutoSequence:
    """Nonblocking movement/capture lifecycle; no direct motor commands."""
    def __init__(self,p,controller,rows,settings,emit):
        self.p,self.controller,self.rows,self.settings,self.emit=p,controller,rows,settings,emit
        self.worker=PlanWorker();self.index=0;self.phase='PAUSED';self.exiting=False
        self.deadline=0.;self.capture_started=0.
        self.window=StationaryWindow(settings);self.home_ok=False
        self.home_verified=False
        self.skipped=[]
        self.capture_failures=0

    @property
    def current(self):return self.rows[self.index] if self.index<len(self.rows) else None

    def pause(self):
        self.worker.cancel();self.controller.hold();self.window.clear();self.phase='PAUSED'

    def resume(self):
        if self.phase=='PAUSED' and not self.worker.active:
            self.worker.poll()  # Drain a completed, cancelled generation.
            self.capture_failures=0
            if self.exiting:self.exit()
            else:self._submit()

    def _submit(self):
        if self.current is None:self.exit();return
        raw=self.controller.planning_start();row=self.current
        self.phase='PLANNING';self.window.clear()
        self.worker.submit(lambda cancel:collection_plan(self.p,raw,row,self.settings,cancel),
                           (self.controller.motion_epoch,'pose',self.index))

    def exit(self):
        self.pause();self.exiting=True
        # A cancelled planner must finish before another uses Pinocchio Data.
        self.phase='HOME_PENDING'

    def next(self,reason=None):
        if reason:
            row=dict(index=self.index,pose=self.current,reason=reason)
            self.skipped.append(row)
            self.emit(dict(event='pose_skipped',**row))
        else:self.capture_failures=0
        self.index+=1;self._submit()

    def pump(self):
        result=self.worker.poll()
        if self.phase=='HOME_PENDING' and not self.worker.active:
            # Drain any cancelled result before submitting a new planning task.
            raw=self.controller.planning_start();self.phase='HOME_PLANNING'
            self.worker.submit(lambda cancel:self.p.plan_home(raw,cancel),
                               (self.controller.motion_epoch,'home',None));return
        if result:
            (epoch,kind,index),plan,error=result
            expected='HOME_PLANNING' if kind=='home' else 'PLANNING'
            if self.phase!=expected or epoch!=self.controller.motion_epoch:return
            if error:
                if kind=='home':
                    self.emit(dict(event='home_refused',error=str(error)));self.phase='PAUSED';return
                self.next('Planning refused: '+str(error));return
            self.controller.move(plan,expected_epoch=epoch)
            self.phase='HOMING' if kind=='home' else 'MOVING'
            self.emit(dict(event='move',kind=kind,index=index,duration_s=plan.duration,
                           target_base_m=plan.target.tolist()))
        completion=self.controller.take_completion()
        if completion and self.phase in ('MOVING','HOMING'):
            self.emit(dict(event='motion_completion',**completion))
            if self.phase=='HOMING':
                self.home_verified=completion['status']=='HOME_REACHED_HOLD'
                self.home_ok=completion['status'] in ('HOME_REACHED_HOLD','HOME_COMMAND_FINISHED_HOLD')
                self.phase='DONE' if self.home_ok else 'PAUSED';return
            if completion['status'] in ('SETTLED_MODEL_ONLY','SETTLE_TIMEOUT_HOLD'):
                # A stationary measured pose is useful even if commanded endpoint
                # has a small PD bias. Acceptance below uses RAW actual feedback.
                self.phase='CAPTURING';self.deadline=time.monotonic()+self.settings['observation_timeout_s']
                self.capture_started=time.monotonic()
                self.window.clear()
            else:self.pause()
        if self.phase=='CAPTURING' and time.monotonic()>self.deadline:
            reason='No acceptable stationary marker window: '+self.window.reason
            self.capture_failures+=1
            limit=self.settings['max_consecutive_capture_failures']
            if limit>0 and self.capture_failures>=limit:
                row=dict(index=self.index,pose=self.current,reason=reason)
                self.skipped.append(row);self.emit(dict(event='pose_skipped',**row))
                self.index+=1;self.pause()
                self.emit(dict(event='capture_paused',consecutive_failures=self.capture_failures,
                               reason='Check marker visibility, mount orientation, dictionary and sampling region'))
            else:self.next(reason)


def run(p,settings,rows,output,execute,resume=None):
    camera=frames=controller=sequence=None
    dataset=None;failure=None;exit_status='STOPPED';saved=0;cleanup_errors=[];camera_attempted=False
    mount=settings.get('mount','external')
    window_name=f'Automatic {"EIH" if mount=="wrist" else "ETH"} | s resume | h pause | q home and finish'
    window_open=False;diagnostic_frame=None;diagnostic_count=0
    def emit(event):
        nonlocal diagnostic_count
        if (event['event']=='pose_skipped' and event.get('reason','').startswith('No acceptable stationary')
                and diagnostic_frame is not None and diagnostic_count<12):
            try:
                directory=output/'rejected_frames';directory.mkdir(exist_ok=True)
                path=directory/f"pose_{event['index']:03d}_{uuid.uuid4().hex[:8]}.png"
                if not cv2.imwrite(str(path),diagnostic_frame.color):raise OSError('Could not save rejected frame')
                event['diagnostic_png']=str(path.relative_to(output))
                event['diagnostic_camera_frame']=diagnostic_frame.metadata
                diagnostic_count+=1
            except (OSError,cv2.error) as exc:event['diagnostic_error']=str(exc)
        with (output/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(unix_s=time.time(),**event),ensure_ascii=False,allow_nan=False)+'\n')
        if event['event']=='move':
            number='回零' if event['kind']=='home' else f"{event['index']+1}/{len(rows)}"
            print(f"[移动] {number}；{event['duration_s']:.1f} 秒",flush=True)
        elif event['event']=='pose_skipped':print('[跳过] '+event['reason'],flush=True)
        elif event['event']=='home_refused':print('[回零未执行] '+event['error'],flush=True)
        elif event['event']=='capture_paused':
            print(f"[暂停] 连续 {event['consecutive_failures']} 个位姿未采到有效码；检查画面/码板朝向。s继续，q回零。",flush=True)
        elif event['event']=='motion_completion' and event['status']=='HOME_TIMEOUT_HOLD':
            print('[保持] 回零未达标；s重试，h保持。',flush=True)
    try:
        robot_metadata=snapshot_robot(p,output)
        if resume:
            dataset=clone_resume(resume,output,robot_metadata)
            saved=len(dataset['samples'])
            emit(dict(event='resume',**dataset['acquisition']['resumed_from']))
            print(f"[补采] 已保留 {saved} 组；从姿态 {resume['start_index']+1}/{len(rows)} 继续，原目录不修改。",flush=True)
        camera_attempted=True
        camera=Camera('realsense',width=1280,height=720,fps=30,serial=settings['serial'])
        camera.metadata.update(fps=30,stream='color_only',image_size=[camera.width,camera.height])
        if hasattr(p,'check_validation_camera'):p.check_validation_camera(camera)
        frames=CollectionFrames(camera);frames.start()
        target=Target(settings['target'])
        cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE);window_open=True
        deadline=time.monotonic()+10.;frame=None
        while frame is None:
            cv2.imshow(window_name,np.zeros((720,1280,3),np.uint8))
            key,window_open=window_event(window_name,window_open)
            if key in (ord('q'),27) or not window_open:raise CollectionCancelled('Camera warmup cancelled')
            frame=frames.latest()
            if time.monotonic()>deadline:raise RuntimeError('No fresh RGB within 10 seconds; motors not enabled')
        print(f"[相机] {'腕部' if mount=='wrist' else '外置'} {settings['serial']}；1280×720，30 Hz，仅RGB。",flush=True)
        if mount=='wrist':print(f"[靶标] DICT_4X4_50，ID 0，黑色外框边长{settings['target']['marker_m']*1000:g} mm；固定在桌面，整个采集期间不移动。",flush=True)
        if settings.get('hardware_setup'):
            print('[布置] '+settings['hardware_setup'],flush=True)
        if settings.get('purpose')=='independent_low_pose_validation':
            print('[布置] 本轮60mm码平放并固定在裸桌面，不要架高；相机保持当前安装。',flush=True)
        if resume:check_resume_camera(dataset,camera)
        else:dataset=dict(schema_version=1,dataset_id=str(uuid.uuid4()),synthetic=False,
            target=settings['target'],image_size=[camera.width,camera.height],camera=camera.metadata,
            robot=robot_metadata,K=camera.k.tolist(),D=camera.d.tolist(),samples=[],
            acquisition=dict(mode='automatic_stationary',split=settings['split'],fps=30,
                config=settings['_path'],previous_calibration_used=False,
                planned_groups=dict(Counter(row['group'] for row in rows))))
        if mount=='wrist':
            dataset['acquisition'].update(camera_mount='wrist',target_fixed=True,
                calibration_mode='eye_in_hand')
            if 'validation_calibration' in p.robot_metadata:
                dataset['acquisition'].update(purpose='independent_low_pose_validation',
                    previous_calibration_used=True,previous_calibration_use='planning_only',
                    calibration_used_for_planning_only=p.robot_metadata['validation_calibration'],
                    saved_fk='nominal_RAW_angles_no_offsets',refit_performed=False)
        save(output/'dataset.json',dataset)
        if execute:
            def before_enable():
                nonlocal window_open
                key,window_open=window_event(window_name,window_open)
                if key in (ord('q'),27) or not window_open:
                    raise CollectionCancelled('Cancelled before enable')
                if frames.latest() is None:raise RuntimeError('No fresh camera before enable')
            controller=HoverController(p,before_enable=before_enable,startup_log=emit)
            controller.start()
            sequence=AutoSequence(p,controller,rows,settings,emit)
            if resume:sequence.index=resume['start_index']
            sequence.resume()
            print('[采集] 自动逐姿态运动、停稳、保存；不需逐次按Enter。h/空格暂停，s继续，q/Esc回零结束。',flush=True)
            if settings['max_consecutive_capture_failures']==0:
                print('[采集] 连续识别失败不暂停；单个姿态超时后自动跳过。',flush=True)
        else:print('[预览] 相机与码检测；没有打开CAN，没有使能电机。q退出。',flush=True)
        last=-1.
        while not sequence or sequence.phase!='DONE':
            key,visible=window_event(window_name,window_open)
            window_open=visible
            if not visible:time.sleep(.005)
            if controller:controller.heartbeat()
            if key in (ord('q'),27) or not visible:
                if not sequence:break
                if not sequence.exiting or (visible and sequence.phase=='PAUSED'):sequence.exit()
            elif sequence and key in (ord('h'),ord(' ')):
                sequence.pause();print('[保持] s继续；q回零结束。',flush=True)
            elif sequence and key==ord('s'):sequence.resume()
            if sequence:sequence.pump()
            if sequence and not visible and sequence.exiting and sequence.phase=='PAUSED':
                controller.hold('HOME_UNCONFIRMED_HOLD')
                failure='Window closed and home was not confirmed; holding for supported release'
                exit_status='HOME_UNCONFIRMED_HOLD'
                print('[保持] 窗口已关闭，回零未确认；托住机械臂后输入release退出。',flush=True)
                break
            frame=frames.latest()
            if frame is None:raise RuntimeError('No fresh RGB')
            if frame.acquired==last:continue
            last=frame.acquired;diagnostic_frame=frame;display=frame.color.copy();obs=None;label='Target not detected'
            # Acquire fresh feedback before expensive detection. The marker image
            # must lie inside the stationary feedback window, not predate arrival.
            meta=g=None
            capturing=(sequence and sequence.phase=='CAPTURING' and frame.acquired>=sequence.capture_started)
            if capturing:
                start=time.time();mono=time.monotonic()
                with controller.io:
                    all_q=controller.snapshot();raw=all_q[:6]
                    ages=(time.perf_counter()-controller.feedback.times).copy()
                    sequences=controller.feedback.sequences.copy()
                g=p.nominal_fk(raw)
                meta=dict(q_rad=raw.tolist(),read_started_unix_s=start,read_ended_unix_s=time.time(),
                    read_started_monotonic_s=mono,read_ended_monotonic_s=time.monotonic(),
                    feedback_age_s=ages[:6].tolist(),feedback_sequences=sequences[:6].tolist(),
                    gripper_q_rad=all_q[6:].tolist(),gripper_feedback_age_s=float(ages[6]),
                    frame_host_delta_s=time.time()-frame.metadata['host_received_unix_s'])
            try:
                obs=target.detect(frame.color,camera.k,camera.d,settings['max_pnp_px'],settings['ambiguity_gap_px'])
                label=f"PnP {obs['pnp_rms_px']:.2f}px | saved {saved}"
                cv2.polylines(display,[np.rint(obs['image_points']).astype(np.int32)],True,(0,230,255),2)
            except ValueError as exc:
                label=str(exc)
                if sequence:sequence.window.clear();sequence.window.reason=label
            if capturing and obs is not None:
                if sequence.window.add(frame,g,meta,obs):
                    try:sample=sample_from_window(sequence.window,dataset)
                    except ValueError as exc:sequence.next(str(exc))
                    else:
                        sample_id=str(uuid.uuid4());image=f"{saved:03d}_{sample_id[:8]}.png"
                        persist_sample(output,dataset,dict(id=sample_id,image=image,**sample,
                            pose_index=sequence.index,requested_pose=sequence.current),frame.color)
                        saved+=1
                        print(f"[保存] {saved} 组；姿态 {sequence.index+1}/{len(rows)}",flush=True)
                        sequence.next()
            if sequence:label+=' | '+sequence.phase
            if window_open:
                cv2.putText(display,label[:140],(15,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,230,255),1)
                cv2.imshow(window_name,display)
        if sequence and sequence.home_ok:
            exit_status='COMPLETED_HOME_HOLD'
            print(f'[完成] 保存 {saved} 组，已回零并保持。',flush=True)
    except KeyboardInterrupt:
        if controller and not controller.fault:controller.hold('USER_INTERRUPT_HOLD')
        exit_status='INTERRUPTED_HOLD'
        print('[中断] 已保持；已保存的数据保留。',flush=True)
    except CollectionCancelled:
        exit_status='CANCELLED_BEFORE_ENABLE'
        print('[取消] 未开始自动运动。',flush=True)
    except Exception as exc:
        failure=str(exc);exit_status='ERROR'
        if controller and controller.manual_disable_only and controller.enabled:
            controller.enter_fault_hold(exc)
        elif controller and not controller.fault:controller.hold('COLLECTION_ERROR_HOLD')
        print('[错误] '+failure+'；已保存数据保留，故障后不自动回零。',flush=True)
    finally:
        def attempt(label,operation):
            nonlocal failure
            try:operation()
            except BaseException as exc:
                detail=label+': '+(str(exc) or type(exc).__name__)
                cleanup_errors.append(detail);failure=failure or detail
        def reports(pending):
            if dataset:
                attempt('coverage report',lambda:save(output/'coverage_report.json',coverage(dataset)))
            attempt('end report',lambda:save(output/'collection_end.json',dict(
                status='ERROR' if failure else exit_status,error=failure,cleanup_pending=pending,
                cleanup_errors=cleanup_errors.copy(),sample_count=len(dataset['samples']) if dataset else 0,
                home_verified=bool(sequence and sequence.home_verified),
                home_motion_completed=bool(sequence and sequence.home_ok),hardware_accessed=camera_attempted,
                camera_opened=camera is not None,
                robot_open_attempted=controller is not None,camera=frames.stats() if frames else None,
                skipped_poses=sequence.skipped if sequence else [])))
        if sequence:attempt('cancel planner',sequence.worker.cancel)
        # Failures in reports, Qt or camera shutdown must never bypass robot cleanup.
        if frames:attempt('camera close',frames.close)
        elif camera:attempt('camera close',camera.close)
        attempt('window close',cv2.destroyAllWindows)
        if controller and controller.failure_diagnostics:
            attempt('feedback log',lambda:emit(dict(event='feedback_failure',diagnostics=controller.failure_diagnostics)))
        reports(True)  # Available while the terminal is waiting for release.
        if controller:
            try:attempt('supported release',lambda:release_supported(controller))
            finally:attempt('robot close',controller.close)
        reports(False)
        if cleanup_errors:
            print('[清理/报告] '+'; '.join(cleanup_errors),flush=True)
    if failure:return 1
    return 0


def main(argv=None, *, mount='external', project_root=None, config_path=None, after_collection=None,
         settings_loader=None,planner_type=None):
    project_root=ROOT if project_root is None else Path(project_root).resolve()
    parser=argparse.ArgumentParser(description=f'Automatic stationary {mount}-camera hand-eye acquisition')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true',help='Enable robot and automatically collect')
    mode.add_argument('--preview',action='store_true',help='RGB/ArUco only, no CAN')
    mode.add_argument('--plan-only',action='store_true',help='Offline SDK paths, no camera/CAN (default)')
    parser.add_argument('--config',default=str(config_path or ROOT/'auto_collection_config.json'))
    parser.add_argument('--split',choices=('train','validation'),default='train')
    parser.add_argument('--output')
    parser.add_argument('--resume-from',help='Read prior session and continue into a NEW output directory')
    parser.add_argument('--start-pose',type=int,help='1-based accepted pose number; requires --resume-from')
    args=parser.parse_args(argv)
    if args.start_pose is not None and not args.resume_from:parser.error('--start-pose requires --resume-from')
    if args.resume_from and args.preview:parser.error('--resume-from cannot be combined with --preview')
    cv2.setNumThreads(1)
    loader=settings_loader or load_settings
    settings,cfg,poses=loader(args.config,args.split,mount=mount)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output=Path(args.output).resolve() if args.output else project_root/'runs'/f'{args.split}_auto_{stamp}'
    if output.exists():parser.error('Output exists; choose a new directory')
    output.mkdir(parents=True)
    p=(planner_type or CollectionPlanner)(cfg)
    # Snapshot the exact independent mass settings and nominal model provenance.
    metadata=snapshot_robot(p,output)
    if hasattr(p,'snapshot_validation_result'):p.snapshot_validation_result(output)
    save(output/'session.json',dict(settings=settings,control_config=cfg,robot=metadata,
        hardware_requested=args.execute,environment=dict(python_executable=sys.executable,
            python=sys.version,opencv=cv2.__version__,numpy=np.__version__,pinocchio=p.pin.__version__,
            motorbridge=sys.modules['motorbridge'].__version__,opencv_threads=cv2.getNumThreads(),
            numeric_threads={name:os.environ.get(name) for name in
                ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}),
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(settings['_path']),Path(cfg['gravity_config']),
            Path(cfg['_path']),
            ROOT/'eyehand/auto_collection.py',ROOT/'eyehand/auto_collection_runtime.py',
            project_root/('scripts/collect_handeye_eih_auto.py' if mount=='wrist' else 'scripts/collect_handeye_eth_auto.py'),
            project_root/'wrist/auto_calibration.py' if mount=='wrist' else ROOT/'eyehand/auto_collection.py',
            ROOT/'runtime/motorbridge_contract.json',
            Path(cfg['sdk_root'])/'reBotArm_control_py/actuator/rebotarm.py',
            Path(cfg['sdk_root'])/'config/rebotarm_rs.yaml')}))
    if args.preview:return run(p,settings,poses,output,False)
    print('[检查] 离线计算官方SDK轨迹，不打开相机/CAN。',flush=True)
    report=preflight(p,poses,settings);save(output/'planned_poses.json',report)
    print(f"[计划] 可执行 {len(report['accepted'])}/{len(poses)} 个姿态；报告：{output/'planned_poses.json'}",flush=True)
    print(f"[负载] 腕部相机 {p.gravity['wrist_camera_mass_kg']*1000:g} g；"
          f"末端靶标 {p.gravity['payload_mass_kg']*1000:g} g。",flush=True)
    if p.gravity_exclusion['removed_mass_kg']:
        print(f"[重力] 已从私有动力学模型扣除已拆部件 {p.gravity_exclusion['removed_mass_kg']*1000:g} g；名义FK保持原模型。",flush=True)
    if settings.get('sampling_region'):
        print('[区域] 常规X范围(m)：'+str(settings['sampling_region']['normal_x_range_m'])+
              '；低位X范围(m)：'+str(settings['sampling_region']['low_x_range_m']),flush=True)
    if not report['accepted']:return 1
    rows=[dict(poses[r['index']],source_index=r['index']) for r in report['accepted']]
    resume=load_resume(args.resume_from,p,settings,rows,args.start_pose) if args.resume_from else None
    if resume:
        save(output/'resume_plan.json',dict(source=resume['source'],dataset_sha256=resume['dataset_sha256'],
            inherited_samples=len(resume['dataset']['samples']),start_pose=resume['start_index']+1,
            remaining_poses=len(rows)-resume['start_index']))
        print(f"[补采计划] 保留 {len(resume['dataset']['samples'])} 组；从 {resume['start_index']+1}/{len(rows)} 开始。",flush=True)
    if not args.execute:return 0
    code=run(p,settings,rows,output,True,resume=resume)
    if after_collection and code==0:after_collection(output,settings)
    return code

#!/usr/bin/env python3
"""Click RGB-D XY -> saved table + hover height; preview has NO CAN access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from eyehand.click_hover import HoverPlanner, depth_at_pixel, hover_target, load_config, in_workspace
from eyehand.table_reference import matching_table_reference, record_table_reference, table_hover_target, effective_hover_m
from eyehand.kinematics import measured_joints
from eyehand.hover_runtime import ClickGate, LatestFrames, PlanWorker, release_supported, RGBDFramePending


class AlignedRGBD:
    def __init__(self, cfg, calibration):
        import pyrealsense2 as rs
        self.rs, self.pipeline = rs, rs.pipeline()
        self.started = False
        config = rs.config()
        config.enable_device(cfg['serial'])
        fps = int(cfg.get('camera_fps', 30))
        if fps not in (15, 30):
            raise ValueError('RGB-D fps must be 15 or 30')
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, fps)
        try:
            profile = self.pipeline.start(config)
            self.started = True
            if profile.get_device().get_info(rs.camera_info.serial_number) != cfg['serial']:
                raise ValueError('Camera serial mismatch')
            self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            k = self.intrinsics
            self.K = np.array([[k.fx, 0, k.ppx], [0, k.fy, k.ppy], [0, 0, 1]])
            # Final calibration used this RGB stream and factory intrinsics.
            # Bind intrinsics to the selected result, not a hard-coded training folder.
            if calibration['camera']['serial'] != cfg['serial'] or calibration['image_size'] != [k.width, k.height]:
                raise ValueError('RGB profile differs from calibrated camera')
            if not np.allclose(self.K, calibration['K'], atol=.02, rtol=0):
                raise ValueError('RGB intrinsics differ from calibration; stop and investigate')
            if not np.isfinite(k.coeffs).all() or np.max(np.abs(k.coeffs)) > 1e-12 or np.max(np.abs(calibration['D'])) > 1e-12:
                raise ValueError('This bundle requires zero-distortion calibrated RGB intrinsics')
            self.scale = profile.get_device().first_depth_sensor().get_depth_scale()
            if not np.isfinite(self.scale) or self.scale <= 0:
                raise ValueError('Invalid RealSense depth scale')
            self.last_numbers = None
            self.last_timestamps = None
            self.align = rs.align(rs.stream.color)
            self.warmup_remaining = 20
            self.rejected_frames = 0
            self.last_rejection = None
        except BaseException:
            self.close()
            raise

    def _pending(self, reason):
        self.last_rejection = reason
        self.rejected_frames = getattr(self, 'rejected_frames', 0)+1
        raise RGBDFramePending(reason)

    def read(self):
        try:
            frames = self.pipeline.wait_for_frames(200)
        except RuntimeError as exc:
            if "Frame didn't arrive within" in str(exc):
                self._pending(f'Waiting for RGB-D: {exc}')
            raise
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            self._pending(f'Incomplete RGB-D frameset: color={bool(color)}, depth={bool(depth)}')
        domains = (color.get_frame_timestamp_domain(),depth.get_frame_timestamp_domain())
        numbers = (color.get_frame_number(), depth.get_frame_number())
        timestamps = (color.get_timestamp(), depth.get_timestamp())
        if not np.isfinite(timestamps).all():
            self._pending(f'Invalid RGB-D timestamps: {timestamps}')
        if domains[0] != domains[1]:
            self._pending(f'RGB-D timestamp domains differ: color={domains[0]}, depth={domains[1]}')
        skew = abs(timestamps[0]-timestamps[1])
        if skew > 40:
            self._pending(f'RGB-D skew={skew:.1f} ms > 40 ms; color={timestamps[0]:.1f}, depth={timestamps[1]:.1f}; numbers={numbers}')
        if self.last_numbers is not None and any(n <= old for n, old in zip(numbers, self.last_numbers)):
            self._pending(f'Repeated/out-of-order RGB-D frame number: {numbers}, last={self.last_numbers}')
        if self.last_timestamps is not None and any(n <= old for n, old in zip(timestamps, self.last_timestamps)):
            self._pending(f'Repeated/out-of-order RGB-D sensor timestamp: {timestamps}')
        aligned = self.align.process(frames)
        color,depth = aligned.get_color_frame(),aligned.get_depth_frame()
        if not color or not depth:
            self._pending(f'Incomplete aligned RGB-D: color={bool(color)}, depth={bool(depth)}')
        image = np.asanyarray(color.get_data()).copy()
        depth_m = np.asanyarray(depth.get_data()).astype(float)*self.scale
        if image.shape != (720,1280,3) or depth_m.shape != (720,1280):
            raise RuntimeError('Aligned RGB-D shape differs from calibrated stream')
        self.last_numbers, self.last_timestamps = numbers, timestamps
        remaining = getattr(self,'warmup_remaining',0)
        if remaining > 0:
            self.warmup_remaining = remaining-1
            self._pending(f'RGB-D warmup: {self.warmup_remaining} valid pairs remaining')
        self.last_rejection = None
        return image, depth_m

    def point(self, u, v, z):
        # Aligned depth pixel is in RGB image; deproject with RGB intrinsics.
        return np.array([(u-self.intrinsics.ppx)*z/self.intrinsics.fx,
                         (v-self.intrinsics.ppy)*z/self.intrinsics.fy, z])

    def close(self):
        if self.started:
            self.pipeline.stop()
            self.started = False


def append_log(path, event):
    with path.open('a') as stream:
        stream.write(json.dumps({'unix_s': time.time(), **event}, allow_nan=False)+'\n')


def click_summary(event):
    x, y, z = np.asarray(event['hover_base_m'])*1000
    text = '[移动]' if event.get('execution_requested') else '[预览]'
    text += f' 目标(mm)：X={x:.1f} Y={y:.1f} Z={z:.1f}'
    if 'duration_s' in event:
        text += f"；轨迹 {event['duration_s']:.1f} 秒"
    return text


def completion_summary(event):
    status = event['status']
    if status in ('HOME_REQUESTED_HOLD', 'HOME_REFUSED_HOLD'):
        return None  # The home-plan / rejection message carries the action.
    if status == 'SETTLED_MODEL_ONLY':
        return f"[到位] FK误差 {event['fk_error_mm']:.2f} mm；耗时 {event['elapsed_s']:.1f} 秒；已保持。"
    if status == 'HOME_REACHED_HOLD':
        return f"[回零完成] 最大关节误差 {event['home_joint_error_deg']:.2f}°；已保持。"
    if status == 'HOME_TIMEOUT_HOLD':
        return (f"[警告] 回零未达标，已保持；FK误差 {event['fk_error_mm']:.2f} mm；"
                f"最大关节误差 {event['home_joint_error_deg']:.2f}°。")
    if status == 'SETTLE_TIMEOUT_HOLD':
        return f"[警告] 到位超时，已保持；FK误差 {event['fk_error_mm']:.2f} mm。"
    if status == 'TRACKING_REFUSED_HOLD':
        reason = event.get('cartesian_outer_loop', {}).get('stop_reason') or status
        return f'[警告] 跟踪已停止并保持：{reason}'
    if status == 'UI_TIMEOUT_HOLD':
        return '[警告] 界面响应超时，已取消运动并保持。'
    if status in ('USER_HOLD', 'CANCELLED_HOLD'):
        return '[保持] 已取消运动，保持当前指令。'
    return f'[状态] {status}'


def save_motion_traces(controller, log):
    """Save completed/cancelled approach buffers on the GUI thread only."""
    for status, rows in controller.take_motion_traces():
        path = log.with_name(log.stem+'_approach_'+uuid.uuid4().hex[:8]+'.npz')
        np.savez_compressed(path, elapsed_s=rows[:, 0], settling=rows[:, 1],
            nominal_q_rad=rows[:, 2:8], command_q_rad=rows[:, 8:14],
            feedback_q_rad=rows[:, 14:20], reference_tcp_m=rows[:, 20:23],
            feedback_tcp_m=rows[:, 23:26], correction_velocity_rad_s=rows[:, 26:32],
            callback_interval_s=rows[:, 32], send_batch_duration_s=rows[:, 33])
        append_log(log, {'event': 'approach_trace', 'status': status, 'path': str(path),
                         'samples': len(rows), 'physical_accuracy_verified': False})


def project_base_pixel(planner, point):
    """OpenCV projection for display only; never changes a robot target."""
    T = np.asarray(planner.result['T_camera_base'], dtype=float)
    point = np.asarray(point, dtype=float)
    camera = T[:3,:3] @ point + T[:3,3]
    if not np.isfinite(camera).all() or camera[2] <= 0:
        return None
    projected, _ = cv2.projectPoints(point.reshape(1,1,3), cv2.Rodrigues(T[:3,:3])[0], T[:3,3],
                                     np.asarray(planner.calibration['K']),
                                     np.asarray(planner.calibration['D']))
    return projected.reshape(2).tolist()


def draw_clicked_point(image, surface_pixel, hover_pixel=None):
    """Yellow surface click and cyan projected hover goal."""
    if surface_pixel is not None:
        pixel = tuple(int(round(value)) for value in surface_pixel)
        cv2.drawMarker(image, pixel, (0,255,255), cv2.MARKER_CROSS, 20, 2)
    if hover_pixel is not None and 0 <= hover_pixel[0] < image.shape[1] and 0 <= hover_pixel[1] < image.shape[0]:
        pixel = tuple(int(round(value)) for value in hover_pixel)
        cv2.drawMarker(image, pixel, (255,255,0), cv2.MARKER_DIAMOND, 22, 2)
        cv2.putText(image, "TCP goal", (pixel[0]+12,pixel[1]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,0), 1)



def offline_check(planner, output):
    """Replay encoder states and synthesize nearby position-only goals; no camera/CAN."""
    dataset = json.loads((ROOT/'runs/final_eye_to_hand_2026-11-20/combined_dataset.json').read_text())
    cases = []
    zero_cases = []
    for sample in dataset['samples']:
        raw = measured_joints(sample)
        target = planner.tcp(raw) + [.005, 0, .005]
        try:
            plan = planner.plan(raw, target)
            cases.append({'id': sample['id'], 'success': True,
                          'endpoint_error_mm': float(np.linalg.norm(planner.tcp(plan.goal)-target)*1000),
                          'duration_s': plan.duration})
        except ValueError as exc:
            cases.append({'id': sample['id'], 'success': False, 'reason': str(exc)})
        _, hover = hover_target(planner.T_base_camera, np.array(sample['T_camera_target'])[:3,3], effective_hover_m(planner.cfg))
        try:
            outward = planner.plan(np.zeros(6), hover)
            home = planner.plan_home(outward.goal)
            zero_cases.append({'id':sample['id'],'success':True,'hover_base_m':hover.tolist(),
                               'out_duration_s':outward.duration,'home_duration_s':home.duration,
                               'home_goal_raw_rad':home.goal.tolist(),
                               'endpoint_model_error_mm':float(np.linalg.norm(planner.tcp(outward.goal)-hover)*1000)})
        except ValueError as exc:
            zero_cases.append({'id':sample['id'],'success':False,'reason':str(exc)})
    # Synthetic bare table plus raised objects exercise the ACTIVE height policy.
    # Legacy recorded-marker cases above still test kinematic reachability only.
    table_z = 0.
    table_cases = []
    T_camera_base = np.linalg.inv(planner.T_base_camera)
    for object_z in (0., .04, .07):
        surface = np.array([.4, 0., object_z])
        point = (T_camera_base @ np.r_[surface, 1.])[:3]
        try:
            base, target = table_hover_target(planner.T_base_camera, point, effective_hover_m(planner.cfg), table_z_m=table_z)
            plan = planner.plan(np.zeros(6), target)
            home = planner.plan_home(plan.goal)
            table_cases.append({'success': True, 'surface_base_m': base.tolist(), 'target_base_m': target.tolist(),
                                'table_z_m': table_z, 'height_above_table_m': float(target[2]-table_z),
                                'home_raw_goal_rad': home.goal.tolist()})
        except ValueError as exc:
            table_cases.append({'success': False, 'reason': str(exc)})
    report = {'hardware_access': False, 'physical_accuracy_verified': False,
              'height_reference': 'table', 'synthetic_table_z_m': table_z, 'table_height_cases': table_cases,
              'legacy_marker_cases_note': 'Recorded-marker cases exercise kinematic reachability, not the active table height policy',

              'tcp_offset_m': planner.cfg['tcp_offset_m'], 'result_sha256': planner.result_sha256, 'cases': cases, 'zero_to_hover_and_home':zero_cases}
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if all(c['success'] for c in cases+zero_cases+table_cases) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'click_hover_config.json'))
    parser.add_argument('--execute', action='store_true', help='enable real robot MIT hold and click motion')
    parser.add_argument('--q-seed', type=float, nargs=6, help='preview IK with six RAW encoder angles, radians; no CAN')
    parser.add_argument('--offline-check', metavar='NEW_REPORT_JSON', help='no camera or CAN; replay planning checks')
    parser.add_argument('--outer-loop', choices=('tracking', 'settle', 'off'),
                        help='Temporary comparison: tracking=XYZ throughout, settle=XYZ after SDK path, off=SDK path only')
    args = parser.parse_args()
    if args.execute and (args.q_seed is not None or args.offline_check):
        parser.error('--execute cannot be combined with --q-seed or --offline-check')
    cfg = load_config(args.config)
    if args.outer_loop is not None:
        cfg['cartesian_outer_loop']['enabled'] = args.outer_loop != 'off'
        cfg['cartesian_outer_loop']['during_motion'] = args.outer_loop == 'tracking'
    planner = HoverPlanner(cfg)
    if args.offline_check:
        return offline_check(planner, args.offline_check)
    logs = Path(cfg['log_dir'])
    logs.mkdir(parents=True, exist_ok=True)
    log = logs/f'{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}.jsonl'
    append_log(log, {'event': 'session', 'execute': args.execute, 'config': cfg,
                     'result_sha256': planner.result_sha256})
    camera = controller = frames = None
    worker = PlanWorker()
    home_requested = home_planning = False
    gate = ClickGate(lambda: home_requested or home_planning or worker.active or not worker.results.empty() or bool(controller and controller.busy))
    window = 'External RGB | t table | click TCP | h hold | q/Esc home | x exit'
    shown = None
    last_pixel = None
    last_hover_pixel = None
    message = 'PREVIEW: camera only, no CAN' if not args.execute else 'EXECUTE: click surface for TCP target'
    saved_table = matching_table_reference(cfg, planner.result_sha256)
    table_z = saved_table['table_z_m'] if saved_table else None
    table_id = saved_table.get('table_reference_id') if saved_table else None
    selecting_table = saved_table is None
    if saved_table:
        append_log(log, {'event': 'table_reference_reused', **saved_table,
                         'target_z_m': table_z+effective_hover_m(cfg), 'motion_requested': False})
    cleanup_errors = []

    def cancel_motion(reason='CANCELLED_HOLD'):
        nonlocal home_requested, home_planning
        home_requested = home_planning = False
        gate.clear()
        worker.cancel()
        if controller:
            controller.hold(reason)

    def record_click(event, plan=None, epoch=None):
        nonlocal message, last_pixel, last_hover_pixel
        event['hover_target_pixel'] = project_base_pixel(planner, event['hover_base_m'])
        if plan is not None:
            event.update({'raw_goal_rad': plan.goal.tolist(), 'duration_s': plan.duration, 'ik_position_only': True, 'path_mode':plan.path_mode})
        # Persist the plan before execution; failed logging cannot launch a move.
        if plan is not None and hasattr(plan, 'diagnostics'):
            event['sdk_trajectory'] = plan.diagnostics
            event['ik_position_only'] = True
        event['execution_requested'] = bool(controller and plan is not None)
        append_log(log, event)
        if controller and plan is not None:
            controller.move(plan, expected_epoch=epoch)
            append_log(log, {'event': 'motion_submitted', 'pixel': event['pixel'], 'duration_s': plan.duration})
        print(click_summary(event), flush=True)
        last_pixel = tuple(event['pixel'])
        last_hover_pixel = event['hover_target_pixel']
        message = ('MOVING' if controller else 'PREVIEW') + ' target mm: ' + str(np.round(np.array(event['hover_base_m'])*1000, 1))

    def rejected(exc):
        nonlocal message
        message = str(exc)
        print(f'[未执行] {message}', flush=True)
        append_log(log, {'event': 'rejected', 'reason': message})

    try:
        print('[启动] 正在连接外置相机…', flush=True)
        camera = AlignedRGBD(cfg, planner.calibration)
        frames = LatestFrames(camera)
        frames.start()
        # GUI and a complete RGB-D frame must work before enabling any motor.
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        deadline = time.monotonic()+10
        while True:
            initial = frames.latest(cfg['camera_max_age_s'])
            if initial is not None:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f'No synchronized RGB-D within 10 s; motors not enabled. Last issue: {frames.pending_reason}')
            if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                return 0
            time.sleep(.005)
        cv2.imshow(window, initial.color)
        if cv2.waitKey(1) & 0xff in (ord('q'), 27):
            return 0
        cv2.setMouseCallback(window, lambda event, x, y, flags, param:
                             gate.offer(x, y, shown) if event == cv2.EVENT_LBUTTONDOWN else None)
        if args.execute:
            from eyehand.hover_control import HoverController
            print('[启动] 相机就绪，正在接管机械臂；请先停止其他控制器。', flush=True)
            controller = HoverController(planner, before_enable=lambda: frames.latest(cfg['camera_max_age_s']),
                                         startup_log=lambda event: append_log(log, event))
            controller.start()
            print(f'[就绪] 已保持当前姿态；桌面悬停高度 {effective_hover_m(cfg)*1000:g} mm。', flush=True)
            print('[操作] 左键移动 | h/空格保持 | q/Esc回零并保持 | t重选桌面 | x退出', flush=True)
        else:
            print('[预览] 仅相机预览，未控制机械臂；左键查看目标，t重选桌面，q/Esc/x退出。', flush=True)
        if selecting_table:
            message = 'CLICK BARE TABLE: record height once; no motion'
            print('[桌面] 请先点一次裸桌面，只记录高度，不移动。', flush=True)
        else:
            message = f"TABLE Z={table_z*1000:.1f} mm; fixed TCP Z={(table_z+effective_hover_m(cfg))*1000:.1f} mm"
        print(f'[日志] {log}', flush=True)
        while True:
            if controller:
                controller.heartbeat()
            frame = frames.latest(cfg['camera_max_age_s'])
            image = frame.color.copy()
            draw_clicked_point(image, last_pixel, last_hover_pixel)
            cv2.putText(image, message[:140], (10,25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,255), 1)
            shown = frame
            cv2.imshow(window, image)
            # Process stop/exit before any click or completed planning result.
            key = cv2.waitKey(5) & 0xff
            if key == ord('x') or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                cancel_motion()
                break
            if key in (ord('q'), 27):
                if not controller:
                    cancel_motion()
                    break
                if home_requested or home_planning or controller.motion_kind == 'home':
                    message = 'RETURNING HOME: h/Space stops; keep holding after raw q=0'
                    continue
                cancel_motion('HOME_REQUESTED_HOLD')
                home_requested = True
                message = 'HOME REQUESTED: cancelling previous motion/IK; will return RAW joints 1..6 to zero'
                continue
            if key in (ord('h'), ord(' ')):
                cancel_motion()
                message = 'HOLD: pending clicks/plans cancelled; holding last command'
                continue
            if key == ord('t'):
                if (home_requested or home_planning or worker.active or not worker.results.empty()
                        or bool(controller and controller.busy)):
                    rejected(ValueError('Wait for motion/planning to finish before selecting table'))
                    continue
                gate.clear()
                selecting_table = True
                last_pixel = last_hover_pixel = None
                message = 'CLICK BARE TABLE: save height; no motion'
                append_log(log, {'event': 'table_reference_requested'})
                continue
            if controller:
                completion = controller.take_completion()
                if completion:
                    if completion['status'] == 'SETTLED_MODEL_ONLY' and 'tcp_base_m' in completion:
                        completion['fk_tcp_pixel'] = project_base_pixel(planner, completion['tcp_base_m'])
                    append_log(log, {'event': 'completion', **completion})
                    summary = completion_summary(completion)
                    if summary:
                        print(summary, flush=True)
                    message = completion['status']
                save_motion_traces(controller, log)
            outcome = worker.poll()
            if outcome is not None:
                context, plan, error = outcome
                is_home = context.get('kind') == 'home'
                if is_home:
                    home_planning = False
                try:
                    if error:
                        raise error
                    if is_home:
                        event = {'event':'home_plan','raw_goal_rad':plan.goal.tolist(),'duration_s':plan.duration,
                                 'sdk_transit':plan.diagnostics}
                        append_log(log,event)
                        controller.move(plan,expected_epoch=context['epoch'])
                        message = 'RETURNING HOME: raw joints -> zero, then stay enabled and hold'
                        print(f'[回零] 轨迹 {plan.duration:.1f} 秒；结束后保持。', flush=True)
                    else:
                        record_click(context['event'], plan, context['epoch'])
                except (ValueError, RuntimeError, InterruptedError) as exc:
                    if is_home:
                        controller.hold('HOME_REFUSED_HOLD')
                    rejected(exc)
            if home_requested and not worker.active and worker.results.empty():
                try:
                    seed = controller.planning_start()
                    epoch = controller.motion_epoch
                    worker.submit(lambda cancelled,q=seed.copy(): planner.plan_home(q,cancelled=cancelled),
                                  {'kind':'home','epoch':epoch})
                    home_requested, home_planning = False, True
                    message = 'PLANNING HOME: h/Space cancels; no new clicks during return'
                except (ValueError,RuntimeError) as exc:
                    home_requested = home_planning = False
                    controller.hold('HOME_REFUSED_HOLD')
                    rejected(exc)
            click = gate.take()
            if click is not None:
                u, v, captured = click
                try:
                    if controller and controller.busy:
                        raise ValueError('Moving/settling: click ignored')
                    if not 0 <= time.monotonic()-captured.acquired <= cfg['camera_max_age_s']:
                        raise ValueError('Displayed frame too old; click again')
                    z, spread = depth_at_pixel(captured.depth, u, v)
                    color_point = camera.point(u, v, z)
                    surface, _ = hover_target(planner.T_base_camera, color_point, effective_hover_m(cfg))
                    event = {'event': 'click', 'pixel': [u,v], 'depth_m': z, 'depth_patch_spread_m': spread,
                             'point_color_optical_m': color_point.tolist(), 'surface_base_m': surface.tolist()}
                    if selecting_table:
                        saved = record_table_reference(cfg, surface, pixel=[u,v], source_log=log,
                                                       calibration_sha256=planner.result_sha256)
                        append_log(log, {'event': 'table_reference', **saved,
                                         'target_z_m': saved['table_z_m']+effective_hover_m(cfg),
                                         'motion_requested': False})
                        table_z, table_id = saved['table_z_m'], saved['table_reference_id']
                        selecting_table = False
                        last_pixel, last_hover_pixel = (u,v), None
                        message = f"TABLE Z={table_z*1000:.1f} mm; fixed TCP Z={(table_z+effective_hover_m(cfg))*1000:.1f} mm"
                        print(f'[桌面] 高度已保存；目标距桌面 {effective_hover_m(cfg)*1000:g} mm。下一次点击移动。', flush=True)
                        continue
                    surface, target = table_hover_target(planner.T_base_camera, color_point, effective_hover_m(cfg),
                                                         table_z_m=table_z)
                    if not in_workspace(target, cfg):
                        raise ValueError('Hover target outside configured base workspace')
                    event.update({'hover_base_m': target.tolist(), 'tcp_goal_base_m': target.tolist(),
                                  'height_reference': 'table', 'table_reference_id': table_id,
                                  'table_z_m': table_z, 'height_above_table_m': effective_hover_m(cfg),
                                  'requested_hover_m': cfg['hover_m'], 'height_compensation_m': cfg['height_compensation_m'],
                                  'height_offset_m': float(target[2]-surface[2])})
                    epoch = controller.motion_epoch if controller else None
                    seed = controller.planning_start() if controller else args.q_seed
                    if seed is None:
                        record_click(event)
                    else:
                        event['raw_start_rad'] = list(seed)
                        worker.submit(lambda cancelled, q=np.array(seed), t=target.copy():
                                      planner.plan(q, t, cancelled=cancelled), {'event': event, 'epoch': epoch})
                        message = 'PLANNING: h/Space cancels, q/Esc returns to zero'
                except (ValueError, RuntimeError) as exc:
                    rejected(exc)
    except KeyboardInterrupt:
        print('[退出] 已取消运动，机械臂仍保持。' if controller and controller.enabled and not controller.fault
              else '[退出] 已中断会话。', flush=True)
    finally:
        cancel_motion()
        # Independent cleanup: a camera/window/input error must not skip motor shutdown.
        try:
            if frames:
                frames.close()
            elif camera:
                camera.close()
        except Exception as exc:
            cleanup_errors.append(f'camera close: {exc}')
        try:
            cv2.destroyAllWindows()
        except Exception as exc:
            cleanup_errors.append(f'window close: {exc}')
        try:
            if controller:
                release_supported(controller)
        finally:
            if controller:
                try:
                    controller.close()
                except Exception as exc:
                    cleanup_errors.append(f'robot close: {exc}')
                try:
                    save_motion_traces(controller, log)
                except Exception as exc:
                    cleanup_errors.append(f'motion trace: {exc}')
        if cleanup_errors:
            print('Cleanup errors: '+'; '.join(cleanup_errors), file=sys.stderr, flush=True)
    return 1 if cleanup_errors else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(1)

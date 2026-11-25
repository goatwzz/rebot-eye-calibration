#!/usr/bin/env python3
"""Wrist RGB-D: ready -> click approach -> r returns ready -> repeat; exit RAW zero."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wrist.click_hover import WristPlanner, load_config, freeze_click, save_table_reference
from wrist.demo_flow import DemoFlow, StationaryWindow, WristFrame, click_pose
from eyehand.click_hover import depth_at_pixel, in_workspace, hover_target
from eyehand.table_reference import effective_hover_m
from eyehand.hover_control import HoverController
from eyehand.hover_runtime import LatestFrames, ClickGate, release_supported
# Reuse the same RGB-D alignment/startup and on-screen markers as the external demo.
sys.path.insert(0, str(ROOT.parent / 'rebot_eye_to_hand/scripts'))
from click_hover_eth import AlignedRGBD, append_log, draw_clicked_point, save_motion_traces
import cv2
import numpy as np


def offline_check(planner, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    ready = planner.plan_ready(np.zeros(6))
    table_z = .005  # Synthetic reference for this offline check only.
    rows = []
    for surface in ([.4, 0., table_z], [.4, .1, table_z+.05], [.45, -.1, table_z+.0815]):
        T = planner.camera_pose(ready.goal)
        camera_point = (np.linalg.inv(T) @ np.r_[surface, 1.])[:3]
        base, target = freeze_click(planner, camera_point, ready.goal, effective_hover_m(planner.cfg), table_z_m=table_z)
        assert np.allclose(base, surface)
        assert np.isclose(target[2], table_z+effective_hover_m(planner.cfg))
        hover = planner.plan(ready.goal, target)
        returned = planner.plan_ready(hover.goal)
        home = planner.plan_home(hover.goal)
        rows.append({'surface_base_m': base.tolist(), 'hover_base_m': target.tolist(),
                     'hover_goal_rad': hover.goal.tolist(), 'home_goal_rad': home.goal.tolist(),
                     'return_ready_goal_rad': returned.goal.tolist(),
                     'return_ready_error_mm': float(np.linalg.norm(planner.tcp(returned.goal)-ready.target)*1000),
                     'fk_hover_error_mm': float(np.linalg.norm(planner.tcp(hover.goal)-target)*1000),
                     'sdk_approach': hover.diagnostics, 'return_ready_duration_s': returned.duration,
                     'durations_s': [ready.duration, hover.duration, home.duration]})
    result = {'hardware_accessed': False, 'physical_accuracy_verified': False,
              'repeat_cycle': 'WAIT_CLICK -> hover -> HOVER_HOLD_PRESS_R -> r -> ready -> WAIT_CLICK',
              'ready_pose_source': 'config/click_hover.json', 'ready_pose': planner.cfg['ready_pose'],
              'calibration_sha256': planner.result_sha256, 'joint_offsets_rad': planner.offsets.tolist(),
              'height_reference': 'table', 'synthetic_table_z_m': table_z,
              'ready_goal_rad': ready.goal.tolist(), 'cases': rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    return 0


def run(cfg, planner, execute):
    logdir = Path(cfg['log_dir']); logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f'{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}.jsonl'
    def emit(event):
        append_log(log, event)
        print(json.dumps(event, ensure_ascii=False), flush=True)
    emit({'event': 'session', 'execute': execute, 'config': cfg, 'result_sha256': planner.result_sha256,
          'T_gripper_camera': planner.T_gripper_camera.tolist(), 'joint_offsets_rad': [0.]*6,
          'base_compensation_m': [0.]*3})
    camera = frames = controller = flow = None
    shown = None
    stationary = StationaryWindow(cfg['click_stationary_s'], cfg['click_joint_span_rad'])
    stationary_ok = False
    selected_surface = selected_hover = None
    saved_table = cfg.get('table_reference') or {}
    reuse_table = (saved_table.get('camera_serial') == cfg['serial']
                   and saved_table.get('calibration_sha256') == planner.result_sha256)
    table_z = saved_table['table_z_m'] if reuse_table else None
    table_reference_id = saved_table.get('table_reference_id') if reuse_table else None
    selecting_table = not reuse_table
    if reuse_table:
        emit({'event': 'table_reference_reused', **saved_table,
              'target_z_m': table_z+effective_hover_m(cfg), 'motion_requested': False})
    window = 'Wrist RGB | t table | r ready | h hold | q/Esc exit home'
    window_open = True
    gate = ClickGate(lambda: not stationary_ok or not (flow and flow.can_click))
    errors = []
    exit_code = 0

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or gate.offer(x, y, shown):
            return
        phase = flow.phase if flow else 'PREVIEW_NO_ROBOT'
        if flow is None:
            reason, hint = 'preview_no_robot', '预览模式未控制机械臂。'
        elif phase in ('HOVER_HOLD_PRESS_R', 'HOLD_PRESS_R', 'HOLD_TIMEOUT_PRESS_R'):
            reason, hint = 'return_ready_required', '请先按 r 返回初始位姿并等待到位。'
        elif not flow.can_click:
            reason, hint = 'motion_or_ready_check_pending', '运动或初始位姿到位检查尚未完成。'
        elif not stationary_ok:
            reason, hint = 'joints_not_stationary', '请等待关节连续稳定 0.5 秒。'
        else:
            reason, hint = 'frame_or_click_pending', '画面尚未就绪，或已有点击等待处理。'
        append_log(log, {'event': 'click_blocked', 'pixel': [x, y], 'phase': phase,
                         'reason': reason, 'stationary_ok': bool(stationary_ok)})
        print(f'[点击未执行] {hint} 当前状态：{phase}', flush=True)

    try:
        camera = AlignedRGBD(cfg, planner.calibration)
        frames = LatestFrames(camera); frames.start()
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        print('Waiting for wrist RGB-D; ArUco is not required at zero. Robot not enabled yet.', flush=True)
        deadline = time.monotonic()+10.
        initial = None
        while initial is None:
            initial = frames.latest(cfg['camera_max_age_s'])
            if time.monotonic() > deadline:
                raise RuntimeError('No synchronized wrist RGB-D within 10 seconds')
            if cv2.waitKey(5) & 0xff in (ord('q'), 27):
                return 0
            time.sleep(.005)
        cv2.imshow(window, initial.color)
        cv2.setMouseCallback(window, on_mouse)
        if execute:
            print('Stop other controllers. Checking robot feedback before enable; no observation motion has started.', flush=True)
            controller = HoverController(planner, before_enable=lambda: frames.latest(cfg['camera_max_age_s']))
            controller.start()
            print(f'Robot enabled and holding startup pose. SDK observation trajectory: '
                  f"{cfg['ready_pose']['duration']*cfg['transit']['time_scale']:.2f} s; "
                  f"home time scale max_vel={cfg['transit']['home_max_vel_rad_s']:.2f} rad/s.", flush=True)
            flow = DemoFlow(planner, controller, emit)
            flow.ready()
        else:
            print('PREVIEW ONLY: no CAN, no ready/home motion. Use --execute for the full sequence.', flush=True)
        if selecting_table:
            print('No matching saved table reference. First stationary left click selects BARE TABLE height, no motion.', flush=True)
        else:
            print(f'Using saved table Z={table_z*1000:.3f} mm; all targets Z={(table_z+effective_hover_m(cfg))*1000:.3f} mm. Left click plans motion after ready.', flush=True)
        print('每次点击运动后，先按 r 返回初始位姿，到位并停稳后才能再次点击。', flush=True)
        print('t = reselect bare table; r = return initial pose; h/Space = hold; q/Esc/x/close = home then exit.', flush=True)
        print('Table is assumed parallel to base XY; height is measured along base +Z.', flush=True)
        print('Successful home disables control and exits; failed/cancelled home holds for supported release.', flush=True)
        while True:
            try:
                raw = None
                if controller:
                    controller.heartbeat()
                    raw = controller.snapshot()[:6]
                    previous_phase = flow.phase
                    flow.pump()
                    save_motion_traces(controller, log)
                    if flow.phase != previous_phase:
                        if flow.phase == 'WAIT_CLICK':
                            print('[就绪] 已到初始位姿；停稳后可左键点击。', flush=True)
                        elif flow.phase == 'HOVER_HOLD_PRESS_R':
                            print('[到位] 已保持；按 r 返回初始位姿后，才能再次点击。', flush=True)
                        elif flow.phase in ('HOLD_PRESS_R', 'HOLD_TIMEOUT_PRESS_R'):
                            print('[保持] 本次运动已停止；按 r 返回初始位姿后再点击。', flush=True)
                    if flow.done:
                        if not flow.home_confirmed:
                            exit_code = 1
                        break
                    stationary_ok = stationary.update(raw, time.monotonic()) and flow.can_click
                if flow and flow.exiting:
                    gate.clear()
                    # Homing depends only on fresh robot feedback, not on the camera remaining visible.
                    if window_open:
                        key = cv2.waitKey(5) & 0xff
                        if key in (ord('h'), ord(' ')):
                            flow.hold()
                    time.sleep(.01)
                    continue
                try:
                    frame = frames.latest(cfg['camera_max_age_s'])
                except RuntimeError as exc:
                    if not flow:
                        raise
                    flow.exit('CAMERA_ERROR: '+str(exc))
                    continue
                if frame is None:
                    time.sleep(.005)
                    continue
                image = frame.color.copy()
                if raw is not None:
                    surface_uv = planner.project_base(selected_surface, raw) if selected_surface is not None else None
                    hover_uv = planner.project_base(selected_hover, raw) if selected_hover is not None else None
                    draw_clicked_point(image, surface_uv, hover_uv)
                phase = flow.phase if flow else 'PREVIEW_NO_ROBOT'
                if not execute:
                    action = ' | preview only'
                elif selecting_table:
                    action = ' | CLICK BARE TABLE: no motion'
                else:
                    action = f' | target Z = {(table_z+effective_hover_m(cfg))*1000:.1f} mm (table +{effective_hover_m(cfg)*1000:.0f})'
                if flow and flow.phase in ('HOVER_HOLD_PRESS_R', 'HOLD_PRESS_R', 'HOLD_TIMEOUT_PRESS_R'):
                    label = phase+' | press r: return to initial pose before next click'
                else:
                    label = phase + (action if stationary_ok or not execute else ' | wait stationary')
                cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1)
                shown = WristFrame(frame.color, frame.depth, frame.acquired,
                                   None if raw is None else raw.copy(), controller.motion_epoch if controller else -1)
                cv2.imshow(window, image)
                key = cv2.waitKey(5) & 0xff
                window_open = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
                if key in (ord('q'), 27, ord('x')) or not window_open:
                    gate.clear()
                    if flow:
                        flow.exit('USER_EXIT')
                        continue
                    break
                if flow and key in (ord('h'), ord(' ')):
                    gate.clear(); flow.hold(); stationary.since = time.monotonic()
                    continue
                if flow and key == ord('r'):
                    gate.clear()
                    if flow.ready():
                        stationary = StationaryWindow(cfg['click_stationary_s'], cfg['click_joint_span_rad'])
                        stationary_ok = False
                        selected_surface = selected_hover = None
                        print('[返回] 正在规划返回初始位姿。', flush=True)
                    continue
                if flow and key == ord('t'):
                    if stationary_ok and flow.can_click:
                        gate.clear()
                        selecting_table = True
                        selected_surface = selected_hover = None
                        emit({'event': 'table_reference_requested'})
                        print('Click BARE TABLE to replace the height reference; this click does not move the robot.', flush=True)
                    continue
                click = gate.take()
                if click is not None:
                    try:
                        u, v, captured = click
                        q = click_pose(captured, controller, cfg['camera_max_age_s'], cfg['click_joint_span_rad'])
                        z, spread = depth_at_pixel(captured.depth, u, v)
                        point = camera.point(u, v, z)
                        surface, _ = hover_target(planner.camera_pose(q), point, effective_hover_m(cfg))
                        event = {'pixel': [u, v], 'depth_m': z, 'depth_patch_spread_m': spread,
                                 'raw_q_at_image_rad': q.tolist(), 'T_base_camera_at_click': planner.camera_pose(q).tolist(),
                                 'point_color_optical_m': point.tolist(), 'surface_base_m': surface.tolist()}
                        if selecting_table:
                            height = float(surface[2])
                            if not cfg['workspace_min_m'][2] <= height+effective_hover_m(cfg) <= cfg['workspace_max_m'][2]:
                                raise ValueError('Table + hover height is outside workspace Z limits')
                            reference_id = uuid.uuid4().hex
                            saved = {'table_z_m': height, 'table_reference_id': reference_id,
                                     'source_log': str(log), 'source_pixel': [u, v],
                                     'camera_serial': cfg['serial'], 'calibration_sha256': planner.result_sha256,
                                     'table_assumption': 'parallel_to_base_XY'}
                            save_table_reference(cfg, saved)
                            emit({'event': 'table_reference', **event, 'table_reference_id': reference_id,
                                  'table_z_m': height, 'target_z_m': height+effective_hover_m(cfg),
                                  'table_assumption': 'parallel_to_base_XY', 'saved_to_config': True,
                                  'motion_requested': False})
                            table_z, table_reference_id = height, reference_id
                            selecting_table = False
                            selected_surface, selected_hover = surface.copy(), None
                            print(f'Table reference set: Z={1000*table_z:.1f} mm; all click targets Z={1000*(table_z+effective_hover_m(cfg)):.1f} mm. Next left click plans motion.', flush=True)
                            continue
                        surface, target = freeze_click(planner, point, q, effective_hover_m(cfg), table_z_m=table_z)
                        if not in_workspace(target, cfg):
                            raise ValueError('Hover target outside workspace')
                        event.update({'hover_base_m': target.tolist(), 'height_reference': 'table',
                                      'table_reference_id': table_reference_id, 'table_z_m': table_z,
                                      'height_above_table_m': effective_hover_m(cfg),
                                      'requested_hover_m': cfg['hover_m'], 'height_compensation_m': cfg['height_compensation_m'],
                                      'height_offset_m': float(target[2]-surface[2])})
                        emit({'event': 'click', **event})
                        flow.hover(target, event)
                        selected_surface, selected_hover = surface.copy(), target.copy()
                    except (ValueError, RuntimeError) as exc:
                        emit({'event': 'click_rejected', 'error': str(exc)})
            except KeyboardInterrupt:
                gate.clear()
                if flow:
                    if flow.exiting:
                        flow.hold()  # A second interrupt cancels automatic home.
                    else:
                        flow.exit('CTRL_C')
                else:
                    break
    except BaseException as exc:
        exit_code = 1
        if flow:
            flow.fail(str(exc))
        print(f'Error: {exc}; automatic home is not claimed after a robot/control fault.', file=sys.stderr, flush=True)
    finally:
        gate.clear()
        if flow:
            flow.worker.cancel()
        if controller:
            try:
                save_motion_traces(controller, log)
            except Exception as exc:
                errors.append('trace save: '+str(exc))
        # Normal exit closes only after measured home arrival; failed home needs supported release.
        try:
            if controller and not (flow and flow.home_confirmed):
                release_supported(controller)
        except BaseException as exc:
            errors.append('release: '+str(exc))
        finally:
            if controller:
                try:
                    controller.close()
                except BaseException as exc:
                    errors.append('robot close: '+str(exc))
        try:
            if frames:
                frames.close()
            elif camera:
                camera.close()
        except BaseException as exc:
            errors.append('camera close: '+str(exc))
        try:
            cv2.destroyAllWindows()
        except Exception as exc:
            errors.append('window close: '+str(exc))
        if errors:
            print('Cleanup errors: '+'; '.join(errors), file=sys.stderr, flush=True)
    return max(exit_code, int(bool(errors)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'config/click_hover.json'))
    parser.add_argument('--execute', action='store_true', help='Automatically move ready, click-hover, and return home on exit')
    parser.add_argument('--offline-check', help='New report path; no camera or CAN')
    parser.add_argument('--outer-loop', choices=('tracking', 'settle', 'off'),
                        help='Temporary comparison: tracking=XYZ throughout, settle=XYZ after SDK path, off=SDK path only')
    args = parser.parse_args()
    if args.execute and args.offline_check:
        parser.error('--execute and --offline-check cannot be combined')
    cfg = load_config(args.config)
    if args.outer_loop is not None:
        cfg['cartesian_outer_loop']['enabled'] = args.outer_loop != 'off'
        cfg['cartesian_outer_loop']['during_motion'] = args.outer_loop == 'tracking'
    planner = WristPlanner(cfg)
    if args.offline_check:
        return offline_check(planner, args.offline_check)
    return run(cfg, planner, args.execute)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(1)

"""RGB-only ArUco demo UI. Camera, IK, trajectories and CAN reuse existing APIs."""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid

import cv2
import numpy as np

from .aruco_corners import CornerDetector, StableCorners
from .corner_flow import CornerFlow
from .hover_control import HoverController
from .hover_runtime import LatestFrames, RGBDFramePending, release_supported
from .vision import Camera


class StartupCancelled(RuntimeError):
    """User exited while SDK startup was still preparing disabled motors."""


class RGBCamera:
    """Adapt existing color-only Camera to the shared nonblocking frame worker."""
    def __init__(self, cfg, calibration):
        self.camera = Camera('realsense', serial=cfg['serial'], fps=int(cfg.get('camera_fps', 30)),
                             intrinsics=cfg.get('intrinsics'))
        self.last_number, self.warmup = -1, 20
        try:
            if (self.camera.metadata['serial'] != calibration['camera']['serial']
                    or [self.camera.width, self.camera.height] != calibration['image_size']
                    or not np.allclose(self.camera.k, calibration['K'], atol=.02, rtol=0)
                    or not np.allclose(self.camera.d, calibration['D'], atol=1e-12, rtol=0)):
                raise ValueError('RGB camera/profile/intrinsics differ from selected calibration')
        except BaseException:
            self.close()
            raise

    def read(self):
        try:
            image = self.camera.read(timeout_ms=200)
        except RuntimeError as exc:
            if "Frame didn't arrive within" in str(exc):
                raise RGBDFramePending(str(exc)) from exc
            raise
        number = self.camera.last_frame_metadata['frame_number']
        if number <= self.last_number:
            raise RGBDFramePending('Repeated/out-of-order RGB frame')
        self.last_number = number
        if self.warmup:
            self.warmup -= 1
            raise RGBDFramePending('RGB warmup')
        return image, None

    def close(self):
        self.camera.close()


def load_settings(path, mount, clearance_mm=None):
    path = Path(path).resolve()
    settings = json.loads(path.read_text())
    settings['mount'] = mount
    settings['_path'] = str(path)
    settings['base_config'] = str((path.parent/settings['base_config']).resolve())
    settings['log_dir'] = str((path.parent/settings['log_dir']).resolve())
    if clearance_mm is not None:
        settings['clearance_m'] = clearance_mm/1000.
    if (settings['dictionary'] not in ('DICT_4X4_50', 'DICT_5X5_50', 'DICT_6X6_50')
            or type(settings['marker_id']) is not int or not 0 <= settings['marker_id'] < 50):
        raise ValueError('Invalid ArUco dictionary/ID')
    ranges = dict(marker_m=(.01, .5), clearance_m=(0., .2),
                  max_pnp_rms_px=(.05, 2.), ambiguity_gap_px=(.01, 1.),
                  corner_span_m=(.0001, .01), stationary_s=(.4, 3.),
                  joint_span_rad=(.0001, .004), home_position_tolerance_m=(.001, .015),
                  home_tolerance_rad=(.001, .02))
    for key, (lo, hi) in ranges.items():
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError('Invalid '+key)
    if type(settings['stable_samples']) is not int or not 3 <= settings['stable_samples'] <= 20:
        raise ValueError('stable_samples must be 3..20')
    pitches = np.asarray(settings['down_pitches_deg'], float)
    if (pitches.ndim != 1 or not 1 <= len(pitches) <= 10 or not np.isfinite(pitches).all()
            or np.any(pitches < 10) or np.any(pitches > 90) or np.any(np.diff(pitches) >= 0)):
        raise ValueError('Downward pitches must decrease within 10..90 degrees')
    if settings['height_mode'] != 'corner':
        raise ValueError('This demo targets the selected corner Z + clearance_m')
    return settings


def project(planner, point, raw, mount):
    if point is None or (mount == 'wrist' and raw is None):
        return None
    pose = planner.camera_pose(raw) if mount == 'wrist' else planner.T_base_camera
    local = np.linalg.inv(pose) @ np.r_[point, 1.]
    if local[2] <= .001:
        return None
    uv, _ = cv2.projectPoints(local[:3].reshape(1, 3), np.zeros(3), np.zeros(3),
                              np.asarray(planner.calibration['K']), np.asarray(planner.calibration['D']))
    u, v = uv.reshape(2)
    if not np.isfinite([u, v]).all() or not (0 <= u < 1280 and 0 <= v < 720):
        return None
    return int(round(u)), int(round(v))


def window_event(window, opened=True):
    """Qt may destroy the last window before WND_PROP_VISIBLE is queried."""
    if not opened:
        return 255, False
    key = cv2.waitKey(5) & 0xff
    try:
        opened = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        opened = False
    return key, opened


def run(cfg, planner, settings, execute):
    # Import the already-used trace writer, with no camera/CAN constructed on import.
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0, str(scripts))
    from click_hover_eth import append_log, completion_summary, save_motion_traces
    logdir = Path(settings['log_dir']); logdir.mkdir(parents=True, exist_ok=True)
    log = logdir/f'{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}.jsonl'

    def emit(event):
        append_log(log, event)
        kind = event['event']
        if kind.endswith('_plan'):
            xyz = np.asarray(event['target_base_m'])*1000
            print(f"[移动] {kind}；XYZ(mm)={np.round(xyz, 1).tolist()}；{event['duration_s']:.1f} 秒", flush=True)
        elif kind == 'completion':
            if event['status'] == 'HOME_COMMAND_FINISHED_HOLD':
                print('[保持] 回零轨迹已执行完成。', flush=True)
            elif event['status'] in ('SETTLED_MODEL_ONLY', 'HOME_REACHED_HOLD', 'HOME_TIMEOUT_HOLD',
                                   'SETTLE_TIMEOUT_HOLD', 'TRACKING_REFUSED_HOLD', 'UI_TIMEOUT_HOLD'):
                print(completion_summary(event), flush=True)
        elif kind == 'plan_refused':
            print('[未执行] '+event['error'], flush=True)
        elif kind == 'corner_acquired':
            print(f"[角点 {event['corner']}] 基座 XYZ(mm)：{np.round(np.asarray(event['corner_base_m'])*1000, 2).tolist()}", flush=True)

    emit(dict(event='session', execute=execute, settings=settings, config=cfg,
              calibration_sha256=planner.result_sha256, depth_used=False,
              joint_offsets_rad=planner.offsets.tolist(), physical_accuracy_verified=False))
    print('[日志] '+str(log), flush=True)
    camera = frames = controller = flow = None
    errors, code = [], 0
    detector = CornerDetector(settings, planner.calibration)
    stable = StableCorners(settings['stable_samples'], settings['corner_span_m'])
    last_frame, displayed_observation = -1., None
    anchor, since = None, time.monotonic()
    window = 'ArUco corners | 1 2 3 4 | r reset | h hold | q exit home'
    window_open, last_phase, notice = True, None, 'Waiting for RGB'
    try:
        camera = RGBCamera(cfg, planner.calibration)
        frames = LatestFrames(camera); frames.start()
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        waiting = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(waiting, 'Waiting for RGB | q / Esc / close: exit before robot enable',
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)
        cv2.imshow(window, waiting)
        deadline, initial = time.monotonic()+10., None
        print('[相机] 等待 RGB；本演示不启用深度流。', flush=True)
        while initial is None:
            key, window_open = window_event(window, window_open)
            if key in (ord('q'), ord('x'), 27) or not window_open:
                return 0
            initial = frames.latest(cfg['camera_max_age_s'])
            if time.monotonic() > deadline:
                raise RuntimeError('No fresh RGB within 10 seconds; robot not enabled')
            time.sleep(.005)
        cv2.imshow(window, initial.color)
        if execute:
            print('[启动] 请停止其他控制器；将保持当前姿态。', flush=True)
            def before_enable():
                nonlocal window_open
                # SDK mode setup can be slow; consume exit queued during setup.
                key, window_open = window_event(window, window_open)
                if key in (ord('q'), ord('x'), 27, ord('h'), ord(' ')) or not window_open:
                    raise StartupCancelled('Exit requested before motor enable')
                if frames.latest(cfg['camera_max_age_s']) is None:
                    raise RuntimeError('No fresh RGB before motor enable')
            controller = HoverController(planner, before_enable=before_enable, startup_log=emit)
            controller.start()
            for checkpoint in controller.feedback_checkpoints:
                emit(dict(event='feedback_checkpoint', **checkpoint))
            flow = CornerFlow(planner, controller, settings, emit)
            if settings['mount'] == 'wrist':
                flow.reset()  # Camera need not see the marker at startup zero.
        else:
            print('[预览] 无 CAN/电机控制；腕部基座坐标需 --execute 后用实测关节角计算。', flush=True)
        reset_name = '回初始观察位' if settings['mount'] == 'wrist' else '回零'
        print(f"[操作] 1/2/3/4：先{reset_name}，再识别并对齐角点上方 {settings['clearance_m']*1000:g} mm；r：只回位。", flush=True)
        print('[操作] h/空格：取消并保持；q/Esc：回零后退出；运动中不接受角点切换。', flush=True)
        while True:
            try:
                # Consume stop/exit before handing a finished async plan to the sender.
                key, window_open = window_event(window, window_open)
                if key in (ord('q'), ord('x'), 27) or not window_open:
                    if not flow:
                        break
                    flow.exit()
                elif flow and key in (ord('h'), ord(' ')):
                    flow.hold()
                elif flow and key in tuple(map(ord, '1234r')):
                    number = None if key == ord('r') else key-ord('0')
                    if not flow.reset(number):
                        print('[未切换] 正在运动、规划或识别；h 可取消。', flush=True)
                now, raw, stationary = time.monotonic(), None, not execute
                if controller:
                    controller.heartbeat()
                    raw = controller.snapshot()[:6]
                    if not flow.exiting:
                        # Check before pump(): it may send a completed async plan.
                        try:
                            frame = frames.latest(cfg['camera_max_age_s'])
                            if frame is None:
                                raise RuntimeError('No fresh RGB frame')
                        except RuntimeError as exc:
                            flow.exit('CAMERA_ERROR: '+str(exc))
                    flow.pump()
                    save_motion_traces(controller, log)
                    if flow.done:
                        code = int(not flow.home_confirmed and flow.phase != 'HOME_FINISHED')
                        break
                    if anchor is None or np.max(np.abs(raw-anchor)) > settings['joint_span_rad']:
                        anchor, since = raw.copy(), now
                    stationary = now-since >= settings['stationary_s'] and not controller.busy
                    if flow.phase != last_phase:
                        stable.clear(); displayed_observation = None
                        last_phase = flow.phase
                        if flow.phase == 'WAIT_MARKER':
                            anchor, since = raw.copy(), now
                            stationary = False
                            print('[识别] 回位完成，等待停稳和连续稳定角点。', flush=True)
                        elif flow.phase in ('WAIT_KEY', 'CORNER_HOLD', 'HOLD'):
                            print('[保持] 可按 1/2/3/4 开始下一次；r 只回位。', flush=True)
                if flow and flow.exiting:
                    time.sleep(.005)
                    continue
                if controller is None:
                    frame = frames.latest(cfg['camera_max_age_s'])
                if frame is None:
                    time.sleep(.005)
                    continue
                if frame.acquired > last_frame:
                    last_frame = frame.acquired
                    if stationary and (not flow or flow.phase in ('WAIT_MARKER', 'WAIT_KEY', 'CORNER_HOLD', 'HOLD')):
                        try:
                            # Wrist preview shows camera-frame coordinates; never invent a measured q.
                            pose = (planner.camera_pose(raw) if settings['mount'] == 'wrist' and raw is not None
                                    else planner.T_base_camera if settings['mount'] == 'external' else np.eye(4))
                            observation = detector.detect(frame, pose, raw,
                                                          controller.motion_epoch if controller else -1)
                            displayed_observation = observation
                            notice = 'Marker found; waiting for stable corners'
                            accepted = stable.update(observation)
                            if accepted is not None:
                                notice = 'Stable marker | keys 1/2/3/4'
                                if flow and flow.phase == 'WAIT_MARKER':
                                    flow.observe(accepted)
                        except ValueError as exc:
                            stable.clear(); displayed_observation = None
                            notice = str(exc)
                    else:
                        stable.clear(); displayed_observation = None
                        notice = 'Waiting for reset / stationary joints'
                image = frame.color.copy()
                if displayed_observation is not None and now-displayed_observation.acquired <= cfg['camera_max_age_s']:
                    coords = displayed_observation.corners_base_m*1000
                    for index, (u, v) in enumerate(displayed_observation.corners_pixel):
                        uv = int(round(u)), int(round(v))
                        cv2.circle(image, uv, 5, (0, 255, 255), 2)
                        cv2.putText(image, str(index+1), (uv[0]+8, uv[1]-8), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
                        name = 'CAM' if settings['mount'] == 'wrist' and raw is None else 'BASE'
                        label = f'{index+1} {name} mm: '+', '.join(f'{v:.1f}' for v in coords[index])
                        cv2.putText(image, label, (12, 78+index*22), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
                pixel = project(planner, flow.target if flow else None, raw, settings['mount'])
                if pixel is not None:
                    cv2.drawMarker(image, pixel, (255, 255, 0), cv2.MARKER_DIAMOND, 22, 2)
                phase = flow.phase if flow else 'PREVIEW_NO_ROBOT'
                cv2.putText(image, phase, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 0), 1)
                cv2.putText(image, notice[:120], (12, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                cv2.imshow(window, image)
            except KeyboardInterrupt:
                if not flow:
                    break
                if flow.exiting:
                    flow.hold()
                else:
                    flow.exit('CTRL_C')
    except StartupCancelled:
        print('[退出] 已在电机使能前取消启动。', flush=True)
    except BaseException as exc:
        code = 1
        if controller and cfg.get('manual_disable_only') and controller.enabled:
            controller.enter_fault_hold(exc)
        if flow:
            flow.fail(str(exc))
        print('[错误] '+str(exc)+'；故障后不声明自动回零成功。', file=sys.stderr, flush=True)
        try:
            emit(dict(event='fault', error=str(exc),
                      fault_action='hold_without_disable' if cfg.get('manual_disable_only') else 'legacy',
                      hold_transport_error=(None if getattr(controller,'hold_transport_error',None) is None
                                            else str(controller.hold_transport_error)),
                      feedback_checkpoints=controller.feedback_checkpoints if controller else [],
                      feedback_diagnostics=(controller.failure_diagnostics or controller.feedback_diagnostics())
                      if controller else None))
        except (OSError, ValueError, TypeError) as log_error:
            errors.append('fault log: '+str(log_error))
    finally:
        if flow:
            flow.worker.cancel()
        if controller:
            try:
                save_motion_traces(controller, log)
            except Exception as exc:
                errors.append('trace save: '+str(exc))
            try:
                if cfg.get('manual_disable_only') or not (flow and flow.home_confirmed):
                    release_supported(controller)
            except BaseException as exc:
                errors.append('release: '+str(exc))
            finally:
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
            print('[清理错误] '+'; '.join(errors), file=sys.stderr)
    return max(code, int(bool(errors)))


def main(mount, config, loader, planner_type):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(config))
    parser.add_argument('--execute', action='store_true', help='Enable robot and perform reset/corner/home motions')
    parser.add_argument('--clearance-mm', type=float, help='Base +Z clearance above each detected corner; default from config')
    parser.add_argument('--control-rate-hz', type=int, choices=(100, 200, 250, 500),
                        help='Temporary SDK send-rate override for communication diagnosis; default from base config')
    parser.add_argument('--outer-loop', choices=('tracking', 'settle', 'off'),
                        help='Temporary comparison: tracking=XYZ throughout, settle=XYZ after SDK path, off=SDK path only')
    args = parser.parse_args()
    settings = load_settings(args.config, mount, args.clearance_mm)
    cfg = loader(settings['base_config'])
    cfg['manual_disable_only']=True
    cfg['home_completion_mode']='trajectory'
    if args.outer_loop is not None:
        cfg['cartesian_outer_loop']['enabled'] = args.outer_loop != 'off'
        cfg['cartesian_outer_loop']['during_motion'] = args.outer_loop == 'tracking'
    if args.control_rate_hz is not None:
        cfg['control_rate_hz'] = args.control_rate_hz
        print(f'[诊断] 本次 SDK 发送频率 {args.control_rate_hz} Hz；反馈超时阈值保持不变。', flush=True)
    cfg['home_position_tolerance_m'] = settings['home_position_tolerance_m']
    cfg['home_tolerance_rad'] = settings['home_tolerance_rad']
    planner = planner_type(cfg)
    return run(cfg, planner, settings, args.execute)

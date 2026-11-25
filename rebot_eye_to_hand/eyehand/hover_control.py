"""Official RebotArm MIT transport with explicit current-pose lifecycle."""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from .click_hover import JOINT_NAMES


class HoverController:
    def __init__(self, planner, robot=None, before_enable=None, startup_log=None):
        self.planner, self.cfg = planner, planner.cfg
        if robot is None:
            from reBotArm_control_py.actuator import RebotArm
            robot = RebotArm('rebotarm_rs.yaml')
            robot._channel = self.cfg['channel']
        self.robot = robot
        self.before_enable = before_enable
        self.startup_log = startup_log
        self.lock = threading.RLock()
        self.io = threading.RLock()
        self.plan = None
        self.fault = None
        self.enabled = False
        self.connected = False
        self.feedback = None
        self.reporting_motors = []
        self.target = None
        self.gripper = None
        self.completion = None
        self.stopped = False
        self.motion_epoch = 0
        self.ui_seen = time.perf_counter()
        self.disable_errors = []
        self.cartesian_settler = None
        self.motion_intervals = []
        self.motion_compute_costs = []
        self.trace_samples = deque(maxlen=3000)
        self.pending_traces = []
        self.last_trace_at = -float('inf')
        self.feedback_checkpoints = []
        self.failure_diagnostics = None
        self.callback_count = 0
        self.send_counts = dict(arm_batches=0, gripper_batches=0)
        self.send_timings = deque(maxlen=1000)
        self.loop_started_at = None
        self.target_executor = None
        self.disabled_confirmed = False
        self.last_gripper_send = -float('inf')
        self.hold_torque = None
        self.hold_transport_error = None
        self.home_confirmed = False
        self.manual_disable_authorized = False

    @property
    def manual_disable_only(self):
        return self.cfg.get('manual_disable_only',False)

    def feedback_diagnostics(self):
        """Copy SDK telemetry without querying motors or refreshing packet ages.

        Kept out of the normal callback path; file I/O belongs to the UI.
        Zero CAN error counters do not establish that every motor is reporting.
        """
        result = dict(configured_rate_hz=self.cfg['control_rate_hz'],
                      feedback_max_age_ms=1000*self.cfg['feedback_max_age_s'],
                      gripper_refresh_hz=50., periodic_active_reports=False,
                      feedback_source=getattr(self, 'feedback_source', None),
                      callback_count=self.callback_count, send_counts=self.send_counts.copy())
        result['fault_hold']=dict(policy='manual_release' if self.manual_disable_only else 'legacy_disable',
            active=bool(self.fault and self.enabled and self.manual_disable_only),
            transport_error=self.hold_transport_error,home_confirmed=self.home_confirmed)
        if self.loop_started_at is not None:
            elapsed = time.perf_counter()-self.loop_started_at
            result['loop_elapsed_s'] = elapsed
            result['observed_callback_rate_hz'] = self.callback_count/elapsed if elapsed > 0 else None
        try:
            with self.io:
                receiver = self.feedback
                guard = getattr(receiver, 'lock', None)
                if guard is not None:
                    guard.acquire()
                try:
                    now = time.perf_counter()
                    times = np.asarray(receiver.times, dtype=float).copy()
                    sequences = np.asarray(receiver.sequences).copy()
                    intervals = [list(gaps) for gaps in receiver.intervals]
                    result['receiver_error'] = getattr(receiver, 'error', None)
                    result['timestamp_bound_adjustments'] = list(map(int,getattr(receiver,'timestamp_bound_adjustments',[])))
                    result['last_timestamp_bound_adjustment'] = getattr(receiver,'last_timestamp_bound_adjustment',None)
                finally:
                    if guard is not None:
                        guard.release()
            names = JOINT_NAMES + self.robot.groups['gripper'].joint_names
            result['motors'] = []
            for i, name in enumerate(names):
                # Public snapshot adapter records (observed gap, sequence delta).
                gaps = intervals[i]
                pairs = [(g, 1) if np.isscalar(g) else g for g in gaps]
                result['motors'].append(dict(name=name, sequence=int(sequences[i]),
                    age_ms=float(1000*(now-times[i])) if np.isfinite(times[i]) else None,
                    interval_count=len(gaps),
                    mean_period_ms=float(1000*sum(g for g,n in pairs)/sum(n for g,n in pairs)) if pairs else None,
                    max_observed_snapshot_gap_ms=float(1000*max(g for g,n in pairs)) if pairs else None))
        except Exception as exc:
            # Diagnostics must never replace the original fault or block disable.
            result['diagnostic_error'] = str(exc)
        with self.io:
            costs = np.asarray(list(self.send_timings))
        if costs.size:
            result['send_cost_ms'] = dict(arm_max=float(1000*max(costs[:, 0])),
                                         gripper_max=float(1000*max(costs[:, 1])))
        return result

    def enter_fault_hold(self,error):
        """Latch fault, freeze the last fully sent target, never start a new path."""
        with self.lock:
            self._queue_trace('FAULT_HOLD')
            self.fault=self.fault or str(error)
            self.motion_epoch+=1
            self._cancel_target_worker();self.plan=None;self.cartesian_settler=None
            self.home_confirmed=False
        self._ensure_hold_loop()

    def _ensure_hold_loop(self):
        if not self.manual_disable_only or not self.enabled or self.stopped:return
        thread=getattr(self.robot,'_ctrl_thread',None)
        if ((isinstance(thread,threading.Thread) and not thread.is_alive()) or
                (thread is None and self.fault)):
            try:self.robot.start_control_loop(self._tick,rate=self.cfg['control_rate_hz'])
            except Exception as exc:self.hold_transport_error='Hold sender unavailable: '+str(exc)

    def _send_fault_hold(self,robot):
        if not self.enabled or self.target is None or self.hold_torque is None:return
        now=time.perf_counter()
        if self.hold_transport_error and now-getattr(self,'last_hold_attempt',-float('inf'))<.1:return
        self.last_hold_attempt=now
        try:
            with self.lock,self.io:
                robot.arm.send_mit(self.target,vel=np.zeros(6),kp=self.cfg['kp'],kd=self.cfg['kd'],
                    tau=self.hold_torque,strict=True,cancelled=lambda:self.stopped)
                if now-self.last_gripper_send>=.02-1e-9:
                    robot.gripper.send_mit(self.gripper,strict=True,cancelled=lambda:self.stopped)
                    self.last_gripper_send=now
            self.hold_transport_error=None
        except Exception as exc:
            self.hold_transport_error=str(exc)  # No disable/re-enable/disconnect.

    def authorize_manual_disable(self):
        # Explicit terminal release is the authorization; no extra home prerequisite.
        self.manual_disable_authorized=True

    def _feedback_checkpoint(self, stage):
        self.feedback_checkpoints.append(dict(stage=stage, **self.feedback_diagnostics()))

    def _read_parameters(self):
        before = time.perf_counter()
        q = np.r_[self.robot.arm.get_positions(fresh=True, strict=True, timeout_ms=150),
                  self.robot.gripper.get_positions(fresh=True, strict=True, timeout_ms=150)]
        if time.perf_counter()-before > .3 or q.shape != (7,) or not np.isfinite(q).all():
            raise RuntimeError('Startup parameter feedback invalid/slow; motors not enabled')
        try:
            self.planner.validate_feedback(q[:6])
        except ValueError as exc:
            raise ValueError(f'Startup mechPos feedback; motors not enabled: {exc}') from exc
        return q

    def snapshot(self):
        with self.io:
            try:
                snapshot = self.feedback.snapshot(max_age=self.cfg['feedback_max_age_s'])
            except Exception:
                if self.failure_diagnostics is None:
                    self.failure_diagnostics = self.feedback_diagnostics()
                raise
            q = np.asarray(snapshot.positions, dtype=float)
            if q.shape != (7,) or not np.isfinite(q).all():
                raise RuntimeError('Expected seven finite, fresh motor positions')
            self.planner.validate_feedback(q[:6])
            return q.copy()

    def start(self):
        try:
            self._start_impl()
        except BaseException as exc:
            if self.manual_disable_only and self.enabled:
                self.enter_fault_hold('Startup failed: '+str(exc))
                raise
            self.fault = f'Startup failed: {exc}'
            if self.connected:
                try:
                    self._disable_motors()
                finally:
                    self.enabled = False
                    self._stop_feedback_reporting()
            raise

    def _start_feedback(self, names):
        """Use upstream public snapshots and MIT replies, as the working project does."""
        from .native_feedback import NativeFeedback
        motors = [self.robot._motor_map[n] for n in names]
        if not NativeFeedback.supported(motors):
            raise RuntimeError('Public SDK get_state_snapshot unavailable; not enabling motors')
        self.feedback = NativeFeedback(self.robot, names)
        self.feedback_source = 'SDK Motor.get_state_snapshot (MIT command/reply)'
        # MIT already generates status replies. Extra periodic reports compete
        # with those replies; do not add a second stream on this shared bus.
        for name, motor in zip(names, motors):
            self.reporting_motors.append((name, motor))
            motor.robstride_set_active_report(False)
        return motors

    def _disabled_probe(self):
        if not self.disabled_confirmed or self.enabled:
            raise RuntimeError('Zero-gain status probe requires confirmed disabled motors')
        for group in (self.robot.arm, self.robot.gripper):
            zeros = np.zeros(len(group.joint_names))
            group.send_mit(zeros, vel=zeros, kp=zeros, kd=zeros, tau=zeros, strict=True)

    def _sync_disabled_feedback(self):
        # Complete all slow UI/parameter work before this bounded status sync.
        started = time.perf_counter()
        deadline = started+self.cfg['feedback_max_age_s']
        self._disabled_probe()
        while time.perf_counter() < deadline:
            try:
                fresh = self.snapshot()
                if np.all(self.feedback.times >= started) and time.perf_counter() <= deadline:
                    return fresh
            except Exception as exc:
                from .native_feedback import FeedbackUnavailable
                if not isinstance(exc, FeedbackUnavailable):
                    raise
            time.sleep(.001)
        raise RuntimeError('No fresh post-probe SDK status before enable; motors remain disabled')

    def _stop_feedback_reporting(self):
        errors, pending = [], []
        for name, motor in self.reporting_motors:
            try:
                motor.robstride_set_active_report(False)
            except Exception as exc:
                errors.append(f'stop active reporting {name}: {exc}')
                pending.append((name, motor))
        self.reporting_motors = pending  # Allow close() to retry startup cleanup.
        return errors

    def _start_impl(self):
        self.connected = True  # Also clean up a partially failed SDK connect.
        self.robot.connect()
        names = JOINT_NAMES + self.robot.groups['gripper'].joint_names
        if self.robot.arm.joint_names != JOINT_NAMES or len(names) != 7:
            raise ValueError('Requires official six-joint RS + one motor gripper configuration')
        expected_models = ['rs-06']*3 + ['rs-00']*4
        if len(self.robot._all_joints) != 7 or any(
                j.name != names[i] or j.vendor != 'robstride' or j.motor_id != i+1
                or j.feedback_id != 0xFD or j.model != expected_models[i]
                for i, j in enumerate(self.robot._all_joints)):
            raise ValueError('Expected RS motors 1..7 and host 0xFD; inspect hardware configuration')
        for gains in (self.robot.gripper._mit_kp, self.robot.gripper._mit_kd):
            gains = np.asarray(gains, dtype=float)
            if gains.shape != (1,) or not np.isfinite(gains).all() or np.any(gains < 0):
                raise ValueError('Invalid gripper MIT gains')
        if self.robot.gripper._mit_kp[0] <= 0:
            raise ValueError('Gripper hold requires positive position gain')
        self._disable_motors()
        if self.disable_errors:
            raise RuntimeError('Cannot confirm disabled startup: '+'; '.join(self.disable_errors))
        self.disabled_confirmed = True
        self._read_parameters()
        if not self.robot.arm.mode_mit(kp=self.cfg['kp'], kd=self.cfg['kd']):
            raise RuntimeError('Arm MIT mode setup failed')
        if not self.robot.gripper.mode_mit():
            raise RuntimeError('Gripper MIT mode setup failed')
        motors = self._start_feedback(names)
        q0 = self._read_parameters()  # Refresh after mode changes, while still disabled.
        deadline = time.perf_counter()+5
        previous = None
        last = None
        while time.perf_counter() < deadline:
            time.sleep(.03)
            self._disabled_probe()
            try:
                fresh = self.snapshot()
                if np.max(np.abs(fresh-q0)) > .03:
                    raise RuntimeError('Status angle and mechPos disagree or arm moved during startup')
                if previous is None:
                    previous = fresh
                    continue
                if np.max(np.abs(fresh-previous)) > .005:
                    previous = fresh
                    raise RuntimeError('Arm/gripper must be stationary before enable')
                break
            except Exception as exc:
                last = exc
        else:
            raise RuntimeError(f'No fresh feedback before enable ({self.feedback_source}; MIT command/reply probes requested): {last}')
        # Capture fresh pose immediately before activation; no default zero target.
        self.failure_diagnostics = None  # Discard recoverable disabled-startup retries.
        self._feedback_checkpoint('disabled_stationary')
        captured_raw = fresh.copy()
        self.target = self.planner.command_from_feedback(fresh[:6])
        if self.startup_log is not None:
            from .native_runtime import activate
            self.startup_log({'event': 'robot_startup', 'feedback_source': self.feedback_source,
                'raw_q_rad': fresh[:6].tolist(), 'hold_q_rad': self.target.tolist(),
                'periodic_active_reports': False, 'gripper_refresh_hz': 50.,
                'motorbridge': activate(),
                'command_projected_to_limits': bool(np.any(self.target != fresh[:6]))})
        else:
            print(f'Feedback ready: {self.feedback_source}; seven-axis MIT command/reply, periodic reports off, gripper 50 Hz.', flush=True)
            print('Startup raw joints (deg): '+str(np.round(np.degrees(fresh[:6]),4).tolist()), flush=True)
            if np.any(self.target != fresh[:6]):
                print('Startup hold command projected to URDF bounds (deg): '
                      +str(np.round(np.degrees(self.target),4).tolist()), flush=True)
        self.gripper = fresh[6:].copy()
        fresh = np.r_[self.target, self.gripper]
        tau0 = self.planner.gravity_torque(self.target)
        self.hold_torque=tau0.copy()
        if self.before_enable is not None:
            self.before_enable()  # Recheck camera after the slow SDK mode setup.
        # UI/event processing and startup logging may have taken time. Never
        # enable using a packet that was only fresh before that work.
        latest = self._sync_disabled_feedback()
        if np.max(np.abs(latest-captured_raw)) > .005:
            raise RuntimeError('Arm/gripper moved during startup checks; motors not enabled')
        self._feedback_checkpoint('immediately_before_enable')
        for i, motor in enumerate(motors):
            if i < 6:
                motor.send_mit(float(fresh[i]), 0., self.cfg['kp'][i], self.cfg['kd'][i], float(tau0[i]))
            else:
                motor.send_mit(float(fresh[i]), 0., float(self.robot.gripper._mit_kp[0]),
                               float(self.robot.gripper._mit_kd[0]), 0.)
            if self.manual_disable_only:self.enabled=True  # Includes uncertain partial enable.
            motor.enable()
            # Some firmware ignores MIT frames while disabled; immediately resend.
            if i < 6:
                motor.send_mit(float(fresh[i]), 0., self.cfg['kp'][i], self.cfg['kd'][i], float(tau0[i]))
            else:
                motor.send_mit(float(fresh[i]), 0., float(self.robot.gripper._mit_kp[0]),
                               float(self.robot.gripper._mit_kd[0]), 0.)
            self.enabled = True  # Includes partial-enable failure cleanup.
        self.disabled_confirmed = False
        self.last_gripper_send = time.perf_counter()
        self._feedback_checkpoint('enabled_before_loop')
        self.ui_seen = time.perf_counter()
        self.loop_started_at = self.ui_seen
        # Official structure: target preparation is separate from the only CAN sender.
        from concurrent.futures import ThreadPoolExecutor
        self.target_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rebot-target')
        self.robot.start_control_loop(self._tick, rate=self.cfg['control_rate_hz'])

    def move(self, plan, expected_epoch=None):
        with self.lock:
            if self.fault or self.stopped or not self.enabled:
                raise RuntimeError(self.fault or 'Controller unavailable')
            if expected_epoch is not None and expected_epoch != self.motion_epoch:
                raise RuntimeError('Pending click was cancelled; click again')
            if time.perf_counter()-self.ui_seen > self.cfg['ui_timeout_s']:
                raise RuntimeError('UI heartbeat expired; click again')
            if self.plan is not None:
                raise RuntimeError('Still moving/settling; wait or press h')
            q = self.snapshot()[:6]
            if np.max(np.abs(q-plan.curve(0))) > .025:
                raise RuntimeError('Arm changed during planning; click again')
            self.plan, self.started = plan, time.perf_counter()
            self.home_confirmed=False
            self.manual_disable_authorized=False
            self._cancel_target_worker()
            self.cartesian_settler = None
            self.motion_intervals.clear()
            self.motion_compute_costs.clear()
            self.trace_samples.clear()
            self.last_trace_at = -float('inf')
            self.stable_since = None
            self.completion = None

    def _cancel_target_worker(self):
        cancel = getattr(self.cartesian_settler, 'cancel', None)
        if cancel is not None:
            cancel()

    def hold(self, reason='CANCELLED_HOLD'):
        with self.lock:
            self._queue_trace(reason)
            self.motion_epoch += 1
            self._cancel_target_worker()
            self.plan = None
            self.cartesian_settler = None
            self.completion = {'status': reason}
            # Freeze last sent position; no new target based on stale feedback.

    def heartbeat(self,allow_fault=False):
        with self.lock:
            now = time.perf_counter()
            if now-self.ui_seen > self.cfg['ui_timeout_s']:
                self.hold('UI_TIMEOUT_HOLD')
            self.ui_seen = now
            thread = getattr(self.robot, '_ctrl_thread', None)
            if isinstance(thread, threading.Thread) and not thread.is_alive() and not self.fault:
                if self.manual_disable_only:self.enter_fault_hold('Official SDK control thread stopped unexpectedly')
                else:
                    self.fault = 'Official SDK control thread stopped unexpectedly'
                    self._disable_motors()
            if self.manual_disable_only:self._ensure_hold_loop()
            if self.fault and not allow_fault:
                raise RuntimeError(self.fault)

    def take_completion(self):
        with self.lock:
            result, self.completion = self.completion, None
            return result

    def _queue_trace(self, status):
        with self.lock:
            if self.trace_samples:
                self.pending_traces.append((status, self.trace_samples))
                self.trace_samples = deque(maxlen=3000)

    def take_motion_traces(self):
        """GUI drains numeric buffers; the control callback never writes files."""
        with self.lock:
            traces, self.pending_traces = self.pending_traces, []
        return [(status, np.asarray(samples)) for status, samples in traces]

    def planning_start(self):
        """Start the next SDK trajectory at the held command without dropping bias."""
        with self.lock:
            measured = self.snapshot()[:6]
            if self.cfg.get('cartesian_outer_loop', {}).get('enabled') and self.target is not None:
                return self.target.copy()
            return measured

    def _disable_motors(self):
        # Bulk SDK disable can stop at one failed motor; attempt every handle.
        self.disable_errors = []
        motors = getattr(self.robot, '_motor_map', {})
        if not motors:
            self.robot.estop()
            return
        for name, motor in list(motors.items()):
            try:
                motor.disable()
            except Exception as exc:
                self.disable_errors.append(f'{name}: {exc}')
        if self.disable_errors:
            try:
                print('DISABLE ERRORS: ' + '; '.join(self.disable_errors) + '; use physical stop/support', flush=True)
            except (OSError, ValueError):
                pass

    @property
    def motion_kind(self):
        with self.lock:
            return getattr(self.plan, 'kind', None) if self.plan is not None else None

    @property
    def busy(self):
        with self.lock:
            return self.plan is not None

    def _motion_completion(self, status, q, tcp, elapsed, command):
        """Record the same feedback diagnostics for arrival and settle timeout.

        Residual signs are feedback/model position minus the requested goal.
        This is model-based telemetry, not a physical accuracy measurement.
        """
        residual = tcp-self.plan.target
        result = {
            'status': status,
            'raw_q_rad': q[:6].tolist(),
            'raw_goal_rad': self.plan.goal.tolist(),
            'raw_command_rad': command.tolist(),
            'joint_feedback_minus_goal_deg': np.degrees(q[:6]-self.plan.goal).tolist(),
            'joint_feedback_minus_command_deg': np.degrees(q[:6]-command).tolist(),
            'target_base_m': self.plan.target.tolist(),
            'tcp_base_m': tcp.tolist(),
            'fk_error_xyz_mm': (1000*residual).tolist(),
            'fk_error_mm': float(1000*np.linalg.norm(residual)),
            'elapsed_s': float(elapsed),
            'planned_duration_s': float(self.plan.duration),
            'physical_accuracy_verified': False,
        }
        if hasattr(self.plan, 'diagnostics'):
            result['sdk_transit'] = self.plan.diagnostics
        if getattr(self.plan, 'kind', None) == 'ready':
            result['ready_position_tolerance_mm'] = float(1000*self.cfg.get(
                'ready_position_tolerance_m', self.cfg['settle_tolerance_m']))
            result['ready_joint_tolerance_deg'] = float(np.degrees(self.cfg.get(
                'ready_joint_tolerance_rad', self.cfg['home_tolerance_rad'])))
            result['ready_joint_error_deg'] = float(np.degrees(np.max(np.abs(q[:6]-self.plan.goal))))
        if getattr(self.plan, 'kind', None) == 'home':
            result['home_position_tolerance_mm'] = float(1000*self.cfg.get(
                'home_position_tolerance_m', self.cfg['settle_tolerance_m']))
            result['home_joint_tolerance_deg'] = float(np.degrees(self.cfg['home_tolerance_rad']))
        result['fk_error_xy_mm'] = float(1000*np.linalg.norm(residual[:2]))
        result['settled_axes'] = self.cartesian_settler.axes if self.cartesian_settler is not None else 'xyz'
        result['controlled_fk_error_mm'] = (float(1000*self.cartesian_settler.error)
                                             if self.cartesian_settler is not None else result['fk_error_mm'])
        if self.cartesian_settler is not None:
            result['cartesian_outer_loop'] = self.cartesian_settler.diagnostics()
        result['control_timing'] = {'configured_rate_hz': self.cfg.get('control_rate_hz'),
                                    'sample_count': len(self.motion_intervals)}
        if self.motion_intervals:
            intervals = np.asarray(self.motion_intervals)
            result['control_timing'].update(
                observed_callback_rate_hz=float(1./np.mean(intervals)),
                p95_callback_interval_ms=float(1000*np.percentile(intervals,95)),
                max_callback_interval_ms=float(1000*np.max(intervals)))
        if self.motion_compute_costs:
            result['control_timing']['callback_work_p50_p95_max_ms'] = (
                1000*np.percentile(self.motion_compute_costs, [50,95,100])).tolist()
        result['mit_velocity_feedforward'] = getattr(self.plan, 'mit_velocity_feedforward', True)
        self._queue_trace(status)
        return result

    def _tick(self, robot, dt):
        if self.stopped:return
        if self.fault:
            if self.manual_disable_only:self._send_fault_hold(robot)
            return
        try:
            callback_started = time.perf_counter()
            with self.lock, self.io:
                self.callback_count += 1
                q = self.snapshot()
                now = time.perf_counter()
                tick_interval = now-getattr(self, '_last_tick', now)
                if hasattr(self, '_last_tick') and now-self._last_tick > .1:
                    raise RuntimeError('Control callback gap exceeded 100 ms')
                if self.plan is not None and hasattr(self, '_last_tick') and now > self._last_tick:
                    self.motion_intervals.append(now-self._last_tick)
                self._last_tick = now
                if now-self.ui_seen > self.cfg['ui_timeout_s'] and self.plan is not None:
                    self.hold('UI_TIMEOUT_HOLD')
                target, velocity = self.target.copy(), np.zeros(6)
                tracking_frame = None
                if self.plan is not None:
                    elapsed = now-self.started
                    target, velocity = self.plan.sample(elapsed)
                    if not getattr(self.plan, 'mit_velocity_feedforward', True):
                        # SDK RebotArmEndPose keeps _qd_target at zero in MIT.
                        # Keep trajectory derivatives for limits, not motor velocity.
                        velocity = np.zeros(6)
                    tracking = (getattr(self.plan, 'tracking_outer_loop', False)
                                and self.cfg.get('cartesian_outer_loop', {}).get('enabled', False)
                                and self.cfg['cartesian_outer_loop'].get('during_motion', True))
                    if tracking:
                        from .cartesian_settle import CartesianTracker
                        nominal, reference = target.copy(), self.plan.reference(elapsed)
                        if self.cartesian_settler is None:
                            if self.target_executor is None:
                                # Deterministic offline replay without starting a thread.
                                self.cartesian_settler = CartesianTracker(self.planner, nominal, reference)
                            else:
                                from .target_worker import BufferedCartesianTracker
                                self.cartesian_settler = BufferedCartesianTracker(
                                    self.planner, nominal, reference, self.target_executor)
                        target = self.cartesian_settler.advance(nominal, reference, q[:6], now,
                                                               settling=elapsed >= self.plan.duration)
                        velocity = np.zeros(6)
                        # Record the reference actually published, including producer delay.
                        nominal = self.cartesian_settler.nominal.copy()
                        reference = self.cartesian_settler.target.copy()
                        tcp = self.cartesian_settler.fk(q[:6])
                        self.cartesian_settler.residual = reference-tcp
                        self.cartesian_settler.error = float(np.linalg.norm(
                            self.cartesian_settler.mask*(reference-tcp)))
                        # Keep only the last 30 seconds, sampled at <=100 Hz.
                        if now-self.last_trace_at >= .01-1e-9:
                            tracking_frame = np.r_[elapsed, float(elapsed >= self.plan.duration),
                                nominal, target, q[:6], reference, tcp,
                                self.cartesian_settler.velocity]
                        if self.cartesian_settler.stop_reason:
                            target = self.target.copy()
                            self.completion = self._motion_completion('TRACKING_REFUSED_HOLD', q, tcp, elapsed, target)
                            self._cancel_target_worker()
                            self.plan = None
                    if self.plan is not None and elapsed >= self.plan.duration:
                        # Separate FK data: Pinocchio data is never shared with GUI planning.
                        import pinocchio as pin
                        if not hasattr(self, 'fk_data'):
                            self.fk_data = self.planner.model.createData()
                        pin.forwardKinematics(self.planner.model, self.fk_data, q[:6]+self.planner.offsets)
                        pin.updateFramePlacements(self.planner.model, self.fk_data)
                        tcp = self.fk_data.oMf[self.planner.tcp_id].translation.copy()
                        error = float(np.linalg.norm(tcp-self.plan.target))
                        is_home = getattr(self.plan,'kind','hover') == 'home'
                        is_ready = getattr(self.plan,'kind','hover') == 'ready'
                        outer = (self.cfg.get('cartesian_outer_loop', {}).get('enabled', False)
                                 and getattr(self.plan, 'kind', 'hover') == 'hover')
                        tolerance = self.cfg['settle_tolerance_m']
                        if is_ready:
                            tolerance = self.cfg.get('ready_position_tolerance_m', tolerance)
                        elif is_home:
                            tolerance = self.cfg.get('home_position_tolerance_m', tolerance)
                        timeout = self.cfg['settle_timeout_s']
                        refused = False
                        if outer:
                            from .cartesian_settle import CartesianSettler
                            if self.cartesian_settler is None:
                                if self.target_executor is None:
                                    self.cartesian_settler = CartesianSettler(self.planner, self.plan.goal, self.plan.target)
                                else:
                                    from .target_worker import BufferedCartesianTracker
                                    self.cartesian_settler = BufferedCartesianTracker(
                                        self.planner, self.plan.goal, self.plan.target, self.target_executor)
                            if not tracking:
                                if hasattr(self.cartesian_settler, 'advance'):
                                    target = self.cartesian_settler.advance(self.plan.goal, self.plan.target,
                                                                          q[:6], now, settling=True)
                                else:
                                    target = self.cartesian_settler.step(q[:6], now)
                            # Arrival always uses this tick's fresh feedback,
                            # even when the outer loop updates more slowly.
                            residual = self.plan.target-tcp
                            self.cartesian_settler.residual = residual
                            self.cartesian_settler.error = float(np.linalg.norm(self.cartesian_settler.mask*residual))
                            error = self.cartesian_settler.error
                            velocity = np.zeros(6)
                            tolerance = self.cfg['cartesian_outer_loop']['tolerance_m']
                            timeout = self.cfg['cartesian_outer_loop']['timeout_s']
                            refused = self.cartesian_settler.stop_reason is not None
                        joint_error = float(np.max(np.abs(q[:6]-self.plan.goal))) if is_home or is_ready else None
                        settled = not refused and error <= tolerance
                        if is_home:
                            settled = settled and joint_error <= self.cfg['home_tolerance_rad']
                        elif is_ready:
                            settled = settled and joint_error <= self.cfg.get(
                                'ready_joint_tolerance_rad', self.cfg['home_tolerance_rad'])
                        if self.cartesian_settler is not None and hasattr(self.cartesian_settler, 'quiet'):
                            settled = settled and self.cartesian_settler.quiet
                        home_command_only = is_home and self.cfg.get('home_completion_mode') == 'trajectory'
                        if home_command_only:
                            # Corner demos use SDK trajectory completion, not an FK accuracy gate.
                            # Normal feedback/tracking faults above still cancel motion and hold.
                            settled = elapsed >= self.plan.duration + .5
                        if settled:
                            self.stable_since = self.stable_since or now
                            if home_command_only or now-self.stable_since >= .5:
                                self.completion = self._motion_completion(
                                    'HOME_COMMAND_FINISHED_HOLD' if home_command_only else
                                    'HOME_REACHED_HOLD' if is_home else 'SETTLED_MODEL_ONLY', q, tcp, elapsed, target)
                                if is_home:
                                    self.completion['home_joint_error_deg'] = float(np.degrees(joint_error))
                                    self.home_confirmed=not home_command_only
                                self._cancel_target_worker()
                                self.plan = None
                        else:
                            self.stable_since = None
                        if self.plan is not None and (refused or elapsed > self.plan.duration+timeout):
                            self.completion = self._motion_completion(
                                'HOME_TIMEOUT_HOLD' if is_home else 'SETTLE_TIMEOUT_HOLD', q, tcp, elapsed, target)
                            if is_home:
                                self.completion['home_joint_error_deg'] = float(np.degrees(joint_error))
                            self._cancel_target_worker()
                            self.plan = None
                # Record identical signals in comparison modes as well as XYZ tracking.
                if (tracking_frame is None and self.plan is not None
                        and getattr(self.plan, 'kind', None) == 'hover'
                        and hasattr(self.plan, 'reference')
                        and now-self.last_trace_at >= .01-1e-9):
                    import pinocchio as pin
                    if not hasattr(self, 'trace_fk_data'):
                        self.trace_fk_data = self.planner.model.createData()
                    pin.framesForwardKinematics(self.planner.model, self.trace_fk_data,
                                                q[:6]+self.planner.offsets)
                    tcp = self.trace_fk_data.oMf[self.planner.tcp_id].translation.copy()
                    tracking_frame = np.r_[elapsed, float(elapsed >= self.plan.duration),
                        self.plan.sample(elapsed)[0], target, q[:6], self.plan.reference(elapsed), tcp,
                        self.cartesian_settler.velocity if self.cartesian_settler is not None else np.zeros(6)]
                target = self.planner.validate_raw(target)
                if np.max(np.abs(q[:6]-target)) > self.cfg['tracking_limit_rad']:
                    raise RuntimeError('Joint tracking error exceeds tracking_limit_rad')
                if np.max(np.abs(q[6:]-self.gripper)) > .10:
                    raise RuntimeError('Gripper moved away from captured startup opening')
                tau = self.planner.gravity_torque(q[:6])
                send_started = time.perf_counter()
                robot.arm.send_mit(target, vel=velocity, kp=self.cfg['kp'], kd=self.cfg['kd'],
                                   tau=tau, strict=True, cancelled=lambda: self.stopped)
                arm_sent = time.perf_counter()
                self.send_counts['arm_batches'] += 1
                # Holding a fixed opening does not require 500 Hz traffic.
                # Same 50 Hz gripper refresh cadence as SDK impedance/working demo.
                if now-self.last_gripper_send >= .02-1e-9:
                    robot.gripper.send_mit(self.gripper, strict=True, cancelled=lambda: self.stopped)
                    self.last_gripper_send = now
                    self.send_counts['gripper_batches'] += 1
                self.send_timings.append((arm_sent-send_started, time.perf_counter()-arm_sent))
                self.target = target.copy()
                self.hold_torque=tau.copy()
                if self.plan is not None:
                    self.motion_compute_costs.append(time.perf_counter()-callback_started)
                if tracking_frame is not None and self.plan is not None:
                    self.trace_samples.append(np.r_[tracking_frame, tick_interval,
                                                    time.perf_counter()-send_started])
                    self.last_trace_at = now
                # Status is refreshed by the SDK's active-report stream.
        except Exception as exc:
            if isinstance(exc, InterruptedError) and self.stopped:
                # close() deliberately cancels an in-flight SDK send batch;
                # it then disables every motor. This is a normal shutdown.
                return
            if self.failure_diagnostics is None:
                self.failure_diagnostics = self.feedback_diagnostics()
            if self.manual_disable_only:
                self.enter_fault_hold(exc)
                self._send_fault_hold(robot)
                # UI/terminal code reports the fault; stdout cannot block the
                # sole sender while it continues issuing hold commands.
                return
            self._queue_trace('FAULT')
            self.fault = str(exc)
            self._cancel_target_worker()
            self.plan = None
            self.cartesian_settler = None
            # Disable before printing: a broken/blocked terminal must not delay it.
            try:
                self._disable_motors()
            except Exception:
                pass
            try:
                print(f'FAULT: {self.fault}; disable requested. Physically support arm.', flush=True)
            except (OSError, ValueError):
                pass

    def close(self):
        if self.manual_disable_only and self.enabled and not self.manual_disable_authorized:
            raise RuntimeError('Still holding: manual release required; sender left running')
        self.stopped = True
        self._cancel_target_worker()
        if self.target_executor is not None:
            self.target_executor.shutdown(wait=False, cancel_futures=True)
            self.target_executor = None
        errors = []
        try:
            if self.connected:
                try:
                    self.robot.stop_control_loop(timeout=1.)
                except Exception as exc:
                    errors.append(f'stop loop: {exc}')
                self._disable_motors()
                errors.extend(self.disable_errors)
                thread = getattr(self.robot, '_ctrl_thread', None)
                if isinstance(thread, threading.Thread) and thread.is_alive():
                    raise RuntimeError('SDK send thread did not stop; handles left open; use physical stop')
                errors.extend(self._stop_feedback_reporting())
                try:
                    self.robot.disconnect()  # No safe_home or zero-pose command.
                except Exception as exc:
                    errors.append(f'disconnect: {exc}')
                # Also covers SDK connect failing before _connected was set.
                for vendor, ctrl in list(getattr(self.robot, '_ctrl_map', {}).items()):
                    try:
                        ctrl.close()
                    except Exception as exc:
                        errors.append(f'close {vendor}: {exc}')
                self.connected = False
        finally:
            if hasattr(self.feedback, 'close'):
                self.feedback.close()
            self.enabled = False
        if errors:
            raise RuntimeError('; '.join(errors))

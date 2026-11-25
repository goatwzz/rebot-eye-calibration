"""Replay SDK transit targets on the existing single MIT sender.

No motor I/O or replacement IK/profile implementation. Ready retains every
RebotArmEndPose.move_to_traj sample. Home samples SDK JointMotion with the
safe_home duration and target-update cadence. Hover retains its own limits.
"""
from types import SimpleNamespace
import numpy as np


class SDKTransitPlan:
    mit_velocity_feedforward = False
    path_mode = 'sdk_transit'

    def __init__(self, points, duration, kind, target, source):
        self.points = np.asarray(points, float).copy()
        self.duration = float(duration)
        self.kind, self.target, self.source = kind, np.array(target, float), source
        if (self.points.ndim != 2 or self.points.shape[1] != 6 or len(self.points) < 2
                or not np.isfinite(self.points).all() or not np.isfinite(duration) or duration <= 0):
            raise ValueError('Invalid SDK transit samples')
        self.goal = self.points[-1].copy()
        self.check_grid = np.r_[np.arange(len(self.points))/len(self.points), 1.]
        self.diagnostics = {
            'source': source, 'target_update_hz': len(self.points)/self.duration,
            'sample_count': len(self.points), 'duration_s': self.duration,
            'max_adjacent_joint_step_rad': float(np.abs(np.diff(self.points, axis=0)).max()),
            'hover_speed_limits_applied': False,
        }

    def curve(self, fraction):
        # Same zero-order target hold as SDK _send_loop; no cubic interpolation,
        # no replanning from endpoints and no overshoot at joint limits.
        indices = np.minimum((np.clip(fraction, 0., 1.)*len(self.points)).astype(int), len(self.points)-1)
        return self.points[indices].copy()

    def sample(self, elapsed):
        return self.curve(np.asarray(elapsed/self.duration)), np.zeros(6)


def ready_plan(planner, raw, cancelled):
    from reBotArm_control_py.controllers import RebotArmEndPose

    class PlanningEndPose(RebotArmEndPose):
        def _send_loop(self, duration):
            # The SDK creates this thread; it must never send to hardware here.
            self._moving = False

    raw = planner.command_from_feedback(raw)
    model_start = raw + planner.offsets
    snapshot = SimpleNamespace(arm_control_mode='mit', has_gripper=False,
        groups={'arm': SimpleNamespace(num_joints=6)},
        get_state=lambda: (model_start.copy(), np.zeros(6), np.zeros(6)))
    ctrl = PlanningEndPose(snapshot)
    ctrl._model, ctrl._data, ctrl._end_frame_id = planner.model, planner.model.createData(), planner.tcp_id
    ctrl._running = True
    if cancelled():
        raise InterruptedError('Ready planning cancelled')
    ready = planner.cfg['ready_pose']
    if not ctrl.move_to_traj(**ready):
        raise ValueError('SDK ready pose IK/trajectory failed')
    ctrl._send_thread.join(timeout=1.)
    if cancelled():
        raise InterruptedError('Ready planning cancelled')
    target = np.array([ready[k] for k in ('x', 'y', 'z')])
    # SDK IK works in calibrated model angles; only RAW angles go to motors.
    points = [np.asarray(q)[:6]-planner.offsets for q in ctrl._traj]
    plan = SDKTransitPlan(points, ready['duration']*planner.cfg['transit']['time_scale'],
                          'ready', target, 'SDK RebotArmEndPose.move_to_traj complete samples')
    return validate_transit(planner, plan, raw, cancelled)


def home_plan(planner, raw, cancelled):
    if cancelled():
        raise InterruptedError('Home planning cancelled')
    from reBotArm_control_py.trajectory import JointMotion
    raw = planner.command_from_feedback(raw)
    goal = planner.validate_raw(np.zeros(6))
    cfg = planner.cfg['transit']
    # Exactly the SDK safe_home time scale (2*max_error/max_vel). JointMotion
    # owns the quintic calculation; no second local polynomial implementation.
    duration = max(2*float(np.abs(raw).max())/cfg['home_max_vel_rad_s'], 2/cfg['home_target_hz'])
    count = max(2, int(duration*cfg['home_target_hz']))
    motion = JointMotion(raw.copy(), goal, duration)
    points = [motion.sample(t)[0] for t in np.linspace(0., duration, count)]
    plan = SDKTransitPlan(points, duration*cfg['time_scale'], 'home', planner.tcp(goal),
                          'SDK JointMotion with RebotArmEndPose.safe_home timing')
    return validate_transit(planner, plan, raw, cancelled)


def validate_transit(planner, plan, raw, cancelled):
    from eyehand.click_hover import in_workspace
    start = planner.tcp(raw)
    if np.linalg.norm(start-plan.target) > planner.cfg['max_move_m']:
        raise ValueError('SDK transit exceeds configured max_move_m')
    previous = raw
    for point in plan.points:
        if cancelled():
            raise InterruptedError('Transit planning cancelled')
        q = planner.validate_raw(point)
        if np.max(np.abs(q-raw)) > planner.cfg['max_joint_move_rad']:
            raise ValueError('SDK transit joint displacement exceeds configured limit')
        if np.max(np.abs(q-previous)) > planner.cfg['tracking_limit_rad']:
            raise ValueError('SDK transit target step exceeds tracking limit')
        if not in_workspace(planner.tcp(q), planner.cfg):
            raise ValueError('SDK transit leaves configured workspace')
        planner.gravity_torque(q)
        previous = q
    if np.linalg.norm(planner.tcp(plan.goal)-plan.target) > .0002:
        raise ValueError('SDK transit final TCP check failed')
    return plan


def configure_transit(cfg):
    """Transit timing is independent of the slower click-hover limits."""
    cfg['transit'] = {**dict(time_scale=1., home_max_vel_rad_s=.5, home_target_hz=50.),
                      **cfg.get('transit', {})}
    transit = cfg['transit']
    if (not np.isfinite(list(transit.values())).all() or transit['time_scale'] < 1.
            or not 0 < transit['home_max_vel_rad_s'] <= .5 or transit['home_target_hz'] != 50.):
        raise ValueError('Invalid SDK transit timing; time_scale must be >= 1, home speed <= 0.5')

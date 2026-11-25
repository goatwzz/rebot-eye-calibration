"""Camera geometry and calibrated, position-only planning. No hardware access."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
import threading

import numpy as np
from scipy.interpolate import BPoly, CubicSpline, PPoly

from .geometry import checked
from .gravity_parameters import (_apply_payload_model, load_gravity_parameters,
                                 configure_gravity_model_exclusions)
from .kinematics import CorrectedKinematics, JOINT_NAMES

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=ROOT / 'click_hover_config.json', *, overrides=None):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    if overrides is not None:
        cfg.update(overrides)
    for key in ('result', 'gravity_config', 'sdk_root', 'log_dir'):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    vector = lambda value, n: np.asarray(value, dtype=float).shape == (n,) and np.isfinite(value).all()
    if not vector(cfg['tcp_offset_m'], 3) or np.linalg.norm(cfg['tcp_offset_m']) > .25:
        raise ValueError('tcp_offset_m must be three finite metres, within 0.25 m')
    for key in ('workspace_min_m', 'workspace_max_m'):
        if not vector(cfg[key], 3):
            raise ValueError(f'Invalid {key}')
    if np.any(np.array(cfg['workspace_min_m']) >= cfg['workspace_max_m']):
        raise ValueError('Invalid workspace bounds')
    for key in ('hover_m', 'max_move_m', 'max_joint_move_rad', 'max_joint_speed_rad_s',
                'max_joint_accel_rad_s2', 'max_tcp_speed_m_s', 'control_rate_hz',
                'feedback_max_age_s', 'tracking_limit_rad', 'settle_tolerance_m', 'settle_timeout_s',
                'ui_timeout_s', 'camera_max_age_s', 'home_tolerance_rad'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if not 20 <= cfg['control_rate_hz'] <= 500 or cfg['feedback_max_age_s'] > .2:
        raise ValueError('Control rate must be 20..500 Hz; feedback age <= 0.2 s')
    for key in ('kp', 'kd', 'torque_limit_nm'):
        if not vector(cfg[key], 6) or np.any(np.array(cfg[key]) <= 0):
            raise ValueError(f'{key} requires six positive values')
    if cfg['ui_timeout_s'] > 2 or cfg['camera_max_age_s'] > .5:
        raise ValueError('UI timeout must be <= 2 s and camera age <= 0.5 s')
    if cfg.get('planning_mode') not in ('joint', 'cartesian', 'sdk_cartesian'):
        raise ValueError('planning_mode must be joint, cartesian or sdk_cartesian')
    if cfg['home_tolerance_rad'] > .02:
        raise ValueError('Home tolerance must be <= 0.02 rad')
    if 'home_position_tolerance_m' in cfg:
        value = cfg['home_position_tolerance_m']
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value <= .015:
            raise ValueError('Home position tolerance must be positive and <= 0.015 m')
    # Legacy configs may still contain this demo-only setting. SDK policy is now
    # authoritative; do not expose/log an unused application tolerance.
    cfg.pop('feedback_limit_tolerance_rad', None)
    if cfg.get('tcp_frame') != 'gripper_end':
        raise ValueError('TCP offset must be expressed in gripper_end')
    if not cfg['serial'] or cfg.get('above_axis') != 'base_link:+Z':
        raise ValueError('Explicit camera serial and base_link:+Z required')
    cfg['_path'] = str(path)
    if 'cartesian_outer_loop' in cfg:
        from .cartesian_settle import validate_settings
        cfg['cartesian_outer_loop'] = validate_settings(cfg['cartesian_outer_loop'], cfg)
    from .sdk_transit import configure_transit
    configure_transit(cfg)
    from .table_reference import validate_table_settings
    validate_table_settings(cfg)
    return cfg


def import_sdk(root):
    """Choose modern SDK explicitly; never silently import another installed copy."""
    from .native_runtime import activate
    activate()  # Same upstream public snapshot runtime as the working robot project.
    root = Path(root).resolve()
    package = root / 'reBotArm_control_py'
    if not (package / 'actuator/rebotarm.py').exists():
        raise FileNotFoundError(f'SDK not found: {root}')
    sys.path.insert(0, str(root))
    sdk = importlib.import_module('reBotArm_control_py')
    if Path(sdk.__file__).resolve().parent != package:
        raise RuntimeError('Another SDK is already imported; use a fresh Python process')
    return importlib.import_module('reBotArm_control_py.kinematics.inverse_kinematics')


def depth_at_pixel(depth_m, u, v, radius=3, min_depth=.15, max_depth=2., max_spread=.012):
    """Reject holes and edges; never fill an invalid clicked centre from background."""
    d = np.asarray(depth_m, dtype=float)
    if d.ndim != 2 or not radius <= u < d.shape[1]-radius or not radius <= v < d.shape[0]-radius:
        raise ValueError('Pixel too close to image border')
    centre = d[v, u]
    patch = d[v-radius:v+radius+1, u-radius:u+radius+1]
    good = np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth)
    if not np.isfinite(centre) or not min_depth <= centre <= max_depth or good.mean() < .8:
        raise ValueError('Invalid/sparse depth: click a textured solid surface')
    values = patch[good]
    lo, hi = np.percentile(values, [5, 95])
    if hi-lo > max_spread or abs(centre-np.median(values)) > max_spread/2:
        raise ValueError('Depth discontinuity: click away from object boundaries')
    return float(np.median(values)), float(hi-lo)


def hover_target(T_base_camera, point_color, hover_m=.10):
    point_color = np.asarray(point_color, dtype=float)
    if point_color.shape != (3,) or not np.isfinite(point_color).all() or point_color[2] <= 0:
        raise ValueError('Invalid color-optical 3D point')
    T = checked(T_base_camera)
    surface = T[:3, :3] @ point_color + T[:3, 3]
    if not np.isfinite(hover_m) or hover_m <= 0:
        raise ValueError('Hover height must be positive and finite')
    return surface, surface + [0, 0, hover_m]


def in_workspace(point, cfg):
    return bool(np.all(np.asarray(point) >= cfg['workspace_min_m']) and
                np.all(np.asarray(point) <= cfg['workspace_max_m']))


class MotionPlan:
    def __init__(self, waypoints, cfg, distance, *, joint_space=False):
        waypoints = np.asarray(waypoints, dtype=float)
        self.sdk_motion = None
        if joint_space:
            from reBotArm_control_py.trajectory import JointMotion
            start, goal = waypoints[0], waypoints[-1]
            self.sdk_motion = JointMotion.plan(
                start, goal, max_speed=cfg['max_joint_speed_rad_s'],
                max_acceleration=cfg['max_joint_accel_rad_s2'], min_duration=3.)
            # Polynomial is only for existing path/derivative inspection.
            # Execution samples the SDK JointMotion, without a second profile.
            endpoints = [[start, np.zeros(6), np.zeros(6)],
                         [goal, np.zeros(6), np.zeros(6)]]
            self.curve = PPoly.from_bernstein_basis(BPoly.from_derivatives([0.,1.], endpoints))
        else:
            self.curve = CubicSpline(np.linspace(0, 1, len(waypoints)), waypoints,
                                     bc_type=((1, np.zeros(6)), (1, np.zeros(6))))
        self.path_mode = 'joint' if joint_space else 'cartesian'
        self.kind = 'hover'
        # Include polynomial extrema, so limits do not depend on a sampling grid.
        critical = [0., 1.]
        for derivative in (1, 2, 3):
            for roots in self.curve.derivative(derivative).roots(extrapolate=False):
                critical.extend(float(r) for r in roots if np.isfinite(r) and 0 <= r <= 1)
        self.check_grid = np.unique(np.r_[np.linspace(0, 1, 301), critical])
        grid = self.check_grid
        joint_duration = self.sdk_motion.duration if joint_space else max(
            3., float(np.max(np.abs(self.curve(grid, 1)))) / cfg['max_joint_speed_rad_s'],
            np.sqrt(float(np.max(np.abs(self.curve(grid, 2)))) / cfg['max_joint_accel_rad_s2']))
        self.duration = 1.01 * max(joint_duration, 2 * distance / cfg['max_tcp_speed_m_s'])
        self.goal = waypoints[-1].copy()
        self.target = None

    def sample(self, elapsed):
        if self.sdk_motion is not None:
            # Path inspection may lengthen duration to satisfy TCP speed.
            self.sdk_motion.duration = self.duration
            return self.sdk_motion.sample(elapsed)
        if elapsed >= self.duration:
            return self.goal.copy(), np.zeros(6)
        s = np.clip(elapsed/self.duration, 0, 1)
        return self.curve(s), self.curve(s, 1)/self.duration


class HoverPlanner:
    def __init__(self, cfg):
        import pinocchio as pin
        self.pin, self.cfg = pin, cfg
        self.ik = import_sdk(cfg['sdk_root'])
        result_bytes = Path(cfg['result']).read_bytes()
        self.result = json.loads(result_bytes)
        self.T_base_camera = checked(self.result['T_base_camera']).copy()
        if not np.allclose(checked(self.result['T_camera_base']) @ self.T_base_camera, np.eye(4), atol=1e-6):
            raise ValueError('Extrinsic transforms are not mutual inverses')
        self.calibration = self.result['provenance']
        if self.calibration.get('synthetic') is not False:
            raise ValueError('Synthetic/missing-provenance calibration is not deployable')
        if self.calibration['camera'].get('serial') != cfg['serial']:
            raise ValueError('Camera serial does not match selected calibration result')
        K = np.asarray(self.calibration['K'], dtype=float)
        if K.shape != (3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1]) <= 0:
            raise ValueError('Invalid calibrated RGB intrinsics')
        D = np.asarray(self.calibration['D'], dtype=float)
        if D.ndim != 1 or len(D) < 4 or not np.isfinite(D).all() or np.max(np.abs(D)) > 1e-12:
            raise ValueError('This RGB-D deployment requires zero-distortion calibrated RGB')
        if self.calibration['camera'].get('kind') != 'realsense':
            raise ValueError('Expected RealSense calibration provenance')
        correction = self.result['kinematic_correction']
        robot = {'urdf': correction['original_urdf'], 'urdf_sha256': correction['original_urdf_sha256'],
                 'ee_frame': correction['ee_frame'], 'base_frame': correction['base_frame'],
                 'joint_names': JOINT_NAMES}
        if robot['base_frame'] != 'base_link' or robot['ee_frame'] != 'gripper_end':
            raise ValueError('This demo requires base_link and gripper_end calibration frames')
        if self.calibration['robot']['urdf_sha256'] != correction['original_urdf_sha256']:
            raise ValueError('Calibration provenance and correction URDF disagree')
        source = CorrectedKinematics(robot, correction)
        self.offsets = source.offsets.copy()
        self._initialize_robot_model(source)
        self.result_sha256 = hashlib.sha256(result_bytes).hexdigest()

    def _initialize_robot_model(self, source):
        """Shared SDK IK, six-axis model and MIT gravity setup for either camera mount."""
        pin, cfg = self.pin, self.cfg
        # Freeze passive finger joints; official IK must not solve with finger DOFs.
        lock = [i for i in range(1, source.model.njoints) if source.model.names[i] not in JOINT_NAMES]
        self.model = pin.buildReducedModel(source.model, lock, pin.neutral(source.model))
        if self.model.nq != 6 or list(self.model.names[1:]) != JOINT_NAMES:
            raise ValueError('Expected reduced six-revolute-joint model')
        self.raw_lower = self.model.lowerPositionLimit.copy()
        self.raw_upper = self.model.upperPositionLimit.copy()
        # IK operates in corrected angles, limits and motor commands remain raw.
        self.model.lowerPositionLimit = self.raw_lower + self.offsets
        self.model.upperPositionLimit = self.raw_upper + self.offsets
        ee = self.model.getFrameId('gripper_end')
        frame = self.model.frames[ee]
        placement = frame.placement * pin.SE3(np.eye(3), np.array(cfg['tcp_offset_m']))
        self.tcp_id = self.model.addFrame(pin.Frame('hover_tcp', frame.parentJoint, ee,
                                                   placement, pin.FrameType.OP_FRAME))
        self.data = self.model.createData()
        # Private bare calibration model + configured point loads, exactly once.
        self._model = self.model.copy()
        self._data = self._model.createData()
        self.gravity = load_gravity_parameters(cfg['gravity_config'])
        _apply_payload_model(self, self.gravity)
        self.gravity_model, self.gravity_data = self._model, self._data
        self.gravity_lock = threading.Lock()
        # Reuse the SDK's joint feedback/target policy and gravity calculation.
        # Supply the calibrated model and this project's explicit load settings;
        # the SDK's default model/profile is not used for these calculations.
        from reBotArm_control_py.controllers.cartesian_impedance import ImpedanceLaw, ControlFault
        self.sdk_law = ImpedanceLaw([0.]*6, control_space='joint')
        self.sdk_law.lower, self.sdk_law.upper = self.raw_lower.copy(), self.raw_upper.copy()
        self.sdk_law.model, self.sdk_law.data = self.gravity_model, self.gravity_data
        self.sdk_law.q_indices = self.sdk_law.v_indices = np.arange(6)
        self.sdk_law.tau_scale = self.gravity['tau_scale'].copy()
        self._sdk_control_fault = ControlFault
        if cfg.get('gravity_excluded_urdf_links'):
            configure_gravity_model_exclusions(self,source.path,cfg['gravity_excluded_urdf_links'])

    def validate_raw(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all() or np.any(q < self.raw_lower-1e-10) or np.any(q > self.raw_upper+1e-10):
            raise ValueError('Raw joint command outside URDF limits')
        command = np.clip(q, self.raw_lower, self.raw_upper)
        self.sdk_law.validate_q(command, strict=True)
        return command

    def validate_feedback(self, q):
        q = np.asarray(q, dtype=float)
        try:
            self.sdk_law.validate_q(q)
        except self._sdk_control_fault as exc:
            # Keep the SDK policy; add joint names/limits for diagnosis only.
            details = '; '.join(f'{name}={np.degrees(value):+.4f}deg '
                                f'URDF=[{np.degrees(lo):.4f},{np.degrees(hi):.4f}]deg'
                                for name, value, lo, hi in zip(JOINT_NAMES, q, self.raw_lower, self.raw_upper)) if q.shape == (6,) else f'shape={q.shape}'
            raise ValueError(f'SDK joint feedback validation failed: {exc}; {details}') from exc
        return q

    def command_from_feedback(self, raw):
        return np.clip(self.validate_feedback(raw), self.raw_lower, self.raw_upper)

    def tcp(self, raw):
        q = self.validate_feedback(raw) + self.offsets
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.tcp_id].translation.copy()

    def gravity_torque(self, raw):
        with self.gravity_lock:
            tau = self.sdk_law.gravity(self.validate_feedback(raw) + self.offsets)
        tau = np.asarray(tau)
        if not np.isfinite(tau).all() or np.any(np.abs(tau) > self.cfg['torque_limit_nm']):
            raise ValueError(f'Gravity torque exceeds configured limits: {tau}')
        return tau.copy()

    def plan(self, raw, target, cancelled=lambda: False):
        if cancelled():
            raise InterruptedError('Planning cancelled')
        raw = np.clip(self.validate_feedback(raw), self.raw_lower, self.raw_upper)
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all() or not in_workspace(target, self.cfg):
            raise ValueError('Hover target outside configured base workspace')
        start = self.tcp(raw)
        distance = float(np.linalg.norm(target-start))
        if distance > self.cfg['max_move_m']:
            raise ValueError(f'Move {distance:.3f} m exceeds configured max_move_m')
        if self.cfg['planning_mode'] == 'sdk_cartesian':
            from .sdk_hover import hover_plan
            return hover_plan(self, raw, target, cancelled)
        if self.cfg['planning_mode'] == 'joint':
            return self._plan_position_joint(raw, target, cancelled)
        q = raw + self.offsets
        points = [raw.copy()]
        # Follow a straight TCP segment, retaining nearby posture without orientation constraint.
        for fraction in np.linspace(0, 1, max(3, int(np.ceil(distance/.004))+1))[1:]:
            if cancelled():
                raise InterruptedError('Planning cancelled')
            point = start + fraction*(target-start)
            out = self.ik.solve_ik(self.model, self.data, self.tcp_id,
                                   self.pin.SE3(np.eye(3), point), q,
                                   self.ik.IKParams(max_iter=300, tolerance=5e-5),
                                   controlled_joints=6, position_only=True)
            if not out.success:
                raise ValueError(f'Position-only IK failed: {out.error*1000:.3f} mm')
            q = out.q
            command = self.validate_raw(q-self.offsets)
            if np.max(np.abs(command-points[-1])) > .12:
                raise ValueError('IK branch/joint step too large')
            points.append(command.copy())
        if np.max(np.abs(points[-1]-raw)) > self.cfg['max_joint_move_rad']:
            raise ValueError('Joint displacement exceeds max_joint_move_rad')
        plan = MotionPlan(points, self.cfg, distance)
        plan.target = target.copy()
        return self._validate_plan(plan, raw, target, cancelled, straight=True)

    def _plan_position_joint(self, raw, target, cancelled):
        failures = []
        seeds = [raw.copy()]
        for elbow in ((.7,.5),(1.3,.8)):
            seed = raw.copy()
            seed[1:3] = elbow
            seeds.append(self.validate_raw(np.clip(seed,self.raw_lower,self.raw_upper)))
        for seed in seeds:
            if cancelled():
                raise InterruptedError('Planning cancelled')
            out = self.ik.solve_ik(self.model, self.data, self.tcp_id,
                                   self.pin.SE3(np.eye(3), target), seed+self.offsets,
                                   self.ik.IKParams(max_iter=1000,tolerance=5e-5),
                                   controlled_joints=6,position_only=True)
            if not out.success:
                failures.append(f'IK error {out.error*1000:.2f} mm')
                continue
            try:
                goal = self.validate_raw(out.q-self.offsets)
                plan = MotionPlan([raw,goal],self.cfg,float(np.linalg.norm(target-self.tcp(raw))),joint_space=True)
                plan.target = target.copy()
                return self._validate_plan(plan,raw,target,cancelled)
            except ValueError as exc:
                failures.append(str(exc))
        raise ValueError('No valid joint path to hover target: '+'; '.join(failures))

    def plan_home(self, raw, cancelled=lambda:False):
        from .sdk_transit import home_plan
        return home_plan(self, raw, cancelled)

    def _validate_plan(self, plan, raw, target, cancelled, *, straight=False):
        start = self.tcp(raw)
        peak_tcp_speed = 0.
        for s in plan.check_grid:
            if cancelled():
                raise InterruptedError('Planning cancelled')
            sample = self.validate_raw(plan.curve(s))
            position = self.tcp(sample)
            if np.max(np.abs(sample-raw)) > self.cfg['max_joint_move_rad']:
                raise ValueError('Intermediate joint displacement exceeds max_joint_move_rad')
            if straight and np.linalg.norm(position-(start+s*(target-start))) > .002:
                raise ValueError('Interpolated TCP path deviates by more than 2 mm from segment')
            self.pin.computeJointJacobians(self.model, self.data, sample+self.offsets)
            J = self.pin.getFrameJacobian(self.model, self.data, self.tcp_id, self.pin.LOCAL_WORLD_ALIGNED)[:3]
            peak_tcp_speed = max(peak_tcp_speed, float(np.linalg.norm(J @ plan.curve(s, 1))))
            if not in_workspace(position, self.cfg):
                raise ValueError('TCP path leaves configured workspace')
            self.gravity_torque(sample)
        plan.duration = max(plan.duration, 1.01*peak_tcp_speed/self.cfg['max_tcp_speed_m_s'])
        if np.linalg.norm(self.tcp(plan.goal)-target) > .0002:
            raise ValueError('Final calibrated TCP check failed')
        return plan

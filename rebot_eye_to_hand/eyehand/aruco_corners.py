"""RGB ArUco corners and SDK pose preference; no camera or actuator I/O."""
from dataclasses import dataclass
import time

import numpy as np

from .geometry import checked
from .vision import Target
from .click_hover import in_workspace
from .sdk_hover import hover_plan


@dataclass(frozen=True)
class CornerObservation:
    acquired: float
    raw_q: object
    epoch: int
    corners_base_m: np.ndarray
    corners_pixel: np.ndarray
    camera_target: np.ndarray
    pnp_rms_px: float
    ambiguity_gap_px: object

    def event(self):
        return dict(corners_base_m=self.corners_base_m.tolist(),
                    corners_pixel=self.corners_pixel.tolist(),
                    T_camera_target=self.camera_target.tolist(),
                    pnp_rms_px=self.pnp_rms_px,
                    ippe_second_solution_gap_px=self.ambiguity_gap_px,
                    raw_q_at_image_rad=None if self.raw_q is None else self.raw_q.tolist(),
                    depth_used=False)


class CornerDetector:
    def __init__(self, settings, calibration):
        self.settings = settings
        self.target = Target(dict(type='aruco', dictionary=settings['dictionary'],
                                  id=settings['marker_id'], marker_m=settings['marker_m']))
        self.k = np.array(calibration['K'], float)
        self.d = np.array(calibration['D'], float)

    def detect(self, frame, T_base_camera, raw_q=None, epoch=-1):
        result = self.target.detect(frame.color, self.k, self.d,
                                    max_rms=self.settings['max_pnp_rms_px'],
                                    ambiguity_gap=self.settings['ambiguity_gap_px'])
        c = checked(result['T_camera_target'])
        transform = checked(T_base_camera) @ c
        points = np.array(result['object_points'], float)
        base = points @ transform[:3, :3].T+transform[:3, 3]
        return CornerObservation(frame.acquired, None if raw_q is None else np.array(raw_q, copy=True),
                                 epoch, base, np.array(result['image_points']), c,
                                 result['pnp_rms_px'], result['ippe_second_solution_gap_px'])


class StableCorners:
    """Require several distinct stationary views, rather than a single PnP pose."""
    def __init__(self, samples=5, max_span_m=.003):
        self.samples, self.max_span_m = samples, max_span_m
        self.rows = []

    def clear(self):
        self.rows.clear()

    def update(self, observation):
        if self.rows and observation.acquired <= self.rows[-1].acquired:
            return None
        if self.rows and observation.epoch != self.rows[-1].epoch:
            self.clear()
        self.rows.append(observation)
        self.rows = self.rows[-self.samples:]
        if len(self.rows) < self.samples:
            return None
        xyz = np.array([row.corners_base_m for row in self.rows])
        # All four corresponding corner estimates must remain consistent.
        if np.max(np.linalg.norm(np.ptp(xyz, axis=0), axis=1)) > self.max_span_m:
            return None
        return observation


def corner_goal(observation, number, settings):
    if number not in (1, 2, 3, 4):
        raise ValueError('Corner key must be 1, 2, 3 or 4')
    point = observation.corners_base_m[number-1].copy()
    goal = point.copy()
    if settings['height_mode'] == 'corner':
        goal[2] += settings['clearance_m']
    else:
        goal[2] = settings['fixed_target_z_m']
    return point, goal


def confirm_observation(observation, controller, max_age_s, joint_span_rad, now=None):
    now = time.monotonic() if now is None else now
    if not 0 <= now-observation.acquired <= max_age_s:
        raise ValueError('ArUco observation expired; reacquire after reset')
    if controller.busy or controller.motion_epoch != observation.epoch:
        raise ValueError('Robot moved after this observation; reacquire after reset')
    raw = controller.snapshot()[:6]
    if observation.raw_q is None or np.max(np.abs(raw-observation.raw_q)) > joint_span_rad:
        raise ValueError('Robot is not stationary with the observed camera pose')
    return raw


def down_angle_deg(planner, raw):
    """The modeled fingertip outward direction is gripper_end local +X."""
    data = planner.model.createData()
    planner.pin.framesForwardKinematics(planner.model, data, raw+planner.offsets)
    direction = data.oMf[planner.tcp_id].rotation[:, 0]
    return float(np.degrees(np.arcsin(np.clip(-direction[2], -1., 1.))))


def plan_corner(planner, raw, goal, settings, cancelled=lambda: False):
    """Try steeper downward pitches first, using only the public SDK IK/path."""
    p = planner
    raw = p.command_from_feedback(raw)
    goal = np.asarray(goal, float)
    if goal.shape != (3,) or not np.isfinite(goal).all() or not in_workspace(goal, p.cfg):
        raise ValueError('Corner target outside configured workspace')
    if np.linalg.norm(goal-p.tcp(raw)) > p.cfg['max_move_m']:
        raise ValueError('Corner target exceeds configured maximum displacement')
    radial_yaw = float(np.arctan2(goal[1], goal[0]))
    seeds = [raw.copy()]
    for elbow in ((.7, .5), (1.3, .8)):
        seed = raw.copy(); seed[1:3] = elbow
        seeds.append(np.clip(seed, p.raw_lower, p.raw_upper))
    failures = []
    for pitch_deg in settings['down_pitches_deg']:
        for yaw_offset in (0., np.pi/4, -np.pi/4):
            if cancelled():
                raise InterruptedError('Corner approach planning cancelled')
            rotation = p.pin.rpy.rpyToMatrix(0., np.radians(pitch_deg), radial_yaw+yaw_offset)
            for seed in seeds:
                if cancelled():
                    raise InterruptedError('Corner approach planning cancelled')
                out = p.ik.solve_ik(p.model, p.model.createData(), p.tcp_id,
                    p.pin.SE3(rotation, goal), seed+p.offsets,
                    p.ik.IKParams(max_iter=300, tolerance=1e-7),
                    controlled_joints=6, position_only=False)
                if not out.success:
                    continue
                endpoint = p.validate_raw(out.q-p.offsets)
                try:
                    plan = hover_plan(p, raw, goal, cancelled, endpoint_goal=endpoint)
                except ValueError as exc:
                    failures.append(str(exc))
                    continue
                plan.diagnostics.update(tool_outward_axis='gripper_end:+X',
                    requested_down_pitch_deg=pitch_deg,
                    achieved_down_angle_deg=down_angle_deg(p, plan.goal),
                    down_preference='steepest feasible configured pitch, then relaxed',
                    orientation_outer_loop=False)
                return plan
    raise ValueError('No valid downward corner approach with SDK IK/trajectory; '
                     +' / '.join(failures[-2:]))

"""Official SDK geodesic sampler + CLIK, replayed by the existing SDK sender.

By default the terminal orientation comes from position-only IK. Callers may
instead supply an endpoint from the official full-pose IK. No local IK,
trajectory polynomial or motor I/O.
"""
import numpy as np

from .sdk_transit import SDKTransitPlan, validate_transit


class SDKHoverPlan(SDKTransitPlan):
    path_mode = 'sdk_cartesian'
    tracking_outer_loop = True

    def __init__(self, points, positions, duration, target, diagnostics):
        super().__init__(points, duration, 'hover', target,
                         'SDK plan_cartesian_geodesic_trajectory + track_trajectory')
        self.positions = np.asarray(positions, float)
        self.diagnostics.update(diagnostics, hover_speed_limits_applied=True,
                                tracking_outer_loop=True)

    def reference(self, elapsed):
        index = min(max(int(np.clip(elapsed/self.duration, 0., 1.)*len(self.points)), 0),
                    len(self.points)-1)
        return self.positions[index].copy()


def hover_plan(p, raw, target, cancelled, *, endpoint_goal=None):
    from reBotArm_control_py.trajectory import (
        JointMotion, TrajPlanParams, TrajProfile,
        plan_cartesian_geodesic_trajectory, track_trajectory)
    from reBotArm_control_py.trajectory.clik_tracker import IKParams

    cfg, pin = p.cfg, p.pin
    outer = cfg.get('cartesian_outer_loop', {})
    # Reserve the correction budget instead of exceeding the existing combined
    # nominal + correction speed/acceleration limits during motion.
    reserve = outer if outer.get('enabled') else {}
    speed = cfg['max_joint_speed_rad_s']-reserve.get('max_joint_speed_rad_s', 0.)
    accel = cfg['max_joint_accel_rad_s2']-reserve.get('max_joint_accel_rad_s2', 0.)
    tcp_speed = cfg['max_tcp_speed_m_s']-reserve.get('max_tcp_speed_m_s', 0.)
    if min(speed, accel, tcp_speed) <= 0:
        raise ValueError('SDK trajectory requires a positive nominal motion budget')
    data = p.model.createData()

    def pose(q):
        pin.framesForwardKinematics(p.model, data, q+p.offsets)
        return data.oMf[p.tcp_id].copy()

    start_pose = pose(raw)
    seeds = [raw.copy()]
    if endpoint_goal is None:
        for elbow in ((.7, .5), (1.3, .8)):
            seed = raw.copy(); seed[1:3] = elbow
            seeds.append(np.clip(seed, p.raw_lower, p.raw_upper))
    failures = []
    for seed in seeds:
        if cancelled():
            raise InterruptedError('SDK hover planning cancelled')
        if endpoint_goal is None:
            result = p.ik.solve_ik(p.model, data, p.tcp_id, pin.SE3(np.eye(3), target),
                                  seed+p.offsets, p.ik.IKParams(max_iter=1000, tolerance=1e-7),
                                  controlled_joints=6, position_only=True)
            if not result.success:
                failures.append(f'endpoint IK {result.error*1000:.3f} mm')
                continue
            goal = p.validate_raw(result.q-p.offsets)
        else:
            goal = p.validate_raw(endpoint_goal)
        end_pose = pose(goal)
        if np.linalg.norm(end_pose.translation-target) > .0002:
            raise ValueError('SDK endpoint does not match the requested TCP position')
        duration = max(JointMotion.plan(raw, goal, max_speed=speed,
                       max_acceleration=accel, min_duration=3.).duration,
                       2*np.linalg.norm(target-start_pose.translation)/tcp_speed)
        for attempt in range(4):
            if cancelled():
                raise InterruptedError('SDK hover planning cancelled')
            cart = plan_cartesian_geodesic_trajectory(
                start_pose, end_pose, duration,
                TrajPlanParams(dt=.01, profile=TrajProfile.MIN_JERK))
            # Tighten the SDK's own convergence tolerance for slow target
            # sampling. Its default 1e-4 can hold several points then jump.
            # null_gain=0 is track_trajectory's public default; hard limits
            # remain enforced by the SDK and by validate_transit below.
            points = track_trajectory(p.model, p.tcp_id, cart.trajectory, raw+p.offsets,
                                      IKParams(tolerance=1e-7), null_gain=0.)
            if cancelled():
                raise InterruptedError('SDK hover planning cancelled')
            if not all(pt.ik_success for pt in points):
                failures.append('SDK CLIK did not converge at every sample')
                break
            qs = np.array([pt.q[:6]-p.offsets for pt in points])
            positions = np.array([pose(q).translation for q in qs])
            # Reproduce the official target-hold interval duration / N.
            dt = duration/len(qs)
            qd = np.diff(qs, axis=0)/dt
            qdd = np.diff(np.vstack((np.zeros((1, 6)), qd, np.zeros((1, 6)))), axis=0)/dt
            speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1)/dt
            scale = max(float(np.abs(qd).max())/speed,
                        np.sqrt(float(np.abs(qdd).max())/accel), float(speeds.max())/tcp_speed)
            if scale > 1.:
                duration *= 1.03*scale
                continue
            plan = SDKHoverPlan(qs, positions, duration, target, dict(
                sampler='official SDK SE3 geodesic MIN_JERK, dt=0.01s',
                clik_tolerance=1e-7, clik_null_gain=0., clik_success_count=len(qs),
                nominal_peak_joint_speed_rad_s=float(np.abs(qd).max()),
                nominal_peak_joint_accel_rad_s2=float(np.abs(qdd).max()),
                nominal_peak_tcp_speed_m_s=float(speeds.max()),
                terminal_orientation=('position_only_endpoint_IK' if endpoint_goal is None
                                      else 'official_full_pose_endpoint_IK'),
                z_outer_feedback=False if outer.get('axes') == 'xy' else bool(reserve)))
            try:
                return validate_transit(p, plan, raw, cancelled)
            except ValueError as exc:
                failures.append(str(exc))
                break
        else:
            failures.append('SDK sampled trajectory could not satisfy motion limits')
    raise ValueError('No valid SDK Cartesian hover path: '+'; '.join(failures))

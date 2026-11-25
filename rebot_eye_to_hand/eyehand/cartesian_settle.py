"""Bounded position settling using the official SDK CLIK; no hardware I/O."""
from __future__ import annotations

import numpy as np


DEFAULTS = dict(enabled=False, during_motion=True, axes='xyz', max_uncontrolled_z_drift_m=.0005,
                gain_per_s=1.0, tolerance_m=.001,
                deadband_m=.0004, max_tcp_speed_m_s=.003,
                max_joint_speed_rad_s=.02, max_joint_accel_rad_s2=.1,
                max_joint_offset_rad=.015, max_tcp_offset_m=.008,
                capture_radius_m=.012, timeout_s=10., tracking_rate_hz=100.,
                tracking_filter_s=.08, tracking_warmup_s=.5)


def validate_settings(raw, motion):
    if not isinstance(raw, dict) or set(raw)-set(DEFAULTS):
        raise ValueError('Invalid cartesian_outer_loop settings')
    cfg = {**DEFAULTS, **raw}
    if not isinstance(cfg['during_motion'], bool):
        raise ValueError('cartesian_outer_loop.during_motion must be boolean')
    if not isinstance(cfg['enabled'], bool):
        raise ValueError('cartesian_outer_loop.enabled must be boolean')
    if cfg['axes'] not in ('xy', 'xyz'):
        raise ValueError('cartesian_outer_loop.axes must be xy or xyz')
    for key in set(DEFAULTS)-{'enabled','during_motion','axes'}:
        if not isinstance(cfg[key], (int, float)) or not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'Invalid cartesian_outer_loop.{key}')
    if not cfg['deadband_m'] < cfg['tolerance_m'] <= motion['settle_tolerance_m']:
        raise ValueError('Outer deadband < tolerance <= existing settle tolerance required')
    if (cfg['max_uncontrolled_z_drift_m'] > .001
            or cfg['gain_per_s'] > 2 or cfg['timeout_s'] > 20
            or cfg['max_tcp_speed_m_s'] > min(.005, motion['max_tcp_speed_m_s'])
            or cfg['max_joint_speed_rad_s'] > min(.03, motion['max_joint_speed_rad_s'])
            or cfg['max_joint_accel_rad_s2'] > motion['max_joint_accel_rad_s2']
            or cfg['max_joint_offset_rad'] > min(.02, motion['tracking_limit_rad']/2)
            or cfg['max_tcp_offset_m'] > .01 or cfg['capture_radius_m'] > .02
            or not 20 <= cfg['tracking_rate_hz'] <= min(100., motion['control_rate_hz'])
            or not .02 <= cfg['tracking_filter_s'] <= .5
            or not .1 <= cfg['tracking_warmup_s'] <= 2.):
        raise ValueError('Outer correction exceeds configured local settling bounds')
    return cfg


class CartesianSettler:
    """Integrate feedback-derived SDK IK increments into the held joint command.

    Solving IK repeatedly for the same nominal goal would not cancel a static
    inner-loop bias. Here each small SDK descent step starts from *feedback*,
    and its increment is added to the previous *command*. Hard bounds override
    rate smoothing at a limit; discarded increments are never accumulated.
    """
    def __init__(self, planner, nominal, target):
        import pinocchio as pin
        self.pin, self.planner = pin, planner
        self.cfg = planner.cfg['cartesian_outer_loop']
        self.axes = self.cfg.get('axes', 'xyz')
        self.mask = np.array([1., 1., 0. if self.axes == 'xy' else 1.])
        self.feedback_tcp = self.residual = None
        self.nominal, self.target = nominal.copy(), target.copy()
        self.command = nominal.copy()
        self.data = planner.model.createData()  # Never share GUI/trajectory data.
        self.nominal_tcp = self.fk(nominal)
        limit = self.cfg['max_joint_offset_rad']
        self.lower = np.maximum(planner.raw_lower, nominal-limit)
        self.upper = np.minimum(planner.raw_upper, nominal+limit)
        self.velocity = np.zeros(6)
        self.last = None
        self.initial_error = self.minimum_error = None
        self.error = 0.
        self.updates = self.saturation_count = 0
        self.limit_reason = self.stop_reason = None
        self.peak_joint_speed = self.peak_tcp_speed = 0.

    def fk(self, raw):
        pin, p = self.pin, self.planner
        pin.forwardKinematics(p.model, self.data, raw+p.offsets)
        pin.updateFramePlacements(p.model, self.data)
        return self.data.oMf[p.tcp_id].translation.copy()

    def step(self, raw, now):
        p, cfg = self.planner, self.cfg
        self.feedback_tcp = self.fk(raw)
        self.residual = self.target-self.feedback_tcp
        self.error = float(np.linalg.norm(self.mask*self.residual))
        if self.initial_error is None:
            self.initial_error = self.minimum_error = self.error
        if self.error > cfg['capture_radius_m']:
            self.stop_reason = 'capture_radius_exceeded'
        self.minimum_error = min(self.minimum_error, self.error)
        dt = 0. if self.last is None else min(max(now-self.last, 0.), .02)
        self.last = now
        if self.stop_reason or dt == 0:
            return self.command.copy()
        desired = np.zeros(6)
        residual = self.mask*self.residual
        if hasattr(self, 'filtered_residual'):
            alpha = dt/(cfg['tracking_filter_s']+dt)
            self.filtered_residual += alpha*(residual-self.filtered_residual)
            residual = self.filtered_residual
        if np.linalg.norm(residual) > cfg['deadband_m']:
            seed = raw+p.offsets
            out = p.ik.solve_ik(
                p.model, self.data, p.tcp_id,
                # XY mode has no Z feedback error: the SDK target keeps the
                # current measured Z while correcting only the horizontal error.
                self.pin.SE3(np.eye(3), self.feedback_tcp+residual), seed,
                p.ik.IKParams(max_iter=1, tolerance=1e-7, step_size=1., damping=1e-6),
                controlled_joints=6, position_only=True)
            # One descent iteration need not report full IK convergence.
            if out.q.shape != (6,) or not np.isfinite(out.q).all() or not np.isfinite(out.error):
                raise ValueError('Nonfinite SDK outer-loop IK result')
            if out.error < self.error:
                desired = getattr(self, 'gain_scale', 1.)*cfg['gain_per_s']*(out.q-seed)
        peak = float(np.max(np.abs(desired)))
        desired *= min(1., cfg['max_joint_speed_rad_s']/max(peak, 1e-15))
        self.pin.computeJointJacobians(p.model, self.data, self.command+p.offsets)
        self.pin.updateFramePlacements(p.model, self.data)
        J = self.pin.getFrameJacobian(p.model, self.data, p.tcp_id,
                                     self.pin.LOCAL_WORLD_ALIGNED)[:3, :6]
        desired *= min(1., cfg['max_tcp_speed_m_s']/max(float(np.linalg.norm(J@desired)), 1e-15))
        accel = cfg['max_joint_accel_rad_s2']*dt
        if self.axes == 'xy':
            # Keep the coordinated SDK joint direction. Per-joint clipping can
            # create a vertical component even when the SDK Z error is zero.
            change = desired-self.velocity
            blend = min(1., accel/max(float(np.max(np.abs(change))), 1e-15))
            velocity = self.velocity+blend*change
        else:
            velocity = self.velocity+np.clip(desired-self.velocity, -accel, accel)
        velocity *= min(1., cfg['max_tcp_speed_m_s']/max(float(np.linalg.norm(J@velocity)), 1e-15))
        if self.axes == 'xy':
            # Scale the coordinated step at a joint bound instead of clipping
            # individual joints, which would introduce unintended Z motion.
            increment = velocity*dt
            fraction = 1.
            for i, value in enumerate(increment):
                if value > 0:
                    fraction = min(fraction, (self.upper[i]-self.command[i])/value)
                elif value < 0:
                    fraction = min(fraction, (self.lower[i]-self.command[i])/value)
            candidate = np.clip(self.command+np.clip(fraction, 0., 1.)*increment, self.lower, self.upper)
        else:
            candidate = np.clip(self.command+velocity*dt, self.lower, self.upper)
        self.limit_reason = None
        if not np.allclose(candidate, self.command+velocity*dt, atol=1e-12, rtol=0):
            self.limit_reason = 'joint_offset_or_urdf_limit'
        tcp = self.fk(candidate)
        if (self.axes == 'xy'
                and abs(tcp[2]-self.nominal_tcp[2]) > cfg['max_uncontrolled_z_drift_m']):
            self.limit_reason = 'uncontrolled_z_drift_limit'
            candidate = self.command.copy()
        if np.linalg.norm(tcp-self.nominal_tcp) > cfg['max_tcp_offset_m']:
            self.limit_reason = 'tcp_offset_limit'
            candidate = self.command.copy()
        if np.any(tcp < p.cfg['workspace_min_m']) or np.any(tcp > p.cfg['workspace_max_m']):
            self.limit_reason = 'workspace_limit'
            candidate = self.command.copy()
        candidate = p.validate_raw(candidate)
        # Anti-windup: retain only the actually applied, bounded increment.
        self.velocity = (candidate-self.command)/dt
        self.peak_joint_speed = max(self.peak_joint_speed, float(np.max(np.abs(self.velocity))))
        self.peak_tcp_speed = max(self.peak_tcp_speed, float(np.linalg.norm(J@self.velocity)))
        self.command = candidate.copy()
        self.updates += 1
        self.saturation_count += int(self.limit_reason is not None)
        return self.command.copy()

    def diagnostics(self):
        return dict(enabled=True, axes=self.axes, z_feedback_enabled=self.axes == 'xyz',
                    feedback_error_xyz_mm=None if self.residual is None else (1000*self.residual).tolist(),
                    uncontrolled_command_z_drift_mm=float(1000*(self.fk(self.command)[2]-self.nominal_tcp[2])),
                    solver='official_sdk_position_clik_single_step_integrated',
                    initial_fk_error_mm=None if self.initial_error is None else 1000*self.initial_error,
                    minimum_fk_error_mm=None if self.minimum_error is None else 1000*self.minimum_error,
                    tolerance_mm=1000*self.cfg['tolerance_m'], updates=self.updates,
                    command_offset_rad=(self.command-self.nominal).tolist(),
                    command_tcp_offset_mm=(1000*(self.fk(self.command)-self.nominal_tcp)).tolist(),
                    peak_joint_command_speed_rad_s=self.peak_joint_speed,
                    peak_tcp_command_speed_m_s=self.peak_tcp_speed,
                    saturation_count=self.saturation_count,
                    last_limit_reason=self.limit_reason, stop_reason=self.stop_reason)


class CartesianTracker(CartesianSettler):
    """Transport a bounded correction along the SDK's moving reference.

    The 500 Hz sender replays nominal samples with the held bias. Only the
    slower outer updates call the existing official position-IK step.
    """
    def __init__(self, planner, nominal, reference):
        super().__init__(planner, nominal, reference)
        self.filtered_residual = np.zeros(3)
        self.gain_scale = 0.
        self.started = self.updated_at = None
        self.trajectory_updates = self.settling_updates = 0
        self.tracking_peak_error = 0.
        self.settling_initial_error = None

    def advance(self, nominal, reference, raw, now, *, settling=False):
        from reBotArm_control_py.trajectory.joint_motion import minimum_jerk_profile
        if self.started is None:
            self.started = now
        bias = self.command-self.nominal
        command = nominal+bias
        nominal_tcp = self.fk(nominal)
        if not np.array_equal(nominal, self.nominal) and np.linalg.norm(bias) > 1e-12:
            # A fixed joint bias changes its Cartesian direction as the arm
            # moves. Transport its selected Cartesian displacement with the
            # public SDK IK; XY mode also keeps its vertical displacement zero.
            displacement = self.mask*(self.fk(self.command)-self.nominal_tcp)
            # Moving the nominal can carry a bound-saturated joint bias just
            # outside the URDF limits. The SDK evaluates q_init before clamping
            # trial steps, so give it a feasible seed for its active-limit solve.
            seed = np.clip(command, self.planner.raw_lower, self.planner.raw_upper)
            out = self.planner.ik.solve_ik(
                self.planner.model, self.data, self.planner.tcp_id,
                self.pin.SE3(np.eye(3), nominal_tcp+displacement), seed+self.planner.offsets,
                self.planner.ik.IKParams(max_iter=20, tolerance=1e-8),
                controlled_joints=6, position_only=True)
            if not out.success or not np.isfinite(out.q).all():
                self.stop_reason = 'SDK_correction_transport_failed'
                return self.command.copy()
            command = out.q-self.planner.offsets
        # Anti-windup applies to transported bias too. Reaching its local
        # offset bound must not abort an otherwise valid nominal trajectory.
        # Scale the coordinated vector, as the existing XY settler does.
        bias = command-nominal
        limit = self.cfg['max_joint_offset_rad']
        low = np.maximum(self.planner.raw_lower-nominal, -limit)
        high = np.minimum(self.planner.raw_upper-nominal, limit)
        fraction = 1.
        for j, value in enumerate(bias):
            if value > 0: fraction = min(fraction, high[j]/value)
            elif value < 0: fraction = min(fraction, low[j]/value)
        fraction = float(np.clip(fraction, 0., 1.))
        for _ in range(24):
            command = nominal+fraction*bias
            tcp = self.fk(command)
            valid = (np.linalg.norm(tcp-nominal_tcp) <= self.cfg['max_tcp_offset_m']
                and (self.axes != 'xy' or abs(tcp[2]-nominal_tcp[2]) <= self.cfg['max_uncontrolled_z_drift_m'])
                and np.all(tcp >= self.planner.cfg['workspace_min_m'])
                and np.all(tcp <= self.planner.cfg['workspace_max_m']))
            if valid: break
            fraction *= .5
        else:
            self.stop_reason = 'moving_reference_correction_bound'
            return self.command.copy()
        if fraction < 1.:
            self.velocity *= fraction
            self.saturation_count += 1
        self.nominal, self.nominal_tcp = nominal.copy(), nominal_tcp
        self.target, self.command = reference.copy(), command
        self.lower = np.maximum(self.planner.raw_lower, nominal-limit)
        self.upper = np.minimum(self.planner.raw_upper, nominal+limit)
        if self.updated_at is None or now-self.updated_at >= 1/self.cfg['tracking_rate_hz']-1e-9:
            self.gain_scale = minimum_jerk_profile(now-self.started, self.cfg['tracking_warmup_s'])[0]
            self.updated_at = now
            self.trajectory_updates += int(not settling)
            self.settling_updates += int(settling)
            command = self.step(raw, now)
            if settling and self.settling_initial_error is None:
                self.settling_initial_error = self.error
            elif not settling:
                self.tracking_peak_error = max(self.tracking_peak_error, self.error)
            return command
        return self.command.copy()

    @property
    def quiet(self):
        # Do not freeze a correction that is still changing the joint target.
        return bool(np.max(np.abs(self.velocity)) <= 1e-6)

    def diagnostics(self):
        result = super().diagnostics()
        result.update(mode='moving_reference_then_settle',
                      trajectory_updates=self.trajectory_updates,
                      settling_updates=self.settling_updates,
                      tracking_rate_hz=self.cfg['tracking_rate_hz'],
                      tracking_peak_error_mm=1000*self.tracking_peak_error,
                      settling_initial_error_mm=(None if self.settling_initial_error is None
                                                 else 1000*self.settling_initial_error),
                      filtered_error_mm=(1000*self.filtered_residual).tolist(),
                      correction_stopped=self.quiet)
        return result

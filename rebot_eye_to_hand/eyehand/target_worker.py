"""Prepare SDK outer-loop targets away from the sole SDK CAN sender.

Like RebotArmEndPose's target producer, this worker only publishes targets.
It never sends commands, queries motors, or owns a robot handle. The existing
CartesianTracker still owns all IK, bounds, filtering and anti-windup.
"""
from __future__ import annotations

import copy
import time

import numpy as np


class BufferedCartesianTracker:
    max_age_s = .05

    def __init__(self, planner, nominal, reference, executor, factory=None):
        from .cartesian_settle import CartesianTracker
        self.planner, self.executor = planner, executor
        self.factory = factory or CartesianTracker
        self.worker_tracker = None  # Accessed only by the single worker.
        self.initial = nominal.copy(), reference.copy()
        self.future = None
        self.submitted_at = None
        self.closed = False
        self.nominal, self.target = nominal.copy(), reference.copy()
        self.command, self.velocity = nominal.copy(), np.zeros(6)
        self.axes = planner.cfg['cartesian_outer_loop']['axes']
        self.mask = np.array([1., 1., 0. if self.axes == 'xy' else 1.])
        self.stop_reason = None
        self.error, self.residual = 0., None
        self.quiet = False
        self.data = planner.model.createData()  # Sender-only FK data.
        self.state = {}
        self.accepted = 0
        self.peak_compute_s = self.peak_result_age_s = 0.

    def fk(self, raw):
        p = self.planner
        p.pin.framesForwardKinematics(p.model, self.data, raw+p.offsets)
        return self.data.oMf[p.tcp_id].translation.copy()

    def _compute(self, nominal, reference, raw, now, settling):
        started = time.perf_counter()
        if self.worker_tracker is None:
            self.worker_tracker = self.factory(self.planner, *self.initial)
        tracker = self.worker_tracker
        command = tracker.advance(nominal, reference, raw, now, settling=settling)
        return dict(command=command.copy(), nominal=tracker.nominal.copy(),
                    target=tracker.target.copy(), velocity=tracker.velocity.copy(),
                    quiet=tracker.quiet, stop_reason=tracker.stop_reason, error=tracker.error,
                    diagnostics=copy.deepcopy(tracker.diagnostics()),
                    compute_s=time.perf_counter()-started)

    def advance(self, nominal, reference, raw, now, *, settling=False):
        if self.closed:
            self.stop_reason = 'target_worker_cancelled'
            return self.command.copy()
        if self.future is not None:
            age = now-self.submitted_at
            if age > self.max_age_s:
                self.stop_reason = 'target_worker_timeout'
                self.cancel()
                return self.command.copy()
            if self.future.done():
                result = self.future.result()  # Nonblocking; propagate SDK failures.
                self.future = None
                self.command, self.nominal = result['command'], result['nominal']
                self.target, self.velocity = result['target'], result['velocity']
                self.quiet, self.stop_reason = result['quiet'], result['stop_reason']
                self.state = result['diagnostics']
                self.error = result['error']
                self.accepted += 1
                self.peak_compute_s = max(self.peak_compute_s, result['compute_s'])
                self.peak_result_age_s = max(self.peak_result_age_s, age)
        period = 1./self.planner.cfg['cartesian_outer_loop']['tracking_rate_hz']
        if (self.future is None and not self.stop_reason
                and (self.submitted_at is None or now-self.submitted_at >= period-1e-9)):
            self.submitted_at = now
            self.future = self.executor.submit(self._compute, nominal.copy(), reference.copy(),
                                               raw.copy(), now, settling)
        return self.command.copy()

    def cancel(self):
        self.closed = True
        if self.future is not None:
            self.future.cancel()

    def diagnostics(self):
        result = copy.deepcopy(self.state)
        result.update(target_producer='separate_single_worker_no_motor_io',
                      accepted_targets=self.accepted,
                      target_worker_max_age_ms=1000*self.max_age_s,
                      peak_target_compute_ms=1000*self.peak_compute_s,
                      peak_target_result_age_ms=1000*self.peak_result_age_s,
                      stop_reason=self.stop_reason)
        return result

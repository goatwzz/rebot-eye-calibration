"""GUI workflow only; execution uses the shared HoverController and SDK plans."""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from .backend import dependencies
from pathlib import Path
_, _geo = dependencies(Path(__file__).resolve().parents[2] / 'rebot_eye_to_hand')
from eyehand.hover_runtime import PlanWorker


class StationaryWindow:
    def __init__(self, seconds=.5, joint_span=.002):
        self.seconds, self.joint_span = seconds, joint_span
        self.anchor = self.since = None

    def update(self, q, now):
        q = np.asarray(q, float)
        if self.anchor is None or np.max(np.abs(q-self.anchor)) > self.joint_span:
            self.anchor, self.since = q.copy(), now
        return now-self.since >= self.seconds


@dataclass(frozen=True)
class WristFrame:
    color: object
    depth: object
    acquired: float
    raw_q: object
    epoch: int


def click_pose(frame, controller, max_age, joint_span, now=None):
    now = time.monotonic() if now is None else now
    if frame.raw_q is None:
        raise ValueError('Preview has no measured robot pose; run --execute for motion')
    if not 0 <= now-frame.acquired <= max_age:
        raise ValueError('Clicked image expired')
    if controller.busy or frame.epoch != controller.motion_epoch:
        raise ValueError('Robot moved or the displayed click was cancelled')
    current = controller.snapshot()[:6]
    if np.max(np.abs(current-frame.raw_q)) > joint_span:
        raise ValueError('Robot pose changed since this image; wait and click again')
    return np.array(frame.raw_q, copy=True)


class DemoFlow:
    def __init__(self, planner, controller, emit, worker=None):
        self.planner, self.controller, self.emit = planner, controller, emit
        self.worker = worker or PlanWorker()
        self.phase = 'IDLE'
        self.pending = None
        self.exiting = self.done = self.home_confirmed = False
        self.error = None

    @property
    def can_click(self):
        return (self.phase == 'WAIT_CLICK' and not self.exiting and self.pending is None
                and not self.controller.busy and not self.worker.active and self.worker.results.empty())

    def _cancel(self, reason):
        self.worker.cancel()
        self.pending = None
        self.controller.hold(reason)

    def ready(self):
        if self.exiting or self.phase in ('READY_PENDING', 'READY_PLANNING', 'READY_MOVING'):
            return False
        self._cancel('READY_REQUESTED_HOLD')
        self.error = None
        self.pending, self.phase = ('ready', None), 'READY_PENDING'
        return True

    def hover(self, target, event):
        if not self.can_click:
            raise ValueError('Press r to return to the initial pose and wait for WAIT_CLICK before clicking')
        self.pending, self.phase = ('hover', (np.array(target, copy=True), event)), 'HOVER_PENDING'

    def exit(self, reason='USER_EXIT'):
        if self.exiting:
            return
        self.exiting = True
        self._cancel('EXIT_HOME_REQUESTED_HOLD')
        self.pending, self.phase = ('home', None), 'HOME_PENDING'
        self.emit({'event': 'exit_requested', 'reason': reason})

    def hold(self):
        self._cancel('USER_HOLD')
        if self.exiting:
            self.phase, self.done = 'EXIT_HOME_CANCELLED_HOLD', True
            self.error = 'Automatic home cancelled; support arm then release'
        else:
            self.phase = 'HOLD_PRESS_R'

    def fail(self, reason):
        self._cancel('FAULT_HOLD')
        self.error, self.done, self.phase = str(reason), True, 'FAULT'

    def pump(self):
        complete = self.controller.take_completion()
        if complete:
            self.emit({'event': 'completion', 'phase': self.phase, **complete})
            status = complete['status']
            if status == 'HOME_REACHED_HOLD' and self.phase == 'HOME_MOVING':
                self.home_confirmed, self.done, self.phase = True, True, 'HOME_CONFIRMED'
            elif status == 'SETTLED_MODEL_ONLY' and self.phase == 'READY_MOVING':
                self.phase = 'WAIT_CLICK'
            elif status == 'SETTLED_MODEL_ONLY' and self.phase == 'HOVER_MOVING':
                self.phase = 'HOVER_HOLD_PRESS_R'
            elif status in ('HOME_TIMEOUT_HOLD', 'SETTLE_TIMEOUT_HOLD',
                            'TRACKING_REFUSED_HOLD', 'UI_TIMEOUT_HOLD'):
                self.error = status
                if self.exiting:
                    self.phase, self.done = status, True
                else:
                    self.phase = ('HOLD_TIMEOUT_PRESS_R' if 'TIMEOUT' in status else 'HOLD_PRESS_R')
        outcome = self.worker.poll()  # Drain even cancelled results before a new plan.
        if outcome is not None:
            context, plan, error = outcome
            try:
                if error:
                    raise error
                event = {**context.get('event', {}), 'event': context['kind']+'_plan',
                         'raw_goal_rad': plan.goal.tolist(), 'duration_s': plan.duration,
                         'execution_requested': True}
                if hasattr(plan, 'diagnostics'):
                    event['sdk_transit'] = plan.diagnostics
                self.emit(event)  # Persist before issuing motion.
                self.controller.move(plan, expected_epoch=context['epoch'])
                self.phase = context['kind'].upper()+'_MOVING'
            except (ValueError, RuntimeError, InterruptedError, OSError) as exc:
                self.controller.hold('PLAN_REFUSED_HOLD')
                self.error = str(exc)
                self.emit({'event': 'plan_refused', 'kind': context['kind'], 'error': str(exc)})
                self.phase = 'HOME_REFUSED_HOLD' if self.exiting else 'HOLD_PRESS_R'
                if self.exiting:
                    self.done = True
        if self.done or self.pending is None or self.worker.active or not self.worker.results.empty():
            return
        kind, value = self.pending
        self.pending = None
        q = self.controller.planning_start()
        epoch = self.controller.motion_epoch
        if kind == 'ready':
            fn = lambda cancel: self.planner.plan_ready(q, cancelled=cancel)
            event = {'ready_pose': self.planner.cfg['ready_pose']}
        elif kind == 'home':
            fn = lambda cancel: self.planner.plan_home(q, cancelled=cancel)
            event = {}
        else:
            target, event = value
            fn = lambda cancel: self.planner.plan(q, target, cancelled=cancel)
        self.worker.submit(fn, {'kind': kind, 'epoch': epoch, 'event': event})
        self.phase = kind.upper()+'_PLANNING'

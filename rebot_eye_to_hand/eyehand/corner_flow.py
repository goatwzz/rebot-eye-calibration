"""Four-corner UI sequence; all trajectories and motor I/O use existing SDK code."""
import time
import numpy as np

from .aruco_corners import corner_goal, confirm_observation, plan_corner
from .hover_runtime import PlanWorker


class CornerFlow:
    def __init__(self, planner, controller, settings, emit, worker=None, clock=time.monotonic):
        self.planner, self.controller, self.settings, self.emit = planner, controller, settings, emit
        self.worker, self.clock = worker or PlanWorker(), clock
        self.phase, self.pending, self.selected = 'WAIT_KEY', None, None
        self.exiting = self.done = self.home_confirmed = False
        self.error, self.reset_at, self.target = None, float('inf'), None

    @property
    def can_request(self):
        return (not self.exiting and self.phase in ('WAIT_KEY', 'CORNER_HOLD', 'HOLD')
                and not self.controller.busy and not self.worker.active and self.worker.results.empty())

    def _cancel(self, reason):
        self.worker.cancel()
        self.pending = None
        self.controller.hold(reason)
        self.selected, self.reset_at = None, float('inf')

    def reset(self, number=None):
        if not self.can_request:
            return False
        if number is not None and number not in (1, 2, 3, 4):
            raise ValueError('Corner must be 1..4')
        self._cancel('RESET_REQUESTED_HOLD')
        self.selected, self.error, self.target = number, None, None
        kind = 'ready' if self.settings['mount'] == 'wrist' else 'home'
        self.pending, self.phase = (kind, None), 'RESET_PENDING'
        self.emit(dict(event='reset_requested', kind=kind, corner=number))
        return True

    def observe(self, observation):
        if self.phase != 'WAIT_MARKER' or self.selected is None:
            return False
        if observation.acquired < self.reset_at:
            return False
        confirm_observation(observation, self.controller,
                            self.planner.cfg['camera_max_age_s'], self.settings['joint_span_rad'],
                            now=self.clock())
        corner, goal = corner_goal(observation, self.selected, self.settings)
        self.target = goal.copy()
        event = dict(event='corner_acquired', corner=self.selected, **observation.event(),
                     corner_base_m=corner.tolist(), target_base_m=goal.tolist(),
                     height_mode=self.settings['height_mode'], clearance_m=self.settings['clearance_m'],
                     physical_accuracy_verified=False)
        self.emit(event)
        self.pending, self.phase = ('corner', (goal, event)), 'CORNER_PENDING'
        return True

    def exit(self, reason='USER_EXIT'):
        if self.exiting:
            return
        self.exiting = True
        self._cancel('EXIT_HOME_REQUESTED_HOLD')
        self.pending, self.phase = ('home', None), 'HOME_PENDING'
        self.emit(dict(event='exit_requested', reason=reason))

    def hold(self):
        self._cancel('USER_HOLD')
        self.phase = 'HOLD'
        if self.exiting:
            self.done, self.error = True, 'Exit home cancelled; support arm then release'

    def fail(self, reason):
        self._cancel('FAULT_HOLD')
        self.error, self.done, self.phase = str(reason), True, 'FAULT'

    def pump(self):
        complete = self.controller.take_completion()
        if complete:
            self.emit(dict(event='completion', phase=self.phase, **complete))
            status = complete['status']
            reset_done = ((status in ('HOME_REACHED_HOLD', 'HOME_COMMAND_FINISHED_HOLD') and self.phase == 'HOME_MOVING')
                          or (status == 'SETTLED_MODEL_ONLY' and self.phase == 'READY_MOVING'))
            if reset_done:
                if self.exiting:
                    self.home_confirmed = status == 'HOME_REACHED_HOLD'
                    self.done, self.phase = True, 'HOME_FINISHED'
                else:
                    self.reset_at = self.clock()
                    self.phase = 'WAIT_MARKER' if self.selected else 'WAIT_KEY'
            elif status == 'SETTLED_MODEL_ONLY' and self.phase == 'CORNER_MOVING':
                self.phase = 'CORNER_HOLD'
            elif status in ('HOME_TIMEOUT_HOLD', 'SETTLE_TIMEOUT_HOLD',
                            'TRACKING_REFUSED_HOLD', 'UI_TIMEOUT_HOLD'):
                self.error, self.selected, self.pending = status, None, None
                self.worker.cancel()
                self.phase = 'HOLD'
                if self.exiting:
                    self.done = True
        outcome = self.worker.poll()
        if outcome is not None:
            context, plan, error = outcome
            try:
                if error:
                    raise error
                self.emit(dict(event=context['kind']+'_plan', corner=self.selected,
                               target_base_m=np.asarray(plan.target).tolist(),
                               raw_goal_rad=plan.goal.tolist(), duration_s=plan.duration,
                               diagnostics=plan.diagnostics, execution_requested=True))
                self.controller.move(plan, expected_epoch=context['epoch'])
                self.phase = context['kind'].upper()+'_MOVING'
            except (ValueError, RuntimeError, InterruptedError, OSError) as exc:
                self.controller.hold('PLAN_REFUSED_HOLD')
                self.error, self.selected, self.pending = str(exc), None, None
                self.emit(dict(event='plan_refused', kind=context['kind'], error=str(exc)))
                self.phase = 'HOLD'
                if self.exiting:
                    self.done = True
        if self.done or self.pending is None or self.worker.active or not self.worker.results.empty():
            return
        kind, value = self.pending
        self.pending = None
        q, epoch = self.controller.planning_start(), self.controller.motion_epoch
        if kind == 'ready':
            fn = lambda cancel: self.planner.plan_ready(q, cancelled=cancel)
        elif kind == 'home':
            fn = lambda cancel: self.planner.plan_home(q, cancelled=cancel)
        else:
            goal, _ = value
            fn = lambda cancel: plan_corner(self.planner, q, goal, self.settings, cancel)
        self.worker.submit(fn, dict(kind=kind, epoch=epoch))
        self.phase = kind.upper()+'_PLANNING'

"""Nonblocking camera/planner handoff. Contains no camera or CAN constructors."""
from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time


class RGBDFramePending(RuntimeError):
    """Transient incomplete/unsynchronized frame; does not refresh image freshness."""


@dataclass(frozen=True)
class DisplayFrame:
    color: object
    depth: object
    acquired: float


class LatestFrames:
    def __init__(self, camera):
        self.camera = camera
        self.lock = threading.Lock()
        self.frame = None
        self.error = None
        self.pending_reason = None
        self.rejected_frames = 0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name='hover-rgbd')

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            while not self.stopped.is_set():
                try:
                    color, depth = self.camera.read()
                except RGBDFramePending as exc:
                    with self.lock:
                        self.pending_reason = str(exc)
                        self.rejected_frames += 1
                    continue
                frame = DisplayFrame(color, depth, time.monotonic())
                with self.lock:
                    self.frame = frame
                    self.pending_reason = None
        except Exception as exc:
            if not self.stopped.is_set():
                with self.lock:
                    self.error = str(exc)

    def latest(self, max_age):
        with self.lock:
            frame, error, pending = self.frame, self.error, self.pending_reason
        if error:
            raise RuntimeError(f'RGB-D acquisition failed: {error}')
        if frame is None:
            return None
        if not 0 <= time.monotonic()-frame.acquired <= max_age:
            raise RuntimeError(f'RGB-D frame expired; motion cancelled; last issue: {pending or "no new frame"}')
        return frame

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=1.)
        try:
            self.camera.close()
        finally:
            self.thread.join(timeout=1.)
        if self.thread.is_alive():
            raise RuntimeError('RGB-D worker did not stop')


class PlanWorker:
    """At most one Pinocchio planning thread. Cancellation invalidates its result."""
    def __init__(self):
        self.thread = None
        self.cancelled = threading.Event()
        self.results = queue.Queue()
        self.generation = 0

    @property
    def active(self):
        return self.thread is not None and self.thread.is_alive()

    def submit(self, function, context):
        if self.active or not self.results.empty():
            raise RuntimeError('Planning is still active')
        token = self.generation
        self.cancelled = threading.Event()
        cancellation = self.cancelled

        def run():
            try:
                result = function(cancellation.is_set)
                self.results.put((token, context, result, None))
            except Exception as exc:
                self.results.put((token, context, None, exc))
        self.thread = threading.Thread(target=run, daemon=True, name='hover-ik')
        self.thread.start()

    def poll(self):
        if self.active:
            return None
        try:
            token, context, result, error = self.results.get_nowait()
        except queue.Empty:
            return None
        if token != self.generation:
            return None
        return context, result, error

    def cancel(self):
        self.generation += 1
        self.cancelled.set()
        # Leave the old worker alive until its cooperative cancellation returns.
        # A new worker must not share Pinocchio data with it.


class ClickGate:
    def __init__(self, unavailable):
        self.unavailable = unavailable
        self.pending = None

    def offer(self, u, v, frame):
        # Check at event receipt: clicks during motion/planning are never deferred.
        if frame is None or self.pending is not None or self.unavailable():
            return False
        self.pending = (u, v, frame)
        return True

    def take(self):
        pending, self.pending = self.pending, None
        return pending

    def clear(self):
        self.pending = None


def release_supported(controller, input_fn=input):
    if getattr(controller,'manual_disable_only',False):
        return _manual_release(controller,input_fn)
    if not controller.enabled or controller.fault:
        return
    print('Holding arm. Physically support arm, then type release in this terminal.', flush=True)
    commands = queue.Queue()

    def read_terminal():
        while True:
            try:
                command = input_fn('support then release> ')
                commands.put(command)
                if command.strip().lower() == 'release':
                    return
            except Exception as exc:
                commands.put(exc)
                return
    threading.Thread(target=read_terminal, daemon=True, name='hover-release-input').start()
    while True:
        try:
            controller.heartbeat()  # Also detect a dead/faulted SDK loop while waiting.
            command = commands.get(timeout=.05)
            if isinstance(command, Exception):
                print('Terminal unavailable; disable requested.', flush=True)
                return
            if command.strip().lower() == 'release':
                return
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            print('Still holding. Support arm, then type release.', flush=True)


def _manual_release(controller,input_fn):
    if not controller.enabled:return
    controller.hold('EXIT_HOLD')
    print('[保持] 不自动失能。托稳机械臂后，输入 release 失能并退出。',flush=True)
    commands=queue.Queue()
    processed=threading.Event()
    def read_terminal():
        while True:
            try:
                command=input_fn('support then release> ')
                processed.clear();commands.put(command);processed.wait()
                if controller.manual_disable_authorized:return
            except Exception as exc:
                commands.put(exc);return
    threading.Thread(target=read_terminal,daemon=True,name='manual-release-input').start()
    transport_was_failed=False
    while True:
        try:
            controller.heartbeat(allow_fault=True)
            failed=bool(controller.hold_transport_error)
            if failed and not transport_was_failed:
                print('[保持未确认] '+controller.hold_transport_error+'；没有请求失能，但链路异常时无法保证保持。',flush=True)
            elif transport_was_failed and not failed:
                print('[保持] 发送已恢复，故障仍锁定；不会自动继续运动。',flush=True)
            transport_was_failed=failed
            command=commands.get(timeout=.05)
            if isinstance(command,Exception):
                print('[保持] 终端输入不可用；不会自动失能，保持发送继续。',flush=True)
                continue
            try:
                text=command.strip().lower()
                if text=='release':
                    controller.authorize_manual_disable()
                    return
            finally:processed.set()
        except queue.Empty:continue
        except KeyboardInterrupt:
            processed.set()
            print('[保持] Ctrl+C不失能；托稳机械臂后输入 release。',flush=True)
        except Exception as exc:
            controller.enter_fault_hold(exc)

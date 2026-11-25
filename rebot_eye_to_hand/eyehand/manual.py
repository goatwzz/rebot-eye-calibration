"""Opt-in hand guiding: gravity compensation while dragging, PD + gravity to hold.

Inspired by rebot_grasp/scripts/collect_handeye_eih.py --manual. Uses fresh
mechPos feedback, explicit drag/hold, and no automatic homing. This is an
active robot controller; do not run alongside another motor controller.
"""
from __future__ import annotations

import sys
import queue
import threading
import time

import numpy as np

from .hardware import RebotReader


class ManualGravityReader(RebotReader):
    uses_explicit_modes = True
    def __init__(self, *args, rate=40.0, kp=8.0, kd=1.5, startup_timeout=15., **kwargs):
        if not sys.stdin.isatty():
            raise ValueError("--manual requires an interactive terminal for supported release")
        if not 10 <= rate <= 100 or not 0 < kp <= 30 or not 0 < kd <= 5:
            raise ValueError("Invalid manual control gains/rate")
        if not 0 < startup_timeout <= 60:
            raise ValueError("Startup timeout must be in (0,60] seconds")
        self.startup_timeout = startup_timeout
        self._active = False
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._thread = None
        self.command_queue = None
        self._enabled = []
        self._mode, self._fault = "hold", None
        self._snapshot, self._target = None, None
        self._tau = np.zeros(6)
        self._last_hold_error_at = 0.
        self.rate, self.kp, self.kd = rate, kp, kd
        self.tau_limits = np.array([6., 20., 10., 3., 2., 1.])
        super().__init__(*args, **kwargs)
        self.metadata.update(source="manual-gravity-mechPos-0x7019", manual_rate_hz=rate,
                             hold_kp=kp, damping_kd=kd, torque_limits_nm=self.tau_limits.tolist(),
                             gripper_controlled=False)
        # Kinematic/FK data belongs to the feedback reader; dynamics has its own data.
        self._gravity_data = self.model.createData()

    def _gravity(self, q6):
        q = self.pin.neutral(self.model)
        q[self.joint_indices] = q6
        gravity = self.pin.computeGeneralizedGravity(self.model, self._gravity_data, q)
        # All six selected joints are scalar revolute joints (validated by parent).
        indices = [self.model.joints[self.model.getJointId(n)].idx_v for n in self.metadata["joint_names"]]
        tau = np.asarray(gravity)[indices]
        if not np.isfinite(tau).all():
            raise ValueError("Nonfinite gravity torque")
        return np.clip(tau, -self.tau_limits, self.tau_limits)

    def start(self):
        from motorbridge import Mode

        print("MANUAL is an active controller for six arm joints. Stop other motor controllers first.\n"
              "Support the arm during startup. Gripper is not enabled or moved.\n"
              "g=gravity drag; h=hold; s/Enter=hold then collect; q=finish collection.\n"
              "After collection, holding continues until you support the arm and type release.", flush=True)
        pose, info = super().wait_for_feedback(self.startup_timeout)
        self._target = np.asarray(info["q_rad"], dtype=float)
        self._tau = self._gravity(self._target)
        self._snapshot = (pose, info)
        self._active = True
        self._started_at = time.monotonic()
        try:
            for motor in self.motors:
                motor.ensure_mode(Mode.MIT, 1000)
            # Mode changes can each block for up to a second. Never enable with
            # the old pre-mode-change target or an already-expired torque ramp.
            pose, info = super().read()
            self._target = np.asarray(info["q_rad"], dtype=float)
            self._tau = self._gravity(self._target)
            self._snapshot = (pose, info)
            self._started_at = time.monotonic()
            for i, motor in enumerate(self.motors):
                # Prime a current-position target; no trajectory or joint-zero operation.
                motor.send_mit(float(self._target[i]), 0., self.kp, self.kd, 0.)
                self._enabled.append(motor)
                motor.enable()
                motor.send_mit(float(self._target[i]), 0., self.kp, self.kd, 0.)
        except BaseException as exc:
            self._latch_fault(f"Startup interrupted: {exc}")
            self._launch_worker()
            raise
        self._launch_worker()

    def _launch_worker(self):
        self._thread = threading.Thread(target=self._worker, name="rebot-gravity", daemon=True)
        self._thread.start()

    def _latch_fault(self, reason):
        with self._lock:
            first = self._fault is None
            self._fault = str(reason)
            self._mode = "hold"
            if first and self._snapshot is not None:
                self._target = np.asarray(self._snapshot[1]["q_rad"]).copy()
        if first:
            print(f"\n[Manual FAULT] {reason}. Sampling/dragging blocked; attempting hold. "
                  "Support the robot; use physical emergency stop if needed.", flush=True)

    def _send(self):
        with self._lock:
            q_target = self._target.copy()
            mode = self._mode
            tau = self._tau.copy()
        ramp = min(1., max(0., (time.monotonic() - self._started_at) / 2.))
        kp = 0. if mode == "drag" else self.kp
        failures = []
        for i, motor in enumerate(self.motors):
            if motor in self._enabled:
                try:
                    motor.send_mit(float(q_target[i]), 0., kp, self.kd, float(tau[i] * ramp))
                except Exception as exc:
                    failures.append(f"{self.metadata['joint_names'][i]}: {exc}")
        if failures:
            raise RuntimeError("MIT send failed (all enabled joints attempted): " + "; ".join(failures))

    def _worker(self):
        previous, previous_time = None, None
        while not self._stop.is_set():
            tick = time.monotonic()
            try:
                pose, info = super().read()
                q = np.asarray(info["q_rad"])
                timestamp = (info["read_started_monotonic_s"] + info["read_ended_monotonic_s"]) / 2
                tau = self._gravity(q)
                if previous is not None:
                    velocity = np.max(np.abs(q - previous)) / max(timestamp - previous_time, 1e-4)
                    if velocity > 1.0:
                        self._latch_fault(f"Measured joint speed {velocity:.2f}rad/s exceeds 1.0rad/s")
                previous, previous_time = q.copy(), timestamp
                with self._lock:
                    self._snapshot = (pose, info)
                    self._tau = tau
                    if self._mode == "drag" and self._fault is None:
                        self._target = q.copy()
                self._send()
            except Exception as exc:
                self._latch_fault(exc)
                try:
                    self._send()
                except Exception as send_exc:
                    # Visible, latched failure; no automatic reset or torque-off.
                    if time.monotonic() - self._last_hold_error_at >= 1.:
                        print(f"[Manual] Hold command failed: {send_exc}", flush=True)
                        self._last_hold_error_at = time.monotonic()
            self._stop.wait(max(0., 1. / self.rate - (time.monotonic() - tick)))

    def set_mode(self, mode):
        if mode not in ("drag", "hold"):
            raise ValueError("Unknown manual mode")
        with self._lock:
            if self._fault:
                raise RuntimeError(f"Manual fault: {self._fault}")
            if time.monotonic() - self._started_at < 2.:
                raise RuntimeError("Wait for the 2s gravity startup ramp before dragging/sampling")
            if self._snapshot is None or time.monotonic() - self._snapshot[1]["read_started_monotonic_s"] > 0.3:
                raise RuntimeError("Manual feedback stale")
            if mode != self._mode:
                self._target = np.asarray(self._snapshot[1]["q_rad"]).copy()
                self._mode = mode
        print(f"[Manual] {mode.upper()}", flush=True)

    @property
    def status(self):
        with self._lock:
            return f"MANUAL {self._mode.upper()}" + (f" FAULT: {self._fault}" if self._fault else "")

    def read(self):
        with self._lock:
            if self._fault:
                raise RuntimeError(f"Manual fault: {self._fault}")
            if self._snapshot is None:
                raise RuntimeError("Manual control not started")
            pose, info = self._snapshot
            age = time.monotonic() - info["read_started_monotonic_s"]
            if not 0 <= age <= 0.3:
                raise RuntimeError(f"Manual feedback stale ({age:.3f}s)")
            return pose.copy(), {**info, "manual_mode": self._mode, "age_s": age}

    def release_supported(self):
        """Operator has physically supported the robot; disable before freeing bus."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.)
            if self._thread.is_alive():
                raise RuntimeError("Control thread did not stop; bus retained, use hardware emergency stop")
        failures = []
        for motor in self._enabled:
            try:
                motor.disable()
            except Exception as exc:
                failures.append(str(exc))
        if failures:
            raise RuntimeError(f"Disable failed; use hardware emergency stop: {failures}")
        self._enabled.clear()
        self._active = False
        super().close()

    def close(self):
        if not self._active:
            super().close()
            return
        # A GUI exit/camera exception must not silently drop gravity support or home.
        with self._lock:
            self._mode = "hold"
            if self._snapshot is not None:
                self._target = np.asarray(self._snapshot[1]["q_rad"]).copy()
        print("\nCollection ended. Holding continues; no automatic homing.\n"
              "Support the arm on a stable fixture. Type release then Enter to disable and exit.", flush=True)
        while True:
            try:
                command = input() if self.command_queue is None else self.command_queue.get(timeout=0.2)
                if command is None:
                    raise EOFError
                if command.strip().lower() == "release":
                    self.release_supported()
                    return
                print("Holding. Support robot, then type release.", flush=True)
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                print("Ctrl+C does not release holding. Support robot, then type release.", flush=True)
            except EOFError:
                # Terminal loss cannot be treated as authorization to disable.
                print("Terminal closed; attempting hold. Use physical robot stop/support.", flush=True)
                self._stop.wait(1.)

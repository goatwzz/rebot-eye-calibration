"""Shared hand-guiding controller extracted from the original wrist collector.

SDK connection, mode changes, MIT commands and Pinocchio dynamics are reused.
The original gravity/integral/velocity-lock law is unchanged. Failed SDK mode
verification is checked before enable_all. Importing this module opens no bus.
"""
from __future__ import annotations
import threading
import time
import numpy as np

class GravityCompController:
    """MIT mode with end-effector velocity locking for hand-guided poses.

    Reference: reBotArm_control_py/example/10_gravity_compensation_lock.py
    """
    KP = 8.0
    KD = 1.5
    V_THRESH  = 0.04   # End-effector linear velocity threshold (m/s)
    W_THRESH  = 0.08   # End-effector angular velocity threshold (rad/s)

    def __init__(self, arm: RebotArm) -> None:
        from reBotArm_control_py.dynamics import compute_generalized_gravity
        from reBotArm_control_py.kinematics import (
            load_robot_model, get_end_effector_frame_id,
        )
        from reBotArm_control_py.kinematics.robot_model import pad_q_for_model
        import pinocchio as pin

        self._compute_gravity = compute_generalized_gravity
        self._pad_q_for_model = pad_q_for_model
        self._pin = pin

        self._arm = arm
        if not self._arm.has_gripper:
            raise ValueError(
                "Hardware config is missing groups.gripper. "
                "Enable the gripper group in the selected hardware YAML under "
                "reBotArm_control_py/config."
            )
        self._model = load_robot_model()
        self._data  = self._model.createData()
        self._ee_id = get_end_effector_frame_id(self._model)

        self._n = None          # Joint count, set after connect.
        self._q_target = None
        self._integral = None
        self._gc_running = threading.Event()
        self._io_lock = threading.RLock()

    def start(self) -> None:
        """Connect, enable, set MIT mode, and start gravity compensation."""
        self._arm.connect()
        print("[GravityComp] Connected")

        arm_mode_ok = self._arm.arm.mode_mit(
            kp=np.full(self._arm.arm.num_joints, self.KP),
            kd=np.full(self._arm.arm.num_joints, self.KD),
        )
        if not arm_mode_ok:
            raise RuntimeError("Arm MIT mode verification failed; enable_all was not called")
        if not self._arm.gripper.mode_mit():
            raise RuntimeError("Gripper MIT mode verification failed; enable_all was not called")
        self._arm.enable_all()
        print("[GravityComp] Enabled")

        n = self._arm.arm.num_joints
        self._n = n
        self._wait_state_valid()
        with self._io_lock:
            q0 = self._arm.get_state()[0][:n]
        self._q_target = q0.copy()
        self._integral = np.zeros(n)

        print(f"[GravityComp] MIT mode, kp={self.KP} kd={self.KD}. Move the arm by hand.")

        self._gc_running.set()
        self._arm.start_control_loop(self._worker, rate=self._arm.rate)

    def _wait_state_valid(self, timeout: float = 2.0) -> None:
        """Wait until all motor feedback is available."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._io_lock:
                self._arm.get_state()
                ok = all(
                    m.get_state() is not None
                    for m in self._arm._motor_map.values()
                )
            if ok:
                return
            time.sleep(0.02)
        raise RuntimeError("[GravityComp] Arm feedback is not ready")

    def safe_home(self) -> None:
        """Stop gravity compensation, home with an SDK controller, then disconnect."""
        self._gc_running.clear()
        try:
            print("[GravityComp] Homing...")
            self._arm.stop_control_loop()
            with self._io_lock:
                q_now = self._arm.get_state()[0][: self._n]
                g_now = self._arm.gripper.get_positions()

            from reBotArm_control_py.controllers import RebotArmEndPose
            ctrl = RebotArmEndPose(
                self._arm,
                arm_control_mode="mit",
                use_gravity_ff=True,
            )
            ctrl.set_gripper_target(float(g_now[0]) if g_now.size else 0.0)
            ctrl._q_target[:] = q_now
            ctrl.start()
            ctrl.safe_home()
            self._arm.stop_control_loop()
        except Exception as e:
            print(f"[GravityComp] Homing failed: {e}")
        try:
            self._arm.disconnect()
        except Exception:
            pass
        print("[GravityComp] Disconnected")

    def _worker(self, r, dt: float) -> None:
        if not self._gc_running.is_set():
            return

        pin = self._pin
        model, data, ee_id = self._model, self._data, self._ee_id
        n = self._n
        KP, KD = self.KP, self.KD

        try:
            with self._io_lock:
                q_all, qd_all, _ = r.get_state()
            q = q_all[:n]
            qd = qd_all[:n]
            q_model = self._pad_q_for_model(model, q, n)
            tau_g = self._compute_gravity(model=model, q=q_model)[:n]

            q_err = self._q_target - q
            self._integral += q_err
            np.clip(self._integral, -0.5, 0.5, out=self._integral)

            qd_model = np.zeros(model.nv)
            qd_model[: min(model.nv, n)] = qd[: min(model.nv, n)]
            pin.computeJointJacobians(model, data, q_model)
            pin.updateFramePlacements(model, data)
            J = pin.getFrameJacobian(model, data, ee_id, pin.ReferenceFrame.WORLD)
            v = J @ qd_model

            if (np.linalg.norm(v[:3]) > self.V_THRESH or
                    np.linalg.norm(v[3:]) > self.W_THRESH):
                self._q_target = q.copy()
                self._integral *= 0.9

            with self._io_lock:
                r.arm.send_mit(
                    pos=self._q_target,
                    vel=np.zeros(n),
                    kp=np.full(n, KP),
                    kd=np.full(n, KD),
                    tau=tau_g + self._integral,
                )
                r.gripper.send_mit(r.gripper.get_positions())
        except Exception:
            pass

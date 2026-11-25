from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import sys
import threading
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import numpy as np
import pinocchio as pin

from eyehand.cli import main
from eyehand.hardware import DEFAULT_SDK
from eyehand.reference_manual import DEFAULT_REFERENCE, load_reference_class
from eyehand.gravity_parameters import apply_gravity_parameters, load_gravity_parameters


class ReferenceControllerTests(unittest.TestCase):
    def test_shared_class_is_imported_from_one_source(self):
        cls, digest = load_reference_class()
        self.assertEqual(digest, hashlib.sha256(DEFAULT_REFERENCE.read_bytes()).hexdigest())
        self.assertEqual(cls._worker.__code__.co_filename, str(DEFAULT_REFERENCE.resolve()))
        self.assertEqual(cls.KP, 8)
        self.assertEqual(cls.KD, 1.5)

    def test_original_start_uses_sdk_enable_feedback_order_and_rate(self):
        cls, _ = load_reference_class()
        calls = []

        class Group:
            num_joints = 6

            def mode_mit(self, **kwargs):
                calls.append("mode")
                return True

        class Motor:
            def get_state(self):
                return object()

        class Arm:
            arm, gripper = Group(), Group()
            rate = 500
            _motor_map = {"joint1": Motor()}

            def connect(self):
                calls.append("connect")

            def enable_all(self):
                calls.append("enable_all")

            def get_state(self):
                calls.append("state")
                return np.zeros(7), np.zeros(7), np.zeros(7)

            def start_control_loop(self, callback, rate):
                calls.append(("control_loop", rate))

        ctrl = cls.__new__(cls)
        ctrl._arm, ctrl._io_lock, ctrl._gc_running = Arm(), threading.RLock(), threading.Event()
        with contextlib.redirect_stdout(io.StringIO()):
            ctrl.start()
        self.assertEqual(calls[:4], ["connect", "mode", "mode", "enable_all"])
        self.assertEqual(calls[-1], ("control_loop", 500))
        self.assertTrue(ctrl._gc_running.is_set())

    def test_original_worker_retains_integral_auto_lock_and_gripper(self):
        cls, _ = load_reference_class()
        model = pin.buildModelFromUrdf(str(DEFAULT_SDK / "urdf/RS/urdf/ReBot_Arm_RS.urdf"))
        q = np.array([.1, .3, .5, .1, -.1, .2])
        sent, gripper_sent = [], []

        class Group:
            def send_mit(self, **kwargs):
                sent.append(kwargs)

        class Gripper:
            def get_positions(self):
                return np.array([.2])

            def send_mit(self, value):
                gripper_sent.append(value)

        class Arm:
            arm, gripper = Group(), Gripper()
            velocity = np.zeros(6)

            def get_state(self):
                return q.copy(), self.velocity.copy(), np.zeros(6)

        ctrl = cls.__new__(cls)
        ctrl._pin, ctrl._model, ctrl._data = pin, model, model.createData()
        ctrl._ee_id, ctrl._n = model.getFrameId("gripper_end"), 6
        ctrl._pad_q_for_model = lambda m, values, n: np.r_[values[:n], np.zeros(m.nq - n)]
        ctrl._compute_gravity = lambda model, q: pin.computeGeneralizedGravity(model, model.createData(), q)
        ctrl._q_target, ctrl._integral = np.zeros(6), np.zeros(6)
        ctrl._io_lock, ctrl._gc_running = threading.RLock(), threading.Event()
        ctrl._gc_running.set()
        arm = Arm()
        ctrl._worker(arm, .002)
        self.assertEqual(len(sent), 1)
        np.testing.assert_equal(sent[-1]["pos"], np.zeros(6))
        np.testing.assert_allclose(ctrl._integral, -q)
        np.testing.assert_equal(sent[-1]["kp"], np.full(6, 8.))
        np.testing.assert_equal(sent[-1]["kd"], np.full(6, 1.5))
        arm.velocity = np.array([0., .8, 0., 0., 0., 0.])
        ctrl._worker(arm, .002)
        self.assertEqual(len(sent), 2)
        np.testing.assert_allclose(sent[-1]["pos"], q)
        np.testing.assert_allclose(ctrl._integral, np.clip(-2 * q, -.5, .5) * .9)
        self.assertEqual(len(gripper_sent), 2)
        # Check actual sent torque using the independently configured settings.
        parameters = load_gravity_parameters()
        apply_gravity_parameters(ctrl, parameters)
        apply_gravity_parameters(ctrl, parameters)  # must not apply scaling twice
        ctrl._integral.fill(0.)
        ctrl._q_target = q.copy()
        arm.velocity = np.zeros(6)
        ctrl._worker(arm, .002)
        self.assertEqual(len(sent), 3)
        q_model = np.r_[q, np.zeros(model.nq - 6)]
        gravity = pin.computeGeneralizedGravity(ctrl._model, ctrl._model.createData(), q_model)[:6]
        np.testing.assert_allclose(sent[-1]["tau"], gravity * parameters["tau_scale"])
        np.testing.assert_equal(sent[-1]["kp"], np.full(6, parameters["kp"]))
        np.testing.assert_equal(sent[-1]["kd"], np.full(6, parameters["kd"]))

    def test_failed_sdk_mode_verification_never_enables_motors(self):
        cls, _ = load_reference_class()
        for arm_ok, gripper_ok in [(False, True), (True, False)]:
            with self.subTest(arm_ok=arm_ok, gripper_ok=gripper_ok):
                arm = Mock()
                arm.arm.num_joints = 6
                arm.arm.mode_mit.return_value = arm_ok
                arm.gripper.mode_mit.return_value = gripper_ok
                ctrl = cls.__new__(cls)
                ctrl._arm = arm
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "verification failed"):
                        ctrl.start()
                arm.enable_all.assert_not_called()
                arm.start_control_loop.assert_not_called()

    def test_manual_cli_defaults_to_original_reference_backend(self):
        with patch.object(sys, "argv", ["eyehand", "collect", "--manual", "--output", "unused"]), \
                patch("eyehand.cli.collect") as collect:
            main()
        self.assertEqual(collect.call_args.args[0].manual_backend, "reference")


if __name__ == "__main__":
    unittest.main()

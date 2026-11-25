from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import pinocchio as pin

from eyehand.gravity_parameters import apply_gravity_parameters, load_gravity_parameters, validate_gravity_parameters
from eyehand.hardware import DEFAULT_SDK


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.model = pin.buildModelFromUrdf(str(
            DEFAULT_SDK / "urdf/RS/urdf/ReBot_Arm_RS.urdf"))
        self.controller = SimpleNamespace(
            _model=self.model, _data=self.model.createData(),
            _compute_gravity=lambda model, q: pin.computeGeneralizedGravity(model, model.createData(), q))
        self.parameters = {**load_gravity_parameters(), "payload_mass_kg": .016,
                           "payload_frame": "gripper_end", "payload_com_m": [0., 0., 0.],
                           "wrist_camera_mass_kg": 0.}

    def test_payload_torque_matches_potential_gradient_at_varied_poses_and_offsets(self):
        # Independent physical check: compensation = derivative of gravitational
        # potential. Tests the sign and fixed-frame/COM transformations, not just
        # the same Pinocchio inertia addition used by the implementation.
        rng = np.random.default_rng(20)
        frame_id = self.model.getFrameId("gripper_end")
        for com in ([0., 0., 0.], [.03, -.02, .015]):
            parameters = {**self.parameters, "payload_com_m": com, "tau_scale": [1.03, 1.02, .95, 1., 1., 1.]}
            apply_gravity_parameters(self.controller, parameters)
            for _ in range(5):
                q = pin.neutral(self.model)
                q[:6] = rng.uniform(-.8, .8, 6)

                def potential(configuration):
                    data = self.model.createData()
                    pin.forwardKinematics(self.model, data, configuration)
                    pin.updateFramePlacements(self.model, data)
                    position = data.oMf[frame_id].act(np.asarray(com))
                    return -.016 * self.model.gravity.linear.dot(position)

                expected = []
                for j in range(self.model.nv):
                    step = np.zeros(self.model.nv)
                    step[j] = 1e-6
                    expected.append((potential(pin.integrate(self.model, q, step)) -
                                     potential(pin.integrate(self.model, q, -step))) / 2e-6)
                base = pin.computeGeneralizedGravity(self.model, self.model.createData(), q).copy()
                expected = base + np.asarray(expected)
                expected[:6] *= parameters["tau_scale"]
                actual = self.controller._compute_gravity(model=self.controller._model, q=q)
                np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_reapply_does_not_stack_mass_and_zero_removes_it_without_changing_fk(self):
        original_mass = sum(i.mass for i in self.model.inertias)
        apply_gravity_parameters(self.controller, self.parameters)
        apply_gravity_parameters(self.controller, self.parameters)
        self.assertAlmostEqual(sum(i.mass for i in self.controller._model.inertias), original_mass + .016)
        self.assertAlmostEqual(sum(i.mass for i in self.model.inertias), original_mass)
        q = pin.neutral(self.model)
        q[:6] = [.1, .4, .7, -.3, .2, .1]
        before, after = self.model.createData(), self.controller._model.createData()
        pin.framesForwardKinematics(self.model, before, q)
        pin.framesForwardKinematics(self.controller._model, after, q)
        for a, b in zip(before.oMf, after.oMf):
            np.testing.assert_allclose(a.homogeneous, b.homogeneous)
        apply_gravity_parameters(self.controller, {**self.parameters, "payload_mass_kg": 0.})
        self.assertAlmostEqual(sum(i.mass for i in self.controller._model.inertias), original_mass)

    def test_unknown_frame_fails_before_replacing_controller_model(self):
        with self.assertRaisesRegex(ValueError, "Unknown payload frame"):
            apply_gravity_parameters(self.controller, {**self.parameters, "payload_frame": "missing"})
        self.assertIs(self.controller._model, self.model)

    def test_rejects_invalid_payload_settings(self):
        for update in ({"payload_mass_kg": -.016}, {"payload_mass_kg": float("nan")},
                       {"payload_com_m": [0, 0]}, {"payload_com_m": [0, 0, float("inf")]},
                       {"payload_frame": ""}):
            with self.assertRaises(ValueError):
                validate_gravity_parameters({**self.parameters, **update})

    def test_camera_and_aruco_torques_add_at_their_separate_locations(self):
        parameters = {**self.parameters, "wrist_camera_mass_kg": .160,
                      "wrist_camera_frame": "link6", "wrist_camera_com_m": [0., 0., 0.],
                      "tau_scale": [1.] * 6}
        apply_gravity_parameters(self.controller, parameters)
        apply_gravity_parameters(self.controller, parameters)
        original_mass = sum(i.mass for i in self.model.inertias)
        self.assertAlmostEqual(sum(i.mass for i in self.controller._model.inertias), original_mass + .176)
        rng = np.random.default_rng(21)
        for _ in range(10):
            q = pin.neutral(self.model)
            q[:6] = rng.uniform(-1., 1., 6)
            data = self.model.createData()
            pin.computeJointJacobians(self.model, data, q)
            pin.updateFramePlacements(self.model, data)
            expected = pin.computeGeneralizedGravity(self.model, self.model.createData(), q).copy()
            for prefix in ("payload", "wrist_camera"):
                frame_id = self.model.getFrameId(parameters[f"{prefix}_frame"])
                # At each frame origin, gravity compensation is -J_linear.T * mg.
                jacobian = pin.getFrameJacobian(self.model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)
                expected -= jacobian[:3].T @ (parameters[f"{prefix}_mass_kg"] * self.model.gravity.linear)
            np.testing.assert_allclose(self.controller._compute_gravity(model=self.controller._model, q=q), expected, atol=1e-12)
        # Turn off only the camera; ArUco is retained.
        apply_gravity_parameters(self.controller, self.parameters)
        self.assertAlmostEqual(sum(i.mass for i in self.controller._model.inertias), original_mass + .016)

    def test_existing_camera_model_is_rejected_to_avoid_duplicate_camera_mass(self):
        self.model.addFrame(pin.Frame("camera_mount_link", self.model.getJointId("joint6"),
                                     pin.SE3.Identity(), pin.BODY))
        with self.assertRaisesRegex(ValueError, "already has wrist-camera"):
            apply_gravity_parameters(self.controller, {**self.parameters, "wrist_camera_mass_kg": .160})
        self.assertIs(self.controller._model, self.model)

    def test_rejects_invalid_camera_mass_and_com(self):
        for update in ({"wrist_camera_mass_kg": -.160}, {"wrist_camera_mass_kg": float("nan")},
                       {"wrist_camera_com_m": [0, 0]}, {"wrist_camera_com_m": [0, 0, float("inf")]}):
            with self.assertRaises(ValueError):
                validate_gravity_parameters({**self.parameters, **update})


if __name__ == "__main__":
    unittest.main()

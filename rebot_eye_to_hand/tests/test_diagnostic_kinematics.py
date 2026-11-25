from __future__ import annotations

import unittest
import hashlib

import cv2
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from eyehand.geometry import inverse, pack, transform
from eyehand.hardware import DEFAULT_SDK
from scripts.diagnose_handeye import DiagnosticFit


class KinematicsDiagnosticTests(unittest.TestCase):
    def make_data(self, seed, count):
        urdf = DEFAULT_SDK / "urdf/RS/urdf/ReBot_Arm_RS.urdf"
        model = pin.buildModelFromUrdf(str(urdf))
        data = model.createData()
        fid = model.getFrameId("gripper_end")
        indices = [model.joints[model.getJointId(f"joint{i}")].idx_q for i in range(1, 7)]
        x = transform(translation=[0., 0., -1.])
        y = transform(Rotation.from_euler("xyz", [.1, -.2, .3]).as_matrix(), [.02, .01, .05])
        offset = np.radians([0., -.4, .6, -1.3, -.8, 0.])
        k = np.array([[910., 0., 640.], [0., 909., 360.], [0., 0., 1.]])
        obj = np.array([[-.03, .03, 0.], [.03, .03, 0.], [.03, -.03, 0.], [-.03, -.03, 0.]])
        rng = np.random.default_rng(seed)
        result = {"schema_version": 1, "dataset_id": f"synthetic_offsets_{seed}", "synthetic": True,
                  "image_size": [1280, 720], "target": {"type": "synthetic_60mm"},
                  "camera": {"kind": "synthetic"}, "K": k.tolist(), "D": [0.] * 5,
                  "robot": {"urdf": str(urdf), "ee_frame": "gripper_end", "base_frame": "base_link",
                            "joint_names": [f"joint{i}" for i in range(1, 7)],
                            "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest()}, "samples": []}
        for index in range(count):
            q = pin.neutral(model)
            q[indices] = rng.uniform(-.8, .8, 6)
            pin.framesForwardKinematics(model, data, q)
            nominal = data.oMf[fid].homogeneous.copy()
            physical_q = q.copy()
            physical_q[indices] += offset
            pin.framesForwardKinematics(model, data, physical_q)
            c = inverse(x) @ data.oMf[fid].homogeneous @ y
            uv, _ = cv2.projectPoints(obj, cv2.Rodrigues(c[:3, :3])[0], c[:3, 3], k, np.zeros(5))
            result["samples"].append({"id": f"offset_pose_{seed}_{index}", "K": k.tolist(), "D": [0.] * 5,
                                      "T_base_ee": nominal.tolist(), "T_camera_target": c.tolist(),
                                      "object_points": obj.tolist(), "image_points": uv.reshape(-1, 2).tolist(),
                                      "robot_reads": [{"q_rad": q[indices].tolist()}]})
        return result, x, y, offset

    def test_recovers_planted_joint_offsets_and_predicts_unused_poses(self):
        train, x, y, offset = self.make_data(11, 30)
        test, _, _, _ = self.make_data(35, 10)
        fit = DiagnosticFit(train)
        opt = fit.fit("joint_offsets_2_5", fit.prepare(train), np.r_[pack(x), pack(y)])
        self.assertTrue(opt.success)
        metrics, parameters = fit.metrics(opt.x, "joint_offsets_2_5", fit.prepare(test))
        np.testing.assert_allclose(parameters["joint_offsets_deg"], np.degrees(offset), atol=1e-4)
        self.assertLess(metrics["reprojection_px"]["max"], 1e-5)
        self.assertLess(metrics["closure_mm"]["max"], 1e-4)

    def test_fixed_model_does_not_silently_apply_offsets(self):
        train, x, y, _ = self.make_data(11, 30)
        fit = DiagnosticFit(train)
        metrics, parameters = fit.metrics(np.r_[pack(x), pack(y)], "fixed", fit.prepare(train))
        self.assertEqual(parameters["joint_offsets_deg"], [0.] * 6)
        self.assertGreater(metrics["reprojection_px"]["rms"], .1)


if __name__ == "__main__":
    unittest.main()

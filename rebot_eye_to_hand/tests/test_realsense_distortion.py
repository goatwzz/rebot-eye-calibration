from __future__ import annotations

import unittest

import cv2
import numpy as np
import pyrealsense2 as rs

from eyehand.vision import realsense_opencv_distortion


class RealSenseDistortionTests(unittest.TestCase):
    def test_zero_inverse_and_modified_brown_are_pinhole(self):
        for model in [rs.distortion.inverse_brown_conrady, rs.distortion.modified_brown_conrady]:
            d, interpretation = realsense_opencv_distortion(model, [0.] * 5)
            np.testing.assert_equal(d, np.zeros(5))
            self.assertEqual(interpretation, "pinhole_exact_zero_brown_coefficients")

    def test_nonzero_inverse_or_modified_coefficients_remain_rejected(self):
        for model in [rs.distortion.inverse_brown_conrady, rs.distortion.modified_brown_conrady]:
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                realsense_opencv_distortion(model, [0.001, 0., 0., 0., 0.])

    def test_standard_brown_preserved_and_invalid_coefficients_rejected(self):
        values = [0.01, -0.02, 0.001, -0.001, 0.002]
        d, _ = realsense_opencv_distortion(rs.distortion.brown_conrady, values)
        np.testing.assert_equal(d, values)
        for values in [[0.] * 4, [float("nan")] + [0.] * 4]:
            with self.assertRaises(ValueError):
                realsense_opencv_distortion(rs.distortion.inverse_brown_conrady, values)

    def test_zero_inverse_matches_sdk_deprojection_and_opencv_projection(self):
        intr = rs.intrinsics()
        intr.width, intr.height = 1280, 720
        intr.fx, intr.fy = 906.2987670898438, 905.4373779296875
        intr.ppx, intr.ppy = 639.9422607421875, 376.76641845703125
        intr.model = rs.distortion.inverse_brown_conrady
        intr.coeffs = [0.] * 5
        k = np.array([[intr.fx, 0., intr.ppx], [0., intr.fy, intr.ppy], [0., 0., 1.]])
        d, _ = realsense_opencv_distortion(intr.model, intr.coeffs)
        for pixel in [[0., 0.], [1279., 719.], [640., 360.], [200., 600.]]:
            point = rs.rs2_deproject_pixel_to_point(intr, pixel, 1.)
            projected, _ = cv2.projectPoints(np.array([point]), np.zeros(3), np.zeros(3), k, d)
            np.testing.assert_allclose(projected.reshape(2), pixel, atol=1e-4)


if __name__ == "__main__":
    unittest.main()

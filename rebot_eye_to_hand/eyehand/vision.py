from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .geometry import transform


def select_realsense_serial(devices, requested=None):
    if requested:
        if requested not in {d["serial"] for d in devices}:
            raise ValueError(f"Requested RealSense {requested} not connected; available={devices}")
        return requested
    if len(devices) != 1:
        raise ValueError(f"Expected one RealSense or explicit --serial; available={devices}")
    return devices[0]["serial"]


def realsense_opencv_distortion(model, coefficients):
    """Return OpenCV-compatible coefficients without approximating inverse models.

    D435 RGB can report inverse_brown_conrady with exactly zero coefficients.
    Both forward and inverse Brown mappings then reduce to the pinhole model.
    Nonzero inverse/modified coefficients must not be treated as standard Brown.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (5,) or not np.isfinite(coefficients).all():
        raise ValueError("Invalid RealSense distortion coefficients")
    name = str(model).split(".")[-1]
    if name == "none":
        return np.zeros(5), "pinhole"
    if name == "brown_conrady":
        return coefficients.copy(), "opencv_brown_conrady"
    if name in ("inverse_brown_conrady", "modified_brown_conrady") and np.all(coefficients == 0):
        return np.zeros(5), "pinhole_exact_zero_brown_coefficients"
    raise ValueError(f"Unsupported OpenCV distortion model {model}, coefficients={coefficients.tolist()}; "
                     "first run intrinsics and supply --intrinsics")


class Target:
    def __init__(self, config):
        self.config = config
        aruco = cv2.aruco
        self.dictionary = aruco.getPredefinedDictionary(getattr(aruco, config["dictionary"]))
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.detector = aruco.ArucoDetector(self.dictionary, params)
        self.board = None
        if config["type"] == "charuco":
            self.board = aruco.CharucoBoard((config["squares_x"], config["squares_y"]),
                                            config["square_m"], config["marker_m"], self.dictionary)
            self.charuco = aruco.CharucoDetector(self.board)
        elif config["type"] != "aruco":
            raise ValueError("Target type must be aruco or charuco")

    def detect(self, frame, k, d, max_rms=1.0, ambiguity_gap=0.15):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.board is None:
            corners, ids, _ = self.detector.detectMarkers(gray)
            target_id = self.config["id"]
            if ids is None or np.count_nonzero(ids == target_id) != 1:
                raise ValueError(f"Need exactly one marker with ID {target_id}")
            uv = corners[int(np.flatnonzero(ids.flatten() == target_id)[0])].reshape(-1, 2)
            h = self.config["marker_m"] / 2
            obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=float)
            flag = cv2.SOLVEPNP_IPPE_SQUARE
        else:
            params = cv2.aruco.CharucoParameters()
            params.cameraMatrix, params.distCoeffs = k, d
            self.charuco.setCharucoParameters(params)
            corners, ids, _, _ = self.charuco.detectBoard(gray)
            if ids is None or len(ids) < 8:
                raise ValueError("Need at least 8 ChArUco corners")
            obj, uv = self.board.matchImagePoints(corners, ids)
            obj, uv = obj.reshape(-1, 3).astype(float), uv.reshape(-1, 2).astype(float)
            if np.linalg.matrix_rank(obj[:, :2] - obj[:, :2].mean(axis=0)) < 2:
                raise ValueError("Collinear ChArUco corners")
            flag = cv2.SOLVEPNP_IPPE
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, uv, k, d, flags=flag)
        if not ok:
            raise ValueError("PnP failed")
        candidates = []
        for rv, tv in zip(rvecs, tvecs):
            rot = cv2.Rodrigues(rv)[0]
            if not np.isfinite(rot).all() or not np.isfinite(tv).all():
                continue
            if np.any((obj @ rot.T + tv.reshape(3))[:, 2] <= 0):
                continue
            projected, _ = cv2.projectPoints(obj, rv, tv, k, d)
            error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - uv) ** 2, axis=1))))
            candidates.append((error, rv, tv))
        candidates.sort(key=lambda v: v[0])
        if not candidates:
            raise ValueError("No positive-depth PnP solution")
        gap = candidates[1][0] - candidates[0][0] if len(candidates) > 1 else None
        if gap is not None and gap < ambiguity_gap:
            raise ValueError(f"Planar pose ambiguous (gap={gap:.3f}px); tilt target or use a larger board")
        rms, rv, tv = candidates[0]
        if rms > max_rms:
            raise ValueError(f"PnP RMS {rms:.2f}px exceeds {max_rms:.2f}px")
        result = {"T_camera_target": transform(cv2.Rodrigues(rv)[0], tv).tolist(),
                  "object_points": obj.tolist(), "image_points": uv.tolist(),
                  "pnp_rms_px": rms, "ippe_second_solution_gap_px": gap,
                  "K": k.tolist(), "D": d.tolist()}
        return result

    def write(self, prefix):
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        if self.board is None:
            marker = cv2.aruco.generateImageMarker(self.dictionary, self.config["id"], 1000)
            image = cv2.copyMakeBorder(marker, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
            w = h = self.config["marker_m"] * 1200  # total mm = black edge * 1.2
        else:
            sx, sy = self.config["squares_x"], self.config["squares_y"]
            image = self.board.generateImage((sx * 200, sy * 200), marginSize=0, borderBits=1)
            image = cv2.copyMakeBorder(image, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
            w, h = (sx + 0.4) * self.config["square_m"] * 1000, (sy + 0.4) * self.config["square_m"] * 1000
        if not cv2.imwrite(str(prefix.with_suffix(".png")), image):
            raise OSError("Could not write target image")
        _, encoded = cv2.imencode(".png", image)
        data = base64.b64encode(encoded).decode("ascii")
        prefix.with_suffix(".svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm" viewBox="0 0 {image.shape[1]} {image.shape[0]}">'
            f'<image width="{image.shape[1]}" height="{image.shape[0]}" href="data:image/png;base64,{data}"/></svg>')


class Camera:
    def __init__(self, kind, width=1280, height=720, fps=30, serial=None, device=0, intrinsics=None,
                 allow_uncalibrated=False):
        self.kind, self.pipeline, self.cap = kind, None, None
        self.metadata = {"kind": kind}
        override = json.loads(Path(intrinsics).read_text()) if intrinsics else None
        if kind == "realsense":
            import pyrealsense2 as rs
            self.rs = rs
            devices = [{"name": d.get_info(rs.camera_info.name), "serial": d.get_info(rs.camera_info.serial_number)}
                       for d in rs.context().query_devices()]
            serial = select_realsense_serial(devices, serial)
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            try:
                profile = self.pipeline.start(cfg)
                intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                actual_serial = profile.get_device().get_info(rs.camera_info.serial_number)
                if actual_serial != serial:
                    raise ValueError(f"Camera serial mismatch: requested {serial}, actual {actual_serial}")
                self.metadata.update(serial=actual_serial, name=profile.get_device().get_info(rs.camera_info.name),
                                     distortion_model=str(intr.model), sdk_distortion_coeffs=list(intr.coeffs))
                self.k = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.]])
                self.d = np.asarray(intr.coeffs, dtype=float)
                self.width, self.height = intr.width, intr.height
                if override is None:
                    try:
                        self.d, interpretation = realsense_opencv_distortion(intr.model, intr.coeffs)
                        self.metadata["opencv_distortion_interpretation"] = interpretation
                    except ValueError:
                        if allow_uncalibrated:
                            self.k, self.d = None, None
                        else:
                            raise
            except Exception:
                self.close()
                raise
        elif kind == "opencv":
            self.cap = cv2.VideoCapture(device)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open camera")
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.k, self.d = None, None
            self.metadata["device"] = device
        else:
            raise ValueError("Unknown camera")
        if override:
            calibrated_serial = override.get("camera", {}).get("serial")
            if calibrated_serial and calibrated_serial != self.metadata.get("serial"):
                self.close()
                raise ValueError(f"Intrinsics belong to camera {calibrated_serial}, selected {self.metadata.get('serial')}")
            if override["image_size"] != [self.width, self.height]:
                self.close()
                raise ValueError("Intrinsics resolution does not match camera; no automatic scaling")
            self.k, self.d = np.asarray(override["K"], dtype=float), np.asarray(override["D"], dtype=float)
            self.metadata["opencv_distortion_interpretation"] = "user_calibrated_opencv_intrinsics"
        if self.k is not None and (self.k.shape != (3, 3) or not np.isfinite(self.k).all()
                                   or self.k[0, 0] <= 0 or self.k[1, 1] <= 0 or not np.isfinite(self.d).all()):
            self.close()
            raise ValueError("Invalid intrinsics")

    def read(self, fresh=False, timeout_ms=5000):
        if self.pipeline:
            if fresh:
                # Discard queued frames, then wait for a newly delivered one.
                for _ in range(32):
                    if not self.pipeline.poll_for_frames():
                        break
            frames = self.pipeline.wait_for_frames(timeout_ms)
            frame = frames.get_color_frame()
            self.last_frame_metadata = {"device_timestamp_ms": frame.get_timestamp(),
                                        "timestamp_domain": str(frame.get_frame_timestamp_domain()),
                                        "frame_number": frame.get_frame_number()}
            image = np.asanyarray(frame.get_data()).copy()
        else:
            if fresh:
                # UVC backends may ignore BUFFERSIZE; drain several frames.
                for _ in range(5):
                    if not self.cap.grab():
                        raise RuntimeError("Camera grab failed")
            ok, image = self.cap.read()
            if not ok:
                raise RuntimeError("Camera frame read failed")
            self.last_frame_metadata = {}
        self.last_frame_metadata["host_received_unix_s"] = time.time()
        if image.shape[1::-1] != (self.width, self.height):
            raise ValueError("Camera resolution changed")
        return image

    def close(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
            except RuntimeError:
                # start() may have failed, or the USB device may be gone.
                # Preserve the acquisition error instead of masking it here.
                pass
            finally:
                self.pipeline = None
        if self.cap:
            self.cap.release()
            self.cap = None

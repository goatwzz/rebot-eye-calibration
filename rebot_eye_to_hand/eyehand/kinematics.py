"""Offline identification of identifiable joint-angle corrections for hand-eye use.

These effective kinematic corrections do not change encoder/motor zeros, dynamics,
or a running controller. q_model = q_measured + offsets. Joints 1/6 are gauges
absorbed by the base-camera and end-target transforms and are fixed to zero.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as et

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import checked, diversity, errors, evaluate, inverse, pack, solve, unpack

from .paths import artifact_path

JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]


class CorrectedKinematics:
    def __init__(self, robot, correction=None):
        import pinocchio as pin

        self.pin = pin
        self.path = artifact_path(Path(robot["urdf"]).expanduser()).resolve()
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if digest != robot["urdf_sha256"]:
            raise ValueError("URDF has changed since collection; cannot apply kinematic identification")
        if robot.get("joint_names") != JOINT_NAMES:
            raise ValueError("Kinematic identification requires joint1..joint6 in order")
        self.model = pin.buildModelFromUrdf(str(self.path))
        self.frame_id = self.model.getFrameId(robot["ee_frame"])
        if self.frame_id >= self.model.nframes:
            raise ValueError("Unknown end-effector frame")
        self.data = self.model.createData()
        self.indices = []
        for name in JOINT_NAMES:
            jid = self.model.getJointId(name)
            if jid >= self.model.njoints or self.model.joints[jid].nq != 1:
                raise ValueError("Expected six scalar arm joints")
            self.indices.append(self.model.joints[jid].idx_q)
        self.offsets = np.zeros(6)
        if correction is not None:
            if correction.get("type") != "effective_joint_angle_offsets" or correction.get("joint_names") != JOINT_NAMES:
                raise ValueError("Unsupported kinematic correction")
            if correction.get("original_urdf_sha256") != digest:
                raise ValueError("Correction belongs to a different URDF")
            for name in ("base_frame", "ee_frame"):
                if correction.get(name) != robot[name]:
                    raise ValueError(f"Kinematic correction {name} mismatch")
            self.offsets = np.asarray(correction["joint_offsets_rad"], dtype=float)
            if (self.offsets.shape != (6,) or not np.isfinite(self.offsets).all()
                    or np.max(np.abs(self.offsets)) > np.radians(5.)
                    or np.any(self.offsets[[0, 5]] != 0.)):
                raise ValueError("Invalid kinematic offsets or gauge joints")

    def fk(self, raw, offsets=None):
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (6,) or not np.isfinite(raw).all():
            raise ValueError("Need six finite measured joint positions")
        q = self.pin.neutral(self.model)
        q[self.indices] = raw + (self.offsets if offsets is None else offsets)
        self.pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.frame_id].homogeneous.copy()


def measured_joints(sample):
    try:
        raw = np.asarray(sample["robot_reads"][-1]["q_rad"], dtype=float)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Kinematic correction requires saved raw joint feedback for each sample") from exc
    if raw.shape != (6,) or not np.isfinite(raw).all():
        raise ValueError("Need six finite measured joint positions")
    return raw


def corrected_dataset(data, correction):
    """Recompute FK from RAW feedback once; never add offsets to stored FK."""
    existing = data["robot"].get("kinematic_correction")
    if existing is not None and existing != correction:
        raise ValueError("Dataset already uses a different kinematic correction")
    fk = CorrectedKinematics(data["robot"], correction)
    samples = [{**s, "T_base_ee": fk.fk(measured_joints(s)).tolist()} for s in data["samples"]]
    return {**data, "robot": {**data["robot"], "kinematic_correction": correction}, "samples": samples}


def identify_kinematics(data, initial=None):
    if len(data["samples"]) < 20:
        raise ValueError("Need at least 20 diverse poses for four joint corrections; recommend >=40")
    if len({s["id"] for s in data["samples"]}) != len(data["samples"]):
        raise ValueError("Duplicate sample IDs in identification data")
    if data["robot"].get("kinematic_correction"):
        raise ValueError("Identify from the original nominal dataset, not an already corrected dataset")
    fk = CorrectedKinematics(data["robot"])
    poses = [checked(s["T_base_ee"]) for s in data["samples"]]
    diversity(poses)
    observations = []
    for s,g in zip(data["samples"], poses):
        raw = measured_joints(s)
        if max(errors(fk.fk(raw), g)) > 1e-5:
            raise ValueError("Saved nominal FK does not match raw feedback and the recorded URDF")
        obj, uv = np.asarray(s["object_points"], dtype=float), np.asarray(s["image_points"], dtype=float)
        k, d = np.asarray(s["K"], dtype=float), np.asarray(s["D"], dtype=float)
        if (obj.ndim != 2 or obj.shape[1] != 3 or len(obj) < 4 or uv.shape != (len(obj), 2)
                or k.shape != (3, 3) or d.size not in (4, 5, 8, 12, 14)
                or not all(np.isfinite(a).all() for a in (obj, uv, k, d))):
            raise ValueError("Invalid image observations for kinematic identification")
        observations.append((raw, obj, uv, k, d))
    if initial is None:
        initial = solve(data["samples"], refinement="reprojection")
    elif (initial.get("training_dataset_id") != data["dataset_id"] or initial.get("kinematic_correction")):
        raise ValueError("Initial result must come from this original dataset without kinematic correction")
    x0, y0 = checked(initial["T_base_camera"]), checked(initial["T_ee_target"])

    def residual(z):
        xi, y = inverse(unpack(z[:6])), unpack(z[6:12])
        offsets = np.r_[0., z[12:16], 0.]
        out = []
        for raw, obj, uv, k, d in observations:
            c = xi @ fk.fk(raw, offsets) @ y
            projected, _ = cv2.projectPoints(obj, cv2.Rodrigues(c[:3, :3])[0], c[:3, 3], k, d)
            out.extend((projected.reshape(-1, 2) - uv).ravel())
            depth = (obj @ c[:3, :3].T + c[:3, 3])[:, 2]
            out.extend(np.maximum(0., .001 - depth) * 1e6)
        return np.asarray(out)

    seed = np.r_[pack(x0), pack(y0), np.zeros(4)]
    lower, upper = np.full(16, -np.inf), np.full(16, np.inf)
    lower[12:], upper[12:] = -np.radians(5.), np.radians(5.)
    opt = least_squares(residual, seed, bounds=(lower, upper), loss="soft_l1", f_scale=1.,
                        max_nfev=500, xtol=1e-11, ftol=1e-11, gtol=1e-11)
    if not opt.success or np.any(np.abs(opt.x[12:]) >= np.radians(4.99)):
        raise ValueError(f"Identification did not converge inside the angle bounds: {opt.message}")
    norms = np.linalg.norm(opt.jac, axis=0)
    sv = np.linalg.svd(opt.jac / np.maximum(norms, 1e-15), compute_uv=False)
    rank = int(np.count_nonzero(sv > sv[0] * 1e-6))
    if rank < 16:
        raise ValueError("Identification is not observable for these poses; add independent rotations")
    offsets = np.r_[0., opt.x[12:16], 0.]
    robot = data["robot"]
    correction = {"schema_version": 1, "type": "effective_joint_angle_offsets",
                  "convention": "q_model_rad = q_measured_rad + joint_offsets_rad",
                  "joint_names": JOINT_NAMES, "joint_offsets_rad": offsets.tolist(),
                  "joint_offsets_deg": np.degrees(offsets).tolist(),
                  "gauge_fixed_joints": ["joint1", "joint6"],
                  "original_urdf": str(fk.path), "original_urdf_sha256": robot["urdf_sha256"],
                  "base_frame": robot["base_frame"], "ee_frame": robot["ee_frame"],
                  "training_dataset_id": data["dataset_id"], "rigid_target_assumed": True,
                  "physical_accuracy_verified": False,
                  "scope": "Effective FK angle correction only; link lengths, axes and compliance not identified. No motor zero or dynamics changes."}
    corrected = corrected_dataset(data, correction)
    x, y = unpack(opt.x[:6]), unpack(opt.x[6:12])
    training = evaluate(corrected["samples"], x, y)
    result = {"schema_version": 1, "convention": "T_A_B maps coordinates from B to A; translation in meters",
              "T_base_camera": x.tolist(), "T_camera_base": inverse(x).tolist(), "T_ee_target": y.tolist(),
              "kinematic_correction": correction, "initial_method": "nominal_reprojection_then_joint_offset_identification",
              "optimization": {"enabled": True, "success": True, "objective": "image_corners_with_joint_offsets_2_5",
                               "loss": "soft_l1", "f_scale_px": 1., "cost": float(opt.cost), "nfev": int(opt.nfev),
                               "normalized_jacobian_rank": rank, "normalized_jacobian_condition": float(sv[0]/sv[-1]),
                               "samples_removed": 0},
              "training": training, "training_ids": [s["id"] for s in data["samples"]],
              "training_dataset_id": data["dataset_id"],
              "provenance": {k: data[k] for k in ["target", "image_size", "camera", "robot", "K", "D", "synthetic"]},
              "warnings": ["Use this result with its paired kinematic correction. This is not full URDF geometric calibration or a motor-zero calibration."],
              "acceptance": "UNVALIDATED: new validation and independent physical accuracy check required"}
    return correction, result


def export_corrected_urdf(correction, destination):
    """Export a local URDF whose FK(raw q) equals original FK(raw q + delta)."""
    source = artifact_path(correction["original_urdf"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != correction["original_urdf_sha256"]:
        raise ValueError("Original URDF changed before export")
    tree = et.parse(source)
    root = tree.getroot()
    for name, delta in zip(correction["joint_names"], correction["joint_offsets_rad"]):
        if delta == 0.: continue
        joint = root.find(f"joint[@name='{name}']")
        if joint is None or joint.get("type") != "revolute":
            raise ValueError(f"Expected revolute joint: {name}")
        origin, axis_node = joint.find("origin"), joint.find("axis")
        if origin is None:
            origin = et.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
        axis = np.fromstring(axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0", sep=" ")
        axis /= np.linalg.norm(axis)
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        rotation = Rotation.from_euler("xyz", rpy) * Rotation.from_rotvec(axis * delta)
        origin.set("rpy", " ".join(f"{v:.17g}" for v in rotation.as_euler("xyz")))
    # Preserve relative mesh references when the copy lives in a new directory.
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if filename and "://" not in filename:
            mesh.set("filename", str((source.parent / filename).resolve()))
    root.insert(0, et.Comment(" Effective joint2..5 angle correction; raw encoder inputs; do not add offsets twice. Not a motor-zero or full geometric calibration. "))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)

#!/usr/bin/env python3
"""Validate an existing wrist hand-eye result, without refitting it.

Camera, marker detection, stationary acquisition and optional hand guiding are
reused from the sibling rebot_eye_to_hand project. No new motor controller here. Runtime defaults live in config/default.yaml.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

from .backend import dependencies

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    # Exclusive creation: never overwrite an earlier measurement/report.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_result(path, geometry, *, allow_joint_offsets=False):
    with np.load(path, allow_pickle=False) as data:
        if str(data["mode"][0]) != "eye_in_hand":
            raise ValueError("Result must be eye_in_hand, not eye-to-hand")
        transform = geometry.checked(data["T_result"])
        info = {"path": str(Path(path).resolve()), "sha256": sha256(path),
                "n_samples": int(data["n_samples"][0]),
                "method": str(data["method"][0]) if "method" in data else "unknown",
                "T_gripper_camera": transform.tolist()}
        for key in ("camera_serial", "urdf_sha256"):
            if key in data:
                info[key] = str(data[key][0])
        for key in ("training_dataset_ids", "training_dataset_sha256"):
            if key in data:
                info[key] = [str(v) for v in data[key].reshape(-1)]
        offsets = np.asarray(data["joint_offsets_rad"], float) if "joint_offsets_rad" in data else np.zeros(6)
        if (offsets.shape != (6,) or not np.isfinite(offsets).all()
                or np.any(np.abs(offsets) > np.radians(5.)) or np.any(offsets[[0,5]] != 0)):
            raise ValueError("Invalid effective joint offsets or gauge joints")
        if np.any(offsets != 0) and not allow_joint_offsets:
            raise ValueError("This validator uses raw joint angles; a candidate with nonzero joint offsets is incompatible")
        info['joint_offsets_rad'] = offsets.tolist()
    return transform, info


def apply_result_correction(dataset, info):
    """Rebuild saved nominal FK from measured RAW feedback, exactly once."""
    offsets=np.asarray(info.get('joint_offsets_rad',[0.]*6),float)
    if not np.any(offsets):return dataset
    from eyehand.kinematics import corrected_dataset,JOINT_NAMES
    correction=dict(type='effective_joint_angle_offsets',joint_names=JOINT_NAMES,
        joint_offsets_rad=offsets.tolist(),original_urdf_sha256=info['urdf_sha256'],
        base_frame='base_link',ee_frame='gripper_end')
    return corrected_dataset(dataset,correction)


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"rms": float(np.sqrt(np.mean(values ** 2))),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def evaluate(dataset, hand_eye, compensation, geometry, min_samples=15,
             translation_p95_mm=3., rotation_p95_deg=1., reprojection_p95_px=3.):
    """Estimate only the unknown fixed-target pose. The hand-eye stays fixed."""
    samples = dataset["samples"]
    if not samples:
        raise ValueError("No saved samples to evaluate")
    ids = [s["id"] for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs")
    hand_eye = geometry.checked(hand_eye)
    compensation = geometry.checked(compensation)
    poses = [geometry.checked(s["T_base_ee"]) for s in samples]
    cameras = [geometry.checked(s["T_camera_target"]) for s in samples]
    target_poses = [g @ hand_eye @ c for g, c in zip(poses, cameras)]
    mean = geometry.average(target_poses)
    shifted_mean = compensation @ mean
    rows = []
    for s, g, y in zip(samples, poses, target_poses):
        mm, deg = geometry.errors(y, mean)
        predicted = geometry.inverse(g @ hand_eye) @ mean
        obj, image = np.asarray(s["object_points"], float), np.asarray(s["image_points"], float)
        k, d = np.asarray(s["K"], float), np.asarray(s["D"], float)
        if (obj.ndim != 2 or obj.shape[1] != 3 or len(obj) < 4
                or image.shape != (len(obj), 2) or k.shape != (3, 3)
                or not all(np.isfinite(a).all() for a in (obj, image, k, d))):
            raise ValueError(f"Invalid saved image observations: {s['id']}")
        if np.any((obj @ predicted[:3, :3].T + predicted[:3, 3])[:, 2] <= 0):
            raise ValueError(f"Predicted target behind camera: {s['id']}")
        uv, _ = cv2.projectPoints(obj, cv2.Rodrigues(predicted[:3, :3])[0], predicted[:3, 3], k, d)
        reproj = float(np.sqrt(np.mean(np.sum((uv.reshape(-1, 2) - image) ** 2, axis=1))))
        shifted_mm, shifted_deg = geometry.errors(compensation @ y, shifted_mean)
        rows.append({"id": s["id"], "image": s["image"], "translation_mm": mm,
                     "rotation_deg": deg, "predicted_reprojection_rms_px": reproj,
                     "with_compensation_translation_mm": shifted_mm,
                     "with_compensation_rotation_deg": shifted_deg,
                     "T_base_target": y.tolist()})
    fields = ("translation_mm", "rotation_deg", "predicted_reprojection_rms_px")
    summary = {key: stats([r[key] for r in rows]) for key in fields}
    warnings = []
    try:
        motion = geometry.diversity(poses) if len(poses) >= 3 else None
    except ValueError as exc:
        motion = None
        warnings.append(str(exc))
    if len(samples) < min_samples:
        warnings.append(f"Need >= {min_samples} poses; got {len(samples)}")
    thresholds = {"translation_p95_mm": translation_p95_mm,
                  "rotation_p95_deg": rotation_p95_deg,
                  "predicted_reprojection_p95_px": reprojection_p95_px}
    passed = (summary["translation_mm"]["p95"] <= translation_p95_mm
              and summary["rotation_deg"]["p95"] <= rotation_p95_deg
              and summary["predicted_reprojection_rms_px"]["p95"] <= reprojection_p95_px)
    status = ("INSUFFICIENT_POSES" if warnings or motion is None else
              "PASS_CONSISTENCY_ONLY" if passed else "FAIL_CONSISTENCY")
    if dataset.get("synthetic", True):
        status = "SYNTHETIC_ONLY"
    return {"status": status, "count": len(samples), "summary": summary,
            "thresholds": thresholds, "motion_diversity": motion,
            "mean_T_base_target_without_compensation": mean.tolist(),
            "mean_T_base_target_with_compensation": shifted_mean.tolist(),
            "compensation": {"T_base_compensation": compensation.tolist(),
                             "note": "A constant base-frame shift changes the mean, not dispersion or reprojection. This test cannot establish whether -31 mm is correct."},
            "samples": rows, "warnings": warnings, "samples_removed": 0,
            "hand_eye_refitted": False, "physical_accuracy_verified": False,
            "interpretation": "Fixed-target consistency with a fixed hand-eye. Mean target pose is estimated from these observations, not independent ground truth. Physical bias and camera-vs-kinematics error cannot be isolated here."}


def report(args, geometry):
    dataset_path = Path(args.dataset).resolve()
    if dataset_path.is_dir():
        dataset_path /= "dataset.json"
    directory = dataset_path.parent
    session = read_json(directory / "validation_session.json")
    if session["purpose"] != "existing_wrist_eye_in_hand_validation":
        raise ValueError("Dataset was not collected as a fixed-target wrist validation")
    result_path = directory / "hand_eye_under_test.npz"
    if sha256(result_path) != session["result_sha256"]:
        raise ValueError("Calibration snapshot changed since collection")
    hand_eye, result_info = load_result(result_path, geometry, allow_joint_offsets=True)
    dataset = read_json(dataset_path)
    if (dataset["dataset_id"] in result_info.get("training_dataset_ids", [])
            or sha256(dataset_path) in result_info.get("training_dataset_sha256", [])):
        raise ValueError("This dataset was used to fit the candidate; collect a new validation dataset")
    if dataset["camera"].get("serial") != session["requested_camera_serial"]:
        raise ValueError("Dataset camera serial differs from the validation session")
    robot = dataset["robot"]
    if (robot.get("ee_frame") != "gripper_end" or robot.get("base_frame") != "base_link"
            or robot.get("urdf_sha256") != session["urdf_sha256"]):
        raise ValueError("Dataset robot model/frame differs from the validation session")
    if (result_info.get("camera_serial", dataset["camera"]["serial"]) != dataset["camera"]["serial"]
            or result_info.get("urdf_sha256", robot["urdf_sha256"]) != robot["urdf_sha256"]):
        raise ValueError("Candidate calibration belongs to a different camera or URDF")
    if dataset["target"] != read_json(directory / "target.json"):
        raise ValueError("Target configuration differs from session snapshot")
    compensation = np.eye(4)
    compensation[:3, 3] = session["runtime_compensation_base_m"]
    evaluated = apply_result_correction(dataset,result_info)
    result = evaluate(evaluated, hand_eye, compensation, geometry, args.min_samples,
                      args.translation_p95_mm, args.rotation_p95_deg, args.reprojection_p95_px)
    result.update(joint_offsets_rad=result_info['joint_offsets_rad'],
                  joint_angle_corrections_applied=bool(np.any(result_info['joint_offsets_rad'])),
                  schema_version=1, dataset=str(dataset_path), dataset_id=dataset["dataset_id"],
                  dataset_sha256=sha256(dataset_path), calibration=result_info,
                  camera=dataset["camera"], robot=robot, target=dataset["target"],
                  original_calibration_camera_identity_verified="camera_serial" in result_info,
                  original_training_sample_independence=(
                      "Dataset ID and file hash differ from recorded candidate training datasets; independent acquisition still requires fixed setup"
                      if result_info.get("training_dataset_ids") and result_info.get("training_dataset_sha256") else
                      "Not verifiable: original paired training samples unavailable"))
    if "camera_serial" not in result_info or "urdf_sha256" not in result_info:
        result["warnings"].append("Original NPZ contains no camera serial or model fingerprint; historical pairing is unverified")
    intrinsics_file = directory / "original_intrinsics.npz"
    if intrinsics_file.exists():
        with np.load(intrinsics_file, allow_pickle=False) as archived:
            old_k = archived["camera_matrix"]
        current_k = np.asarray(dataset["K"])
        result["intrinsics_comparison"] = {"archived_K": old_k.tolist(), "live_K": current_k.tolist(),
                                           "max_absolute_K_difference": float(np.max(np.abs(current_k - old_k)))}
        if not np.allclose(old_k, current_k, atol=.5, rtol=0):
            result["warnings"].append("Live intrinsics differ from the archived file; check camera identity/profile. Archived file may itself be stale.")
    output = Path(args.output) if args.output else directory / "validation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps({"output": str(output), **{key: result[key] for key in
                      ("status", "count", "summary", "motion_diversity", "physical_accuracy_verified", "warnings")}}, indent=2))
    return result


def acquisition_args(args, target_path, output, sdk):
    cfg = yaml.safe_load(Path(getattr(args, "config", ROOT / "config/default.yaml")).read_text())
    sampling, camera = cfg["acquisition"], cfg["camera"]
    return argparse.Namespace(command=args.command, camera="realsense", serial=args.serial,
        width=camera["color_width"], height=camera["color_height"], fps=args.fps, device=0, intrinsics=None,
        target=str(target_path), output=str(output), max_pnp_px=sampling["max_pnp_px"], ambiguity_gap_px=sampling["ambiguity_gap_px"],
        manual=getattr(args, "manual", False), manual_backend="reference",
        reference_script=str(ROOT / "robot/gravity_controller.py"),
        gravity_config=str(output / "gravity_parameters.json"),
        sdk=str(sdk), channel=getattr(args, "channel", None),
        ee_frame="gripper_end", base_frame="base_link", pose_file=None,
        settle_s=sampling["settle_s"], static_mm=sampling["static_mm"], static_deg=sampling["static_deg"],
        report_options=cfg["validation"])


def collect_and_report(cli, capture_args, directory, geometry):
    """The existing collector finishes hold/release cleanup before returning."""
    failure = None
    try:
        cli.collect(capture_args)
    except Exception as exc:
        failure = exc
    dataset_path = directory / "dataset.json"
    count = len(read_json(dataset_path)["samples"]) if dataset_path.exists() else 0
    write_json(directory / "collection_end.json", {
        "status": "ERROR" if failure else "STOPPED",
        "error_type": type(failure).__name__ if failure else None,
        "error": str(failure) if failure else None,
        "sample_count": count,
        "dataset_created": dataset_path.exists(),
        "note": "Recorded after collector cleanup. Saved samples are retained; no device automatically restarted."})
    if count:
        try:
            options = getattr(capture_args, "report_options", dict(min_samples=15,
                              translation_p95_mm=3., rotation_p95_deg=1., reprojection_p95_px=3.))
            report(argparse.Namespace(dataset=str(directory), output=None, **options), geometry)
        except Exception as report_error:
            if failure is None:
                raise
            print(f"Partial report failed: {report_error}; original samples remain in {dataset_path}", flush=True)
    if failure is not None:
        detail = (f"Saved data retained in {directory}" if count else
                  f"No samples were recorded; session metadata retained in {directory}")
        raise RuntimeError(f"{failure}. {detail}; see collection_end.json") from failure


def collect(args, cli, geometry):
    hand_eye, info = load_result(args.result, geometry, allow_joint_offsets=True)
    del hand_eye
    sdk = Path(args.sdk).resolve()
    hw = yaml.safe_load((sdk / "config/rebotarm_rs.yaml").read_text())
    selected = yaml.safe_load((sdk / "config/rebotarm.yaml").read_text())
    if Path(selected["hardware_yaml"]).name != "rebotarm_rs.yaml" or hw.get("end_effector_frame") != "gripper_end":
        raise ValueError("Validation requires RS SDK with gripper_end frame")
    urdf = sdk / hw["urdf_path"]
    if (info.get("camera_serial", args.serial) != args.serial
            or info.get("urdf_sha256", sha256(urdf)) != sha256(urdf)):
        raise ValueError("Candidate camera serial or URDF does not match this acquisition")
    cfg = yaml.safe_load(Path(getattr(args, "config", ROOT / "config/default.yaml")).read_text())
    comp = cfg.get("calibration", {}).get("hand_eye_compensation_m", {})
    comp_xyz = [float(comp.get(axis, 0)) for axis in "xyz"]
    if not np.isfinite(comp_xyz).all():
        raise ValueError("Runtime compensation must be finite")
    if args.channel != hw.get("channel"):
        raise ValueError(f"SDK channel is {hw.get('channel')}; --channel must match")
    # Pin imports to the same legacy SDK/model as the original wrist workflow.
    if "reBotArm_control_py" in sys.modules:
        raise ValueError("SDK already imported; launch this script as a fresh process")
    sys.path.insert(0, str(sdk))
    gravity_path = Path(args.gravity_config)
    from eyehand.gravity_parameters import load_gravity_parameters
    gravity = load_gravity_parameters(gravity_path)
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.result, directory / "hand_eye_under_test.npz")
    if sha256(directory / "hand_eye_under_test.npz") != info["sha256"]:
        raise ValueError("Calibration changed while creating the session; choose a new output directory")
    original_intrinsics = Path(args.result).with_name("intrinsics.npz")
    if original_intrinsics.exists():
        shutil.copyfile(original_intrinsics, directory / "original_intrinsics.npz")
    shutil.copyfile(gravity_path, directory / "gravity_parameters.json")
    write_json(directory / "target.json", target_config(args))
    write_json(directory / "validation_session.json", {
        "purpose": "existing_wrist_eye_in_hand_validation", "result_original_path": str(Path(args.result).resolve()),
        "result_sha256": info["sha256"], "requested_camera_serial": args.serial,
        "sdk_path": str(sdk), "urdf_path": str(urdf), "urdf_sha256": sha256(urdf),
        "runtime_compensation_base_m": comp_xyz,
        "joint_angle_corrections_applied": False,
        "evaluation_joint_offsets_rad": info['joint_offsets_rad'],
        "gravity_parameters": gravity, "mode": "reference-manual" if args.manual else "position-queries-only",
        "setup": "Camera rigidly attached to wrist; marker fixed in base/world, never attached to gripper"})
    print("WRIST VALIDATION: camera on wrist, target FIXED on table. Old calibration is never overwritten.", flush=True)
    if args.manual:
        print("Reusing original reference gravity controller: active arm AND gripper control. Stop other controllers; support arm at startup. q ends collection; support arm and type release to disable. No automatic homing.", flush=True)
    else:
        print("Read-only joint queries: existing controller must hold/reposition robot. This program supplies no holding torque.", flush=True)
    capture_args = acquisition_args(args, directory / "target.json", directory, sdk)
    # One-second waits keep window key processing responsive; ten attempts total.
    capture_args.camera_timeout_ms = cfg["acquisition"]["camera_timeout_ms"]
    capture_args.camera_timeout_retries = cfg["acquisition"]["camera_timeout_retries"]
    collect_and_report(cli, capture_args, directory, geometry)


def target_config(args):
    if not np.isfinite(args.marker_mm) or not 1 <= args.marker_mm <= 500:
        raise ValueError("--marker-mm must be measured black outer-edge length, between 1 and 500 mm")
    from eyehand.vision import Target
    config = {"type": "aruco", "dictionary": args.dictionary,
              "id": args.marker_id, "marker_m": args.marker_mm / 1000}
    # Validate the dictionary/ID before opening any device.
    detector = Target(config)
    if not 0 <= args.marker_id < len(detector.dictionary.bytesList):
        raise ValueError("Marker ID out of range for dictionary")
    return config


def main(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=str(ROOT / "config/default.yaml"))
    initial, _ = bootstrap.parse_known_args(argv)
    cfg = yaml.safe_load(Path(initial.config).read_text())
    camera, marker, paths = cfg["camera"], cfg["calibration"]["aruco"], cfg["paths"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=initial.config, help="Wrist defaults YAML")
    parser.add_argument("--eyehand-root", default=str(ROOT.parent / "rebot_eye_to_hand"),
                        help="Existing local acquisition project to reuse")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preview", "collect"):
        command = commands.add_parser(name)
        command.add_argument("--serial", default=camera["serial"], help="Wrist camera serial, explicitly bound")
        command.add_argument("--marker-mm", type=float, default=marker["marker_length_m"] * 1000, help="Measured BLACK outer edge, excluding white margin")
        command.add_argument("--dictionary", default=marker["dictionary"])
        command.add_argument("--marker-id", type=int, default=marker["target_marker_id"])
        command.add_argument("--fps", type=int, choices=(15, 30), default=camera["fps"])
        if name == "collect":
            command.add_argument("--output", required=True, help="New directory; existing directories are refused")
            command.add_argument("--manual", action="store_true", help="Active reference gravity controller; exclusive arm/gripper control")
            command.add_argument("--channel", default="can0")
            command.add_argument("--sdk", default=str(ROOT / paths["sdk"]))
            command.add_argument("--result", default=str(ROOT / paths["result"]))
            command.add_argument("--gravity-config", default=str(ROOT / paths["gravity"]))
    check = commands.add_parser("report", help="Offline only; never opens camera or CAN")
    check.add_argument("--dataset", required=True)
    check.add_argument("--output")
    check.add_argument("--min-samples", type=int, default=cfg["validation"]["min_samples"])
    check.add_argument("--translation-p95-mm", type=float, default=cfg["validation"]["translation_p95_mm"])
    check.add_argument("--rotation-p95-deg", type=float, default=cfg["validation"]["rotation_p95_deg"])
    check.add_argument("--reprojection-p95-px", type=float, default=cfg["validation"]["reprojection_p95_px"])
    args = parser.parse_args(argv)
    try:
        cli, geometry = dependencies(args.eyehand_root)
        if args.command == "report":
            limits = (args.translation_p95_mm, args.rotation_p95_deg, args.reprojection_p95_px)
            if args.min_samples < 15 or not all(np.isfinite(v) and v > 0 for v in limits):
                raise ValueError("Require min-samples>=15 and finite positive thresholds")
            report(args, geometry)
        elif args.command == "collect":
            target_config(args)
            collect(args, cli, geometry)
        else:
            with tempfile.TemporaryDirectory(prefix="wrist_marker_preview_") as tmp:
                target = Path(tmp) / "target.json"
                write_json(target, target_config(args))
                cli.preview(acquisition_args(args, target, Path(tmp), Path()))
    except (ValueError, RuntimeError, OSError, KeyError, ImportError, AttributeError, cv2.error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

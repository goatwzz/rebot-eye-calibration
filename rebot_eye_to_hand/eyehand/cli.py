from __future__ import annotations

import argparse
import csv
import hashlib
import json
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import checked, errors, evaluate, inverse, solve, transform
from .hardware import DEFAULT_SDK, PoseFile, RebotReader
from .vision import Camera, Target


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def target_config(args):
    config = load(args.target)
    if not 0 < config["marker_m"] < 1:
        raise ValueError("marker_m must be in meters, between 0 and 1")
    if config["type"] == "charuco" and not 0 < config["marker_m"] < config["square_m"]:
        raise ValueError("ChArUco marker_m must be smaller than square_m")
    return config


def open_camera(args):
    serial = args.serial
    config_path = Path(__file__).resolve().parents[1] / "camera_realsense.json"
    if args.camera == "realsense" and serial is None and config_path.exists():
        serial = load(config_path)["serial"]
    return Camera(args.camera, args.width, args.height, args.fps, serial, args.device, args.intrinsics,
                  allow_uncalibrated=args.command == "intrinsics")


def start_terminal_commands():
    commands = queue.Queue()

    def receive():
        while True:
            try:
                commands.put(input())
            except EOFError:
                commands.put(None)
                return

    threading.Thread(target=receive, name="calibration-terminal", daemon=True).start()
    return commands


def preview(args):
    """Check the exact image/intrinsics/target path without opening a robot bus."""
    target, camera = Target(target_config(args)), None
    try:
        camera = open_camera(args)
        if camera.k is None:
            raise ValueError("Provide --intrinsics for this camera")
        print("CAMERA ONLY: no robot bus opened. q/Esc to close.", flush=True)
        print(json.dumps({"camera": camera.metadata, "image_size": [camera.width, camera.height],
                          "K": camera.k.tolist(), "D": camera.d.tolist()}, indent=2), flush=True)
        while True:
            frame = camera.read()
            try:
                obs = target.detect(frame, camera.k, camera.d, args.max_pnp_px, args.ambiguity_gap_px)
                t = np.asarray(obs["T_camera_target"])
                cv2.drawFrameAxes(frame, camera.k, camera.d, cv2.Rodrigues(t[:3, :3])[0],
                                  t[:3, 3], target.config["marker_m"] / 2)
                label = f"VALID | RMS {obs['pnp_rms_px']:.2f}px | target z={t[2, 3]:.3f}m"
            except ValueError as exc:
                label = str(exc)
            cv2.putText(frame, label[:130], (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
            cv2.imshow("Camera only - no robot control", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if cv2.getWindowProperty("Camera only - no robot control", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        try:
            if camera:
                camera.close()
        finally:
            cv2.destroyAllWindows()


def read_collection_frame(camera, args, fresh=False):
    """Opt-in bounded retries for RealSense timeouts; never restart hardware."""
    retries = getattr(args, "camera_timeout_retries", 0)
    if not retries or camera.kind != "realsense":
        # Preserve existing callers, fake cameras and non-RealSense behavior.
        return camera.read(fresh=True) if fresh else camera.read()
    for attempt in range(retries + 1):
        try:
            return camera.read(fresh=fresh, timeout_ms=args.camera_timeout_ms)
        except RuntimeError as exc:
            if "Frame didn't arrive within" not in str(exc) or attempt == retries:
                raise
            print(f"[Camera] {exc}; retry {attempt + 1}/{retries}. No device restart or robot mode change.", flush=True)
            if cv2.waitKey(1) & 0xff in (ord("q"), 27):
                raise KeyboardInterrupt


def collect(args):
    config = target_config(args)
    target = Target(config)
    output = Path(args.output)
    if (output / "dataset.json").exists():
        raise ValueError("Dataset exists; use a new directory to avoid accidental mixing")
    output.mkdir(parents=True, exist_ok=True)
    camera, robot = None, None
    try:
        camera = open_camera(args)
        if camera.k is None:
            raise ValueError("OpenCV camera requires --intrinsics; first run the intrinsics command")
        # Check a real frame and initialize Qt before enabling hardware. A Qt
        # plugin failure can abort the process without running Python finally.
        first_frame = read_collection_frame(camera, args)
        print("Selected camera: " + json.dumps(camera.metadata, ensure_ascii=False), flush=True)
        cv2.namedWindow("reBot eye-to-hand", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("reBot eye-to-hand", first_frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return
        if args.manual:
            if args.manual_backend == "reference":
                from .reference_manual import ReferenceManualReader
                robot = ReferenceManualReader(reference_script=args.reference_script, channel=args.channel,
                                              ee_frame=args.ee_frame, base_frame=args.base_frame,
                                              gravity_config=args.gravity_config)
            else:
                from .manual import ManualGravityReader
                robot = ManualGravityReader(args.sdk, args.channel, args.ee_frame, args.base_frame,
                                            rate=args.manual_rate, kp=args.manual_kp, kd=args.manual_kd,
                                            startup_timeout=args.startup_timeout)
            robot.start()
            robot.command_queue = start_terminal_commands()
        else:
            robot = (PoseFile(args.pose_file, args.base_frame, args.ee_frame) if args.pose_file else
                     RebotReader(args.sdk, args.channel, args.ee_frame, args.base_frame))
        dataset = {"schema_version": 1, "dataset_id": str(uuid.uuid4()), "synthetic": False,
                   "target": config, "image_size": [camera.width, camera.height],
                   "camera": camera.metadata, "robot": robot.metadata,
                   "K": camera.k.tolist(), "D": camera.d.tolist(), "samples": []}
        save(output / "dataset.json", dataset)
        if not args.manual:
            print("Use existing robot controller to move and HOLD each pose. s/Enter: collect; q: quit.\n"
                  "This process performs position parameter reads only; it does not hold or move the arm.", flush=True)
        explicit_modes = args.manual and robot.uses_explicit_modes
        status = ("g: drag, h: hold, s/Enter: sample, q: finish" if explicit_modes else
                  "Drag by hand, stop, Enter:sample, q:finish" if args.manual else "s/Enter: sample, q: quit")
        while True:
            frame = read_collection_frame(camera, args)
            display = frame.copy()
            try:
                observation = target.detect(frame, camera.k, camera.d, args.max_pnp_px, args.ambiguity_gap_px)
                t = np.asarray(observation["T_camera_target"])
                cv2.drawFrameAxes(display, camera.k, camera.d, cv2.Rodrigues(t[:3, :3])[0], t[:3, 3], config["marker_m"] / 2)
                label = f"PnP {observation['pnp_rms_px']:.2f}px | {len(dataset['samples'])} saved"
            except ValueError as exc:
                label = str(exc)
            if args.manual:
                label += " | " + robot.status
            cv2.putText(display, label[:120], (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1)
            cv2.putText(display, status[:120], (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1)
            cv2.imshow("reBot eye-to-hand", display)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty("reBot eye-to-hand", cv2.WND_PROP_VISIBLE) < 1:
                break
            if args.manual:
                try:
                    line = robot.command_queue.get_nowait()
                    if line is None:
                        break
                    command = line.strip().lower()
                    if command == "pos":
                        pose, _ = robot.read()
                        print("T_base_ee:\n", pose, flush=True)
                    elif command in ("", "s", "g", "h", "q", "c"):
                        key = ord({"": "s", "c": "q"}.get(command, command))
                    else:
                        print("Enter/s=sample, g=drag, h=hold, pos=FK, q/c=finish", flush=True)
                except queue.Empty:
                    pass
            if key in (ord("q"), 27):
                break
            if args.manual and key in (ord("g"), ord("h")):
                if not explicit_modes:
                    print("Original manual mode: drag by hand, stop, then press Enter. g/h switching is not used.", flush=True)
                    continue
                try:
                    robot.set_mode("drag" if key == ord("g") else "hold")
                except RuntimeError as exc:
                    print(str(exc), flush=True)
                continue
            if key not in (ord("s"), 10, 13):
                continue
            try:
                if explicit_modes:
                    robot.set_mode("hold")
                # Continuous camera draining and repeated feedback reads over a dwell window.
                # This checks stationary acquisition, not hardware-clock synchronization.
                poses, robot_meta = [], []
                begin = time.monotonic()
                while time.monotonic() - begin < args.settle_s or len(poses) < 3:
                    g, meta = robot.read()
                    poses.append(g)
                    robot_meta.append(meta)
                    frame = read_collection_frame(camera, args)
                    cv2.waitKey(1)
                frame = read_collection_frame(camera, args, fresh=True)
                frame_meta = dict(camera.last_frame_metadata)
                g, meta = robot.read()
                poses.append(g)
                robot_meta.append(meta)
                movement = [errors(poses[0], t) for t in poses]
                mm, deg = np.max(movement, axis=0)
                if mm > args.static_mm or deg > args.static_deg:
                    raise ValueError(f"Robot moved during dwell: {mm:.2f}mm/{deg:.2f}deg")
                q_values = [m["q_rad"] for m in robot_meta if "q_rad" in m]
                if q_values and np.max(np.ptp(q_values, axis=0)) > np.radians(0.2):
                    raise ValueError("Joint movement >0.2deg during dwell")
                obs = target.detect(frame, camera.k, camera.d, args.max_pnp_px, args.ambiguity_gap_px)
                # Reject repetitions; nearby views can inflate count without adding information.
                for s in dataset["samples"]:
                    dt, dr = errors(g, np.asarray(s["T_base_ee"]))
                    if dt < 5 and dr < 5:
                        raise ValueError("Pose too close to an existing sample (<5mm and <5deg)")
                sample_id = str(uuid.uuid4())
                image_file = f"{len(dataset['samples']):03d}_{sample_id[:8]}.png"
                if not cv2.imwrite(str(output / image_file), frame):
                    raise OSError("Image save failed")
                dataset["samples"].append({"id": sample_id, "image": image_file,
                                           "T_base_ee": g.tolist(), **obs,
                                           "camera_frame": frame_meta, "robot_reads": robot_meta,
                                           "static_span_mm": float(mm), "static_span_deg": float(deg)})
                save(output / "dataset.json", dataset)
                status = f"Saved {len(dataset['samples'])} samples"
                print(status, flush=True)
            except (ValueError, RuntimeError, OSError) as exc:
                status = "Rejected: " + str(exc)
                print(status, flush=True)
    except KeyboardInterrupt:
        print("\nCollection interrupted; any successfully saved samples remain on disk.", flush=True)
    finally:
        # Stop imaging first. In manual mode close() intentionally keeps holding
        # until the operator supports the arm and enters the release command.
        try:
            if camera:
                camera.close()
        finally:
            try:
                cv2.destroyAllWindows()
            finally:
                if robot:
                    robot.close()


def dataset_path(path):
    path = Path(path)
    return path / "dataset.json" if path.is_dir() else path


def provenance(data):
    return {k: data[k] for k in ["target", "image_size", "camera", "robot", "K", "D", "synthetic"]}


def calibrate(args):
    path = dataset_path(args.dataset)
    data = load(path)
    result = solve(data["samples"], refine=not args.no_refine,
                   refinement=getattr(args, "refinement", "closure"))
    result["provenance"] = provenance(data)
    result["training_dataset_id"] = data["dataset_id"]
    result["training_dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(args.output, result)
    print(json.dumps({"output": args.output, "training": result["training"]["summary"],
                      "warnings": result["warnings"], "acceptance": result["acceptance"]}, indent=2))


def identify_kinematic_model(args):
    from .kinematics import export_corrected_urdf, identify_kinematics
    path = dataset_path(args.dataset)
    data = load(path)
    initial = load(args.initial) if args.initial else None
    if initial is not None and initial.get("training_dataset_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Initial result does not match this dataset file hash")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Identification output directory already exists; use a new directory")
    correction, result = identify_kinematics(data, initial)
    result["training_dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    output.mkdir(parents=True)
    urdf = output / "corrected_arm.urdf"
    export_corrected_urdf(correction, urdf)
    correction["exported_urdf"] = str(urdf.resolve())
    correction["exported_urdf_sha256"] = hashlib.sha256(urdf.read_bytes()).hexdigest()
    save(output / "kinematic_correction.json", correction)
    save(output / "result.json", result)
    print(json.dumps({"output": str(output), "joint_offsets_deg": correction["joint_offsets_deg"],
                      "training": result["training"]["summary"], "optimization": result["optimization"],
                      "acceptance": result["acceptance"]}, indent=2))


def validate(args):
    result, data = load(args.result), load(dataset_path(args.dataset))
    if data["dataset_id"] == result["training_dataset_id"]:
        raise ValueError("Validation must use a separately collected dataset")
    if set(result["training_ids"]) & {s["id"] for s in data["samples"]}:
        raise ValueError("Training and validation sample IDs overlap")
    if len(data["samples"]) < 8:
        raise ValueError("Need at least 8 independent validation poses")
    original = result["provenance"]
    current = provenance(data)
    for key in ["target", "image_size", "K", "D", "synthetic"]:
        if current[key] != original[key]:
            raise ValueError(f"Validation {key} differs from training")
    for group, keys in [("robot", ["base_frame", "ee_frame", "urdf_sha256"]), ("camera", ["serial", "device", "kind"])]:
        for key in keys:
            if original[group].get(key) != current[group].get(key):
                raise ValueError(f"Validation {group}.{key} differs from training")
    if len({s["id"] for s in data["samples"]}) != len(data["samples"]):
        raise ValueError("Duplicate validation sample IDs")
    correction = result.get("kinematic_correction")
    if correction:
        from .kinematics import corrected_dataset
        data = corrected_dataset(data, correction)
    elif data["robot"].get("kinematic_correction"):
        raise ValueError("Corrected dataset requires a result with the same kinematic correction")
    report = evaluate(data["samples"], checked(result["T_base_camera"]), checked(result["T_ee_target"]))
    if correction:
        report["kinematic_correction_applied"] = correction
    report["thresholds"] = {"translation_p95_mm": args.translation_mm, "rotation_p95_deg": args.rotation_deg,
                            "predicted_reprojection_p95_px": args.pixel_px}
    checks = [report["summary"]["translation_mm"]["p95"] <= args.translation_mm,
              report["summary"]["rotation_deg"]["p95"] <= args.rotation_deg]
    if "predicted_reprojection_rms_px" in report["summary"]:
        checks.append(report["summary"]["predicted_reprojection_rms_px"]["p95"] <= args.pixel_px)
    report["status"] = "PASS_INTERNAL_CONSISTENCY" if all(checks) else "FAIL"
    report["physical_accuracy_verified"] = False
    report["synthetic"] = data["synthetic"]
    report["validation_dataset_id"] = data["dataset_id"]
    save(args.output, report)
    print(json.dumps(report, indent=2))
    if not all(checks):
        raise SystemExit(2)


def intrinsics(args):
    if args.square_m <= 0 or args.columns < 3 or args.rows < 3:
        raise ValueError("Invalid chessboard size")
    cam = open_camera(args)
    objects, images = [], []
    obj = np.zeros((args.columns * args.rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:args.columns, 0:args.rows].T.reshape(-1, 2) * args.square_m
    output = Path(args.output)
    if output.exists():
        cam.close()
        raise ValueError("Intrinsics output exists; use a new path")
    raw = output.parent / (output.stem + "_images")
    raw.mkdir(parents=True, exist_ok=True)
    print("Chessboard inner corners; s: save diverse tilted views, c: calibrate (>=20), q: quit")
    try:
        while True:
            frame = cam.read()
            found, corners = cv2.findChessboardCornersSB(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                                        (args.columns, args.rows))
            view = frame.copy()
            if found:
                cv2.drawChessboardCorners(view, (args.columns, args.rows), corners, True)
            cv2.putText(view, f"{len(images)} views | s:save c:calibrate q:quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow("Camera intrinsics", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return
            if key == ord("s") and found:
                objects.append(obj.copy())
                images.append(corners.astype(np.float32))
                if not cv2.imwrite(str(raw / f"{len(images):03d}.png"), frame):
                    raise OSError("Image write failed")
            if key == ord("c") and len(images) >= 20:
                rms, k, d, rv, tv = cv2.calibrateCamera(objects, images, (cam.width, cam.height), None, None)
                per_view = []
                for o, uv, r, t in zip(objects, images, rv, tv):
                    pred, _ = cv2.projectPoints(o, r, t, k, d)
                    per_view.append(float(np.sqrt(np.mean(np.sum((pred - uv) ** 2, axis=2)))))
                save(output, {"K": k.tolist(), "D": d.reshape(-1).tolist(), "image_size": [cam.width, cam.height],
                              "rms_px": rms, "per_view_rms_px": per_view, "camera": cam.metadata,
                              "inner_corners": [args.columns, args.rows], "square_m": args.square_m})
                print(f"Saved {output}; RMS={rms:.3f}px. Inspect per-view errors and held-out views.")
                return
    finally:
        cam.close()
        cv2.destroyAllWindows()


def synthetic_dataset(seed=7, count=40, noisy=False):
    rng = np.random.default_rng(seed)
    x = transform(Rotation.from_euler("xyz", [2.8, 0.1, -0.3]).as_matrix(), [0.2, -0.3, 0.9])
    y = transform(Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix(), [0.02, 0.01, 0.1])
    data = {"schema_version": 1, "dataset_id": str(uuid.uuid4()), "synthetic": True,
            "target": {"type": "synthetic"}, "camera": {"kind": "synthetic"},
            "robot": {"base_frame": "base_link", "ee_frame": "gripper_end"},
            "image_size": [1280, 720], "K": [[900, 0, 640], [0, 900, 360], [0, 0, 1]], "D": [0.] * 5,
            "samples": []}
    for _ in range(count):
        g = transform(Rotation.from_rotvec(rng.normal(0, 0.5, 3)).as_matrix(),
                      rng.uniform([0.15, -0.15, 0.2], [0.4, 0.15, 0.5]))
        c = inverse(x) @ g @ y
        if noisy:
            c = c @ transform(Rotation.from_rotvec(rng.normal(0, np.radians(0.05), 3)).as_matrix(),
                              rng.normal(0, 0.00015, 3))
        data["samples"].append({"id": str(uuid.uuid4()), "T_base_ee": g.tolist(), "T_camera_target": c.tolist()})
    return data, x, y


def self_test(args):
    train, x, y = synthetic_dataset(count=35, noisy=True)
    test, _, _ = synthetic_dataset(seed=22, count=15, noisy=True)
    result = solve(train["samples"])
    result["provenance"], result["training_dataset_id"] = provenance(train), train["dataset_id"]
    dx, dy = errors(np.asarray(result["T_base_camera"]), x), errors(np.asarray(result["T_ee_target"]), y)
    report = evaluate(test["samples"], np.asarray(result["T_base_camera"]), np.asarray(result["T_ee_target"]))
    if dx[0] > 1 or dx[1] > 0.1 or dy[0] > 1 or dy[1] > 0.1:
        raise RuntimeError(f"Synthetic recovery failed: {dx}, {dy}")
    output = Path(args.output)
    save(output / "train/dataset.json", train)
    save(output / "validation/dataset.json", test)
    save(output / "result.json", result)
    report["truth_error"] = {"camera_translation_mm": dx[0], "camera_rotation_deg": dx[1],
                             "mount_translation_mm": dy[0], "mount_rotation_deg": dy[1]}
    report["synthetic"] = True
    save(output / "self_test_report.json", report)
    print(json.dumps(report["truth_error"], indent=2))


def check_points(args):
    """Compare independently measured correspondences using the fixed extrinsic."""
    result = load(args.result)
    x = checked(result["T_base_camera"])
    with Path(args.csv).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 5:
        raise ValueError("Need at least 5 independently measured point pairs, recommend >=10")
    records, residuals = [], []
    for i, row in enumerate(rows):
        camera = np.array([float(row[k]) for k in ["camera_x_m", "camera_y_m", "camera_z_m"]])
        base = np.array([float(row[k]) for k in ["base_x_m", "base_y_m", "base_z_m"]])
        if not np.isfinite(np.r_[camera, base]).all() or camera[2] <= 0:
            raise ValueError("Points must be finite, metric, and in front of the camera")
        predicted = (x @ np.r_[camera, 1])[:3]
        residual = (predicted - base) * 1000
        residuals.append(residual)
        records.append({"id": row.get("id", str(i)), "predicted_base_m": predicted.tolist(),
                        "reference_base_m": base.tolist(), "error_xyz_mm": residual.tolist(),
                        "error_norm_mm": float(np.linalg.norm(residual))})
    residuals = np.asarray(residuals)
    lengths = np.linalg.norm(residuals, axis=1)
    report = {"count": len(rows), "fit_performed": False, "bias_xyz_mm": residuals.mean(axis=0).tolist(),
              "rms_mm": float(np.sqrt(np.mean(lengths ** 2))), "p95_mm": float(np.percentile(lengths, 95)),
              "max_mm": float(lengths.max()), "threshold_p95_mm": args.threshold_mm, "points": records,
              "status": "PASS_MEASURED_POINTS" if np.percentile(lengths, 95) <= args.threshold_mm else "FAIL",
              "note": "Accuracy includes camera 3D measurement and reference/TCP uncertainty; validate across the workspace"}
    save(args.output, report)
    print(json.dumps(report, indent=2))
    if report["status"] == "FAIL":
        raise SystemExit(2)


def gravity_settings(args):
    from .gravity_parameters import load_gravity_parameters, set_gravity_parameters
    if args.gravity_action == "show":
        parameters = load_gravity_parameters(args.config)
    else:
        updates = {name: getattr(args, name) for name in
                   ("tau_scale", "kp", "kd", "payload_mass_kg", "payload_frame", "payload_com_m",
                    "wrist_camera_mass_kg", "wrist_camera_frame", "wrist_camera_com_m")
                   if getattr(args, name) is not None}
        if (args.joint is None) != (args.scale is None):
            raise ValueError("--joint and --scale must be supplied together")
        if args.joint is not None:
            if args.tau_scale is not None:
                raise ValueError("Use either --tau-scale or --joint/--scale")
            scale = load_gravity_parameters(args.config)["tau_scale"]
            scale[args.joint - 1] = args.scale
            updates["tau_scale"] = scale
        if not updates:
            raise ValueError("Supply gravity scale, gains or payload settings")
        parameters = set_gravity_parameters(args.config, **updates)
    print(json.dumps(parameters, ensure_ascii=False, indent=2))


def main():
    from .gravity_parameters import DEFAULT_GRAVITY_CONFIG
    parser = argparse.ArgumentParser(description="reBot eye-to-hand calibration: fixed camera, target on hand")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("target", help="Generate target PNG + physical-size SVG")
    gen.add_argument("--target", required=True)
    gen.add_argument("--output", required=True)
    gen.set_defaults(func=lambda a: Target(target_config(a)).write(a.output))
    capture = sub.add_parser("collect", help="Stationary collection; --manual opts into active gravity control")
    capture.add_argument("--target", default=str(Path(__file__).resolve().parents[1] / "target_aruco_60mm.json"))
    capture.add_argument("--output", required=True)
    capture.add_argument("--pose-file", help="Atomic pose JSON from the existing controller; otherwise RS parameter reader")
    capture.add_argument("--manual", action="store_true", help="Active gravity compensation + drag/hold; exclusive motor control")
    capture.add_argument("--manual-backend", choices=["reference", "parameter"], default="reference",
                         help="reference: reuse original script's controller (default); parameter: previous custom g/h controller")
    from .reference_manual import DEFAULT_REFERENCE
    capture.add_argument("--reference-script", default=str(DEFAULT_REFERENCE))
    capture.add_argument("--gravity-config", default=str(DEFAULT_GRAVITY_CONFIG),
                         help="Independent JSON gravity settings for reference manual mode")
    capture.add_argument("--manual-rate", type=float, default=40.)
    capture.add_argument("--manual-kp", type=float, default=8.)
    capture.add_argument("--manual-kd", type=float, default=1.5)
    capture.add_argument("--startup-timeout", type=float, default=15., help="Wait for two full feedback reads before enable (seconds)")
    capture.add_argument("--sdk", default=str(DEFAULT_SDK))
    capture.add_argument("--channel", default=None)
    capture.add_argument("--base-frame", default="base_link")
    capture.add_argument("--ee-frame", default="gripper_end")
    capture.add_argument("--settle-s", type=float, default=1.0)
    capture.add_argument("--static-mm", type=float, default=0.5)
    capture.add_argument("--static-deg", type=float, default=0.2)
    capture.add_argument("--max-pnp-px", type=float, default=1.0)
    capture.add_argument("--ambiguity-gap-px", type=float, default=0.15)
    capture.set_defaults(func=collect)
    view = sub.add_parser("preview", help="Camera/marker preview only; no robot bus opened")
    view.add_argument("--target", default=str(Path(__file__).resolve().parents[1] / "target_aruco_60mm.json"))
    view.add_argument("--max-pnp-px", type=float, default=1.)
    view.add_argument("--ambiguity-gap-px", type=float, default=0.15)
    view.set_defaults(func=preview)
    intrinsic = sub.add_parser("intrinsics", help="Chessboard intrinsic calibration")
    intrinsic.add_argument("--output", required=True)
    intrinsic.add_argument("--columns", type=int, default=9)
    intrinsic.add_argument("--rows", type=int, default=6)
    intrinsic.add_argument("--square-m", type=float, required=True)
    intrinsic.set_defaults(func=intrinsics)
    for p in [capture, intrinsic, view]:
        p.add_argument("--camera", choices=["realsense", "opencv"], default="realsense")
        p.add_argument("--width", type=int, default=1280)
        p.add_argument("--height", type=int, default=720)
        p.add_argument("--fps", type=int, default=30)
        p.add_argument("--serial", help="Override serial pinned in camera_realsense.json; never fall back to another device")
        p.add_argument("--device", type=int, default=0)
        p.add_argument("--intrinsics", help="K,D,image_size JSON matching the exact camera stream")
    cal = sub.add_parser("calibrate")
    cal.add_argument("--dataset", required=True)
    cal.add_argument("--output", required=True)
    cal.add_argument("--no-refine", action="store_true")
    cal.add_argument("--refinement", choices=["closure", "reprojection"], default="closure",
                     help="closure: original 3D pose residual; reprojection: fit measured image corners with fixed K/D")
    cal.set_defaults(func=calibrate)
    identify = sub.add_parser("identify-kinematics", help="Offline joint2..5 effective FK correction and paired hand-eye fit")
    identify.add_argument("--dataset", required=True)
    identify.add_argument("--initial", help="Optional nominal calibration result from the same dataset")
    identify.add_argument("--output", required=True, help="New directory for correction, local URDF and paired result")
    identify.set_defaults(func=identify_kinematic_model)
    val = sub.add_parser("validate")
    val.add_argument("--dataset", required=True)
    val.add_argument("--result", required=True)
    val.add_argument("--output", required=True)
    val.add_argument("--translation-mm", type=float, default=5)
    val.add_argument("--rotation-deg", type=float, default=1)
    val.add_argument("--pixel-px", type=float, default=3)
    val.set_defaults(func=validate)
    check = sub.add_parser("self-test")
    check.add_argument("--output", default="reports/synthetic")
    check.set_defaults(func=self_test)
    measured = sub.add_parser("check-points", help="Verify independent measured camera/base 3D point pairs; no fit")
    measured.add_argument("--result", required=True)
    measured.add_argument("--csv", required=True)
    measured.add_argument("--output", required=True)
    measured.add_argument("--threshold-mm", type=float, default=5)
    measured.set_defaults(func=check_points)
    point = sub.add_parser("point", help="Transform a metric 3D color-camera point into the robot base")
    point.add_argument("--result", required=True)
    point.add_argument("--xyz", nargs=3, type=float, required=True)
    point.set_defaults(func=lambda a: print((checked(load(a.result)["T_base_camera"]) @ np.r_[a.xyz, 1])[:3]))
    gravity = sub.add_parser("gravity", help="Show/set independent gravity parameters; no hardware access")
    gravity_sub = gravity.add_subparsers(dest="gravity_action", required=True)
    show = gravity_sub.add_parser("show")
    edit = gravity_sub.add_parser("set")
    for p in (show, edit):
        p.add_argument("--config", default=str(DEFAULT_GRAVITY_CONFIG))
        p.set_defaults(func=gravity_settings)
    edit.add_argument("--tau-scale", type=float, nargs=6, metavar=("J1", "J2", "J3", "J4", "J5", "J6"))
    edit.add_argument("--joint", type=int, choices=range(1, 7))
    edit.add_argument("--scale", type=float)
    edit.add_argument("--kp", type=float)
    edit.add_argument("--kd", type=float)
    edit.add_argument("--payload-mass-kg", type=float, help="Additional payload mass in kilograms (0 disables)")
    edit.add_argument("--payload-frame", help="Model frame in which payload COM is expressed")
    edit.add_argument("--payload-com-m", type=float, nargs=3, metavar=("X", "Y", "Z"),
                      help="Payload COM relative to payload frame in metres")
    edit.add_argument("--wrist-camera-mass-kg", type=float, help="Additional wrist camera + mount mass in kg")
    edit.add_argument("--wrist-camera-frame", help="Model frame for wrist camera COM (default link6)")
    edit.add_argument("--wrist-camera-com-m", type=float, nargs=3, metavar=("X", "Y", "Z"),
                      help="Wrist camera COM relative to wrist camera frame in metres")
    args = parser.parse_args()
    if getattr(args, "manual", False) and args.pose_file:
        parser.error("--manual and --pose-file are mutually exclusive")
    if hasattr(args, "settle_s") and (args.settle_s < 0.5 or min(args.static_mm, args.static_deg, args.max_pnp_px) <= 0 or args.ambiguity_gap_px < 0):
        parser.error("Require settle-s>=0.5, positive thresholds, ambiguity-gap-px>=0")
    try:
        args.func(args)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

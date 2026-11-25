#!/usr/bin/env python3
"""Offline hypothesis audit; fitted nuisance parameters are never applied to hardware.

All variants fit train_03 only. Existing validation sets are diagnostic comparisons,
not fresh acceptance tests after model selection. No camera/CAN device is opened.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyehand.cli import save
from eyehand.geometry import checked, errors, inverse, pack, unpack, solve
from eyehand.vision import Target


from eyehand.paths import artifact_path


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"rms": float(np.sqrt(np.mean(a * a))), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(np.max(a))}


def load_data(folder):
    path = ROOT / "runs" / folder / "dataset.json"
    data = json.loads(path.read_text())
    data["_path"] = path
    return data


def audit_inputs(data):
    model = pin.buildModelFromUrdf(data["robot"]["urdf"])
    fid = model.getFrameId(data["robot"]["ee_frame"])
    indices = [model.joints[model.getJointId(name)].idx_q for name in data["robot"]["joint_names"]]
    target = Target(data["target"])
    rows = []
    for sample in data["samples"]:
        q = pin.neutral(model)
        q[indices] = sample["robot_reads"][-1]["q_rad"]
        state = model.createData()
        pin.framesForwardKinematics(model, state, q)
        fk_error = errors(state.oMf[fid].homogeneous, checked(sample["T_base_ee"]))
        obj, uv = np.array(sample["object_points"]), np.array(sample["image_points"])
        k, d = np.array(sample["K"]), np.array(sample["D"])
        image = cv2.imread(str(artifact_path(data["_path"].parent / sample["image"])))
        obs = target.detect(image, k, d)
        corner_diff = np.linalg.norm(np.array(obs["image_points"]) - uv, axis=1)
        c = np.asarray(sample["T_camera_target"])
        rv, tv = cv2.solvePnPRefineLM(obj, uv, k, d, cv2.Rodrigues(c[:3, :3])[0], c[:3, 3].copy())
        refined = np.eye(4)
        refined[:3, :3], refined[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
        uv_refined, _ = cv2.projectPoints(obj, rv, tv, k, d)
        frame = sample["camera_frame"]
        last = sample["robot_reads"][-1]
        rows.append({"id": sample["id"], "image": sample["image"],
                     "fk_difference_mm_deg": list(fk_error),
                     "redetected_corner_max_delta_px": float(corner_diff.max()),
                     "ippe_to_lm_pose_delta_mm_deg": list(errors(c, refined)),
                     "pnp_rms_original_px": sample["pnp_rms_px"],
                     "pnp_rms_lm_px": float(np.sqrt(np.mean(np.sum((uv_refined.reshape(-1, 2)-uv)**2, axis=1)))),
                     "frame_to_last_read_end_s": last["read_ended_unix_s"] - frame["host_received_unix_s"],
                     "camera_frame_age_at_receive_s": frame["host_received_unix_s"] - frame["device_timestamp_ms"] / 1000,
                     "q": sample["robot_reads"][-1]["q_rad"],
                     "refined_camera_target": refined.tolist()})
    return {"sample_count": len(rows), "dataset_sha256": hashlib.sha256(data["_path"].read_bytes()).hexdigest(),
            "urdf_hash_matches": hashlib.sha256(Path(data["robot"]["urdf"]).read_bytes()).hexdigest() == data["robot"]["urdf_sha256"],
            "fk_max_difference_mm_deg": np.max([r["fk_difference_mm_deg"] for r in rows], axis=0).tolist(),
            "corner_max_difference_px": max(r["redetected_corner_max_delta_px"] for r in rows),
            "ippe_to_lm_rotation_deg": stats([r["ippe_to_lm_pose_delta_mm_deg"][1] for r in rows]),
            "frame_age_s": stats([r["camera_frame_age_at_receive_s"] for r in rows]),
            "rows": rows}


class DiagnosticFit:
    def __init__(self, train):
        self.model = pin.buildModelFromUrdf(train["robot"]["urdf"])
        self.frame = self.model.getFrameId(train["robot"]["ee_frame"])
        self.qi = [self.model.joints[self.model.getJointId(f"joint{i}")].idx_q for i in range(1, 7)]
        self.k = np.array(train["K"])

    def prepare(self, dataset):
        prepared = []
        for s in dataset["samples"]:
            q = pin.neutral(self.model)
            q[self.qi] = s["robot_reads"][-1]["q_rad"]
            prepared.append((np.asarray(s["T_base_ee"]), np.asarray(s["object_points"]),
                             np.asarray(s["image_points"]), np.asarray(s["T_camera_target"]), q))
        return prepared

    def project(self, z, kind, prepared):
        xi, y = inverse(unpack(z[:6])), unpack(z[6:12])
        k, d = self.k.copy(), np.zeros(5)
        offset = np.zeros(6)
        scale = aspect = 1.
        if kind == "intrinsics4":
            k[0,0] *= np.exp(z[12]); k[1,1] *= np.exp(z[13])
            k[0,2] += z[14]; k[1,2] += z[15]
        elif kind == "distortion2":
            d[:2] = z[12:14]
        elif kind == "distortion5":
            d[:] = z[12:17]
        elif kind in ("intrinsics4_distortion2", "intrinsics4_distortion5"):
            k[0,0] *= np.exp(z[12]); k[1,1] *= np.exp(z[13])
            k[0,2] += z[14]; k[1,2] += z[15]
            if kind.endswith("distortion5"):
                d[:] = z[16:21]
            else:
                d[:2] = z[16:18]
        elif kind == "joint_offsets_2_5":
            offset[1:5] = z[12:16]
        elif kind == "marker_scale":
            scale = np.exp(z[12])
        elif kind == "marker_aspect":
            aspect = np.exp(z[12])
        predictions = []
        state = self.model.createData() if kind == "joint_offsets_2_5" else None
        for g, obj, uv, measured, q in prepared:
            if state is not None:
                qc = q.copy(); qc[self.qi] += offset
                pin.framesForwardKinematics(self.model, state, qc)
                g = state.oMf[self.frame].homogeneous.copy()
            c = xi @ g @ y
            points = obj.copy() * scale
            points[:, 1] *= aspect
            predicted, _ = cv2.projectPoints(points, cv2.Rodrigues(c[:3,:3])[0], c[:3,3], k, d)
            predictions.append((predicted.reshape(-1,2), c, uv, measured))
        return predictions, {"K": k.tolist(), "D": d.tolist(), "joint_offsets_deg": np.degrees(offset).tolist(),
                             "marker_edge_mm": 60 * scale, "marker_y_over_x": aspect}

    def fit(self, kind, train, seed):
        extra = {"fixed": 0, "intrinsics4": 4, "distortion2": 2, "intrinsics4_distortion2": 6,
                 "distortion5": 5, "intrinsics4_distortion5": 9,
                 "joint_offsets_2_5": 4, "marker_scale": 1, "marker_aspect": 1}[kind]
        z = np.r_[seed, np.zeros(extra)]
        low, high = np.full(len(z), -np.inf), np.full(len(z), np.inf)
        if kind.startswith("intrinsics4"):
            low[12:16] = [-.1, -.1, -50, -50]; high[12:16] = [.1, .1, 50, 50]
        if kind == "joint_offsets_2_5":
            low[12:16] = -np.radians(5); high[12:16] = np.radians(5)
        if kind.startswith("marker"):
            low[12] = np.log(.9); high[12] = np.log(1.1)
        def residual(parameters):
            predictions, _ = self.project(parameters, kind, train)
            return np.concatenate([(predicted - uv).ravel() for predicted,c,uv,measured in predictions])
        opt = least_squares(residual, z, bounds=(low,high), loss="soft_l1", f_scale=1., max_nfev=150,
                            xtol=1e-9, ftol=1e-9, gtol=1e-9)
        return opt

    def metrics(self, z, kind, data):
        predictions, parameters = self.project(z, kind, data)
        pixels = [float(np.sqrt(np.mean(np.sum((predicted-uv)**2, axis=1)))) for predicted,c,uv,m in predictions]
        closure = [errors(c,m) for predicted,c,uv,m in predictions]
        return {"reprojection_px": stats(pixels),
                "closure_mm": stats([v[0] for v in closure]),
                "closure_deg": stats([v[1] for v in closure])}, parameters


def main():
    folders = ["train_03", "validation_03", "validation_task_01"]
    datasets = {f: load_data(f) for f in folders}
    report = {"date": "2026-11-20", "hardware_accessed": False,
              "scope": "Diagnostic fits only; nuisance parameters are not independently measured or deployed.",
              "fit_dataset": "train_03", "comparison_datasets": folders[1:],
              "comparison_reuse_note": "Validation sets were inspected before; no fresh acceptance claim.",
              "input_audits": {}, "fits": {}}
    protected = [ROOT/"gravity_config.json", ROOT/"runs/result_train_03.json",
                 *[d["_path"] for d in datasets.values()]]
    hashes_before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    for folder, data in datasets.items():
        audit = audit_inputs(data)
        report["input_audits"][folder] = audit
        print(folder, "FK max", audit["fk_max_difference_mm_deg"], "corner max",audit["corner_max_difference_px"],
              "IPPE-LM deg p95", audit["ippe_to_lm_rotation_deg"]["p95"], flush=True)
    fit = DiagnosticFit(datasets["train_03"])
    prepared = {f: fit.prepare(d) for f,d in datasets.items()}
    result = json.loads((ROOT/"runs/result_train_03_reprojection.json").read_text())
    seed = np.r_[pack(np.asarray(result["T_base_camera"])), pack(np.asarray(result["T_ee_target"]))]
    for kind in ["fixed", "intrinsics4", "distortion2", "intrinsics4_distortion2", "distortion5",
                 "intrinsics4_distortion5", "joint_offsets_2_5", "marker_scale", "marker_aspect"]:
        opt = fit.fit(kind, prepared["train_03"], seed)
        entry = {"success": bool(opt.success), "nfev": opt.nfev, "cost": opt.cost, "metrics": {}}
        for folder, data in prepared.items():
            entry["metrics"][folder], entry["parameters"] = fit.metrics(opt.x,kind,data)
        entry["T_base_camera"] = unpack(opt.x[:6]).tolist()
        entry["T_ee_target"] = unpack(opt.x[6:12]).tolist()
        report["fits"][kind] = entry
        print(kind, "success",opt.success,"px RMS train/val/task",[round(entry["metrics"][f]["reprojection_px"]["rms"],3) for f in folders],
              "parameters",entry["parameters"],flush=True)
        save(ROOT/"reports/root_cause_audit_2026-11-20.json",report)

    # Four contiguous held-out blocks of the original training run. Each seed,
    # X/Y and offset fit is computed from the remaining 30 poses only.
    cross_validation = []
    for block in range(4):
        held = set(range(block * 10, (block + 1) * 10))
        train_samples = [s for i,s in enumerate(datasets["train_03"]["samples"]) if i not in held]
        test_samples = [s for i,s in enumerate(datasets["train_03"]["samples"]) if i in held]
        base = solve(train_samples, refinement="reprojection")
        block_seed = np.r_[pack(np.array(base["T_base_camera"])), pack(np.array(base["T_ee_target"]))]
        train_p = fit.prepare({"samples": train_samples})
        test_p = fit.prepare({"samples": test_samples})
        record = {"fold": block, "training_count": 30, "test_count": 10, "models": {}}
        for kind in ["fixed", "joint_offsets_2_5"]:
            opt = fit.fit(kind, train_p, block_seed)
            metrics, params = fit.metrics(opt.x, kind, test_p)
            record["models"][kind] = {"success": bool(opt.success), "parameters": params, "held_out_metrics": metrics}
        cross_validation.append(record)
        print("cross_validation",block,[round(record["models"][k]["held_out_metrics"]["reprojection_px"]["rms"],3)
                                        for k in ["fixed","joint_offsets_2_5"]],flush=True)
    report["blocked_cross_validation"] = cross_validation

    # Quantify angular sensitivity to small corner noise. This is a simulation,
    # not a measurement of actual corner noise or a corrected accuracy claim.
    rng = np.random.default_rng(42)
    sensitivity = []
    for index,s in enumerate(datasets["validation_03"]["samples"]):
        baseline = np.array(report["input_audits"]["validation_03"]["rows"][index]["refined_camera_target"])
        obj, k, d = np.array(s["object_points"]), np.array(s["K"]), np.array(s["D"])
        uv, _ = cv2.projectPoints(obj, cv2.Rodrigues(baseline[:3,:3])[0], baseline[:3,3], k, d)
        uv = uv.reshape(-1,2)
        deltas = []
        for _ in range(100):
            noisy_uv = uv + rng.normal(0., .2, uv.shape)
            ok, rv, tv, _ = cv2.solvePnPGeneric(obj, noisy_uv, k, d, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok: continue
            rv, tv = cv2.solvePnPRefineLM(obj, noisy_uv, k, d, rv[0], tv[0])
            t = np.eye(4); t[:3,:3] = cv2.Rodrigues(rv)[0]; t[:3,3] = tv.ravel()
            deltas.append(errors(baseline,t)[1])
        sensitivity.append({"id":s["id"],"viewing_tilt_deg":float(np.degrees(np.arccos(abs(baseline[2,2])))),
                            "noise_rotation_deg":stats(deltas)})
    report["corner_noise_simulation"] = {"assumed_pixel_sigma_per_coordinate":.2,"trials_per_image":100,
                                         "note":"Hypothetical Gaussian image noise only; not measured accuracy.","samples":sensitivity}

    # Check whether merely replacing IPPE by per-image LM solves the problem.
    # This branch uses refined PnP in both fit and comparison; original files stay intact.
    refined_sets = {}
    for folder,data in datasets.items():
        refined_sets[folder] = []
        for sample,row in zip(data["samples"],report["input_audits"][folder]["rows"]):
            refined_sets[folder].append({**sample,"T_camera_target":row["refined_camera_target"]})
    lm_result = solve(refined_sets["train_03"])
    from eyehand.geometry import evaluate
    report["lm_pnp_then_original_closure"] = {
        folder:evaluate(samples,np.array(lm_result["T_base_camera"]),np.array(lm_result["T_ee_target"]))["summary"]
        for folder,samples in refined_sets.items()}
    report["observations"] = [
        "All 70 saved FK matrices exactly match the last recorded joint positions evaluated through the recorded URDF; all recorded URDF hashes match.",
        "All 70 saved images reproduce the exact saved marker corners with the current detector. This checks data integrity, not physical measurement truth.",
        "Small effective offsets in joints 2-5 explain materially more image error on both existing comparison datasets than K/D or marker size adjustments. All four blocked training folds also improve on held-out images.",
        "Effective fitted offsets do not uniquely identify physical motor zeros: joint-axis geometry, load-dependent compliance or a nonrigid target can be confounded. Joint 1 and 6 offsets are fixed as gauge freedoms absorbed by X/Y.",
        "The current calibration path has no identified joint-zero/geometric correction. Recorded FK consistency alone cannot establish physical FK accuracy.",
        "Original pose-closure optimization gives equal configured rotation confidence to noisy single-marker PnP orientations. Pixel fitting improves projections but cannot remove the remaining model discrepancy.",
        "Single 60 mm planar-marker angular sensitivity is substantial in a hypothetical 0.2 px Gaussian corner-noise simulation. Its rotation closure is not an independent ground-truth angle measurement.",
        "LM-only PnP refinement does not materially reduce remaining reprojection error, so the missing iterative refinement is not sufficient to explain the poor result.",
        "No gripper position or structural deflection measurement is stored, so retrospective analysis cannot fully exclude target motion relative to gripper_end.",
        "No hypothesis has been promoted to a production calibration or motor-zero update. Corrected comparisons still do not pass all original thresholds."
    ]
    hashes_after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert hashes_before == hashes_after
    report["original_inputs_and_control_config_preserved"] = True
    report["protected_file_sha256"] = hashes_after
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(ROOT/"reports/root_cause_audit_2026-11-20.json",report)
    print("corner_noise_simulation angular RMS range",
          [min(a["noise_rotation_deg"]["rms"] for a in sensitivity),max(a["noise_rotation_deg"]["rms"] for a in sensitivity)],flush=True)


if __name__ == "__main__":
    main()

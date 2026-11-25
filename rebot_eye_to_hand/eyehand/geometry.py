from __future__ import annotations

import itertools

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def transform(rotation=None, translation=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = np.asarray(translation).reshape(3)
    return result


def checked(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("Expected a finite 4x4 transform")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("Invalid homogeneous bottom row")
    r = value[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError("Rotation must belong to SO(3)")
    return value


def inverse(t):
    return transform(t[:3, :3].T, -t[:3, :3].T @ t[:3, 3])


def pack(t):
    return np.r_[Rotation.from_matrix(t[:3, :3]).as_rotvec(), t[:3, 3]]


def unpack(x):
    return transform(Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6])


def average(transforms):
    return transform(Rotation.from_matrix(np.array([t[:3, :3] for t in transforms])).mean().as_matrix(),
                     np.mean([t[:3, 3] for t in transforms], axis=0))


def errors(a, b):
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000), float(
        np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()))


def diversity(poses):
    vectors = np.array([Rotation.from_matrix((inverse(a) @ b)[:3, :3]).as_rotvec()
                        for a, b in itertools.combinations(poses, 2)])
    singular = np.linalg.svd(vectors, compute_uv=False)
    maximum = float(np.degrees(np.linalg.norm(vectors, axis=1).max()))
    ratio = float(singular[1] / max(singular[0], 1e-12))
    if maximum < 15 or ratio < 0.1:
        raise ValueError(f"Degenerate poses: rotation span={maximum:.1f} deg, second-axis ratio={ratio:.3f}. "
                         "Use substantial rotations about at least two nonparallel axes.")
    return {"max_relative_rotation_deg": maximum, "rotation_axis_singular_values": singular.tolist(),
            "second_axis_ratio": ratio,
            "translation_span_m": np.ptp([t[:3, 3] for t in poses], axis=0).tolist()}


def evaluate(samples, x, y):
    """X = T_base_camera, Y = T_ee_target; compare X*C_i with G_i*Y."""
    rows = []
    for s in samples:
        g, c = checked(s["T_base_ee"]), checked(s["T_camera_target"])
        mm, deg = errors(x @ c, g @ y)
        row = {"id": s["id"], "translation_mm": mm, "rotation_deg": deg}
        if "object_points" in s:
            predicted = inverse(x) @ g @ y
            obj = np.asarray(s["object_points"], dtype=float)
            if np.any((obj @ predicted[:3, :3].T + predicted[:3, 3])[:, 2] <= 0):
                raise ValueError("Predicted target behind camera")
            uv, _ = cv2.projectPoints(obj, cv2.Rodrigues(predicted[:3, :3])[0], predicted[:3, 3],
                                      np.asarray(s["K"], dtype=float), np.asarray(s["D"], dtype=float))
            row["predicted_reprojection_rms_px"] = float(np.sqrt(np.mean(np.sum(
                (uv.reshape(-1, 2) - np.asarray(s["image_points"])) ** 2, axis=1))))
        rows.append(row)
    summary = {}
    for field in ["translation_mm", "rotation_deg", "predicted_reprojection_rms_px"]:
        values = np.array([r[field] for r in rows if field in r])
        if values.size:
            summary[field] = {"rms": float(np.sqrt(np.mean(values ** 2))),
                              "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                              "max": float(values.max())}
    return {"count": len(rows), "summary": summary, "samples": rows}


def _reprojection_observations(samples):
    observations = []
    for s in samples:
        try:
            g = checked(s["T_base_ee"])
            obj = np.asarray(s["object_points"], dtype=float)
            uv = np.asarray(s["image_points"], dtype=float)
            k, d = np.asarray(s["K"], dtype=float), np.asarray(s["D"], dtype=float)
        except KeyError as exc:
            raise ValueError("Reprojection refinement needs object_points, image_points, K and D for every sample") from exc
        if (obj.ndim != 2 or obj.shape[1] != 3 or len(obj) < 4 or uv.shape != (len(obj), 2)
                or k.shape != (3, 3) or d.size not in (4, 5, 8, 12, 14)
                or not all(np.isfinite(v).all() for v in (obj, uv, k, d))):
            raise ValueError("Invalid reprojection observations")
        observations.append((g, obj, uv, k, d))
    return observations


def _refine_reprojection(samples, x, y):
    """Fit X/Y to training image corners with fixed K/D and metric target points.

    PnP poses provide the initial estimate only. No intrinsic, robot pose or
    per-view target pose is adjusted to improve the fit.
    """
    observations = _reprojection_observations(samples)

    def residual(z):
        xi, ty = inverse(unpack(z[:6])), unpack(z[6:])
        out = []
        for g, obj, uv, k, d in observations:
            c = xi @ g @ ty
            projected, _ = cv2.projectPoints(obj, cv2.Rodrigues(c[:3, :3])[0], c[:3, 3], k, d)
            out.extend((projected.reshape(-1, 2) - uv).ravel())
            # Fixed-size barrier against crossing the camera plane during fitting.
            depth = (obj @ c[:3, :3].T + c[:3, 3])[:, 2]
            out.extend(np.maximum(0., 0.001 - depth) * 1e6)
        return np.asarray(out)

    evaluate(samples, x, y)  # All initial corners must be in front of the camera.
    opt = least_squares(residual, np.r_[pack(x), pack(y)], loss="soft_l1", f_scale=1.,
                        max_nfev=500, xtol=1e-11, ftol=1e-11, gtol=1e-11)
    if not opt.success:
        raise ValueError(f"Reprojection refinement failed: {opt.message}")
    x, y = unpack(opt.x[:6]), unpack(opt.x[6:])
    evaluate(samples, x, y)  # Reject a nonphysical final solution.
    sv = np.linalg.svd(opt.jac, compute_uv=False)
    return x, y, {"enabled": True, "success": bool(opt.success), "objective": "image_corner_reprojection",
                  "loss": "soft_l1", "f_scale_px": 1., "cost": float(opt.cost), "nfev": int(opt.nfev),
                  "jacobian_condition": float(sv[0] / max(sv[-1], 1e-15)),
                  "fixed": ["camera_K_D", "metric_target_geometry", "measured_robot_poses"],
                  "sample_count": len(samples), "samples_removed": 0}


def solve(samples, refine=True, refinement="closure"):
    if refinement not in ("closure", "reprojection"):
        raise ValueError("Unknown refinement objective")
    if not refine and refinement != "closure":
        raise ValueError("--no-refine cannot be combined with reprojection refinement")
    if refinement == "reprojection":
        _reprojection_observations(samples)
    if len(samples) < 10:
        raise ValueError("Need at least 10 training poses; recommend 25-40")
    if len({s["id"] for s in samples}) != len(samples):
        raise ValueError("Duplicate sample IDs")
    g = [checked(s["T_base_ee"]) for s in samples]
    c = [checked(s["T_camera_target"]) for s in samples]
    diagnostics = diversity(g)
    if not hasattr(cv2, "calibrateHandEye"):
        raise RuntimeError("cv2.calibrateHandEye unavailable. Use the audited rebotarm environment (OpenCV 4.10).")
    gi = [inverse(t) for t in g]
    candidates, failures = [], {}
    for name in ["PARK", "TSAI", "HORAUD"]:
        try:
            r, t = cv2.calibrateHandEye([t[:3, :3] for t in gi], [t[:3, 3] for t in gi],
                                      [t[:3, :3] for t in c], [t[:3, 3] for t in c],
                                      method=getattr(cv2, "CALIB_HAND_EYE_" + name))
            x = checked(transform(r, t))
            y = average([inverse(a) @ x @ b for a, b in zip(g, c)])
            metrics = evaluate(samples, x, y)
            score = metrics["summary"]["translation_mm"]["rms"] / 3 + metrics["summary"]["rotation_deg"]["rms"] / 0.5
            candidates.append((score, name, x, y, metrics))
        except (ValueError, cv2.error) as exc:
            failures[name] = str(exc)
    if not candidates:
        raise ValueError(f"All solvers failed: {failures}")
    _, method, x, y, _ = min(candidates, key=lambda v: v[0])
    optimization = {"enabled": False}
    if refine:
        def residual(z):
            tx, ty = unpack(z[:6]), unpack(z[6:])
            out = []
            for a, b in zip(g, c):
                lhs, rhs = tx @ b, a @ ty
                out.extend((lhs[:3, 3] - rhs[:3, 3]) / 0.003)
                out.extend(Rotation.from_matrix(rhs[:3, :3].T @ lhs[:3, :3]).as_rotvec() / np.radians(0.5))
            return np.asarray(out)
        opt = least_squares(residual, np.r_[pack(x), pack(y)], loss="soft_l1", max_nfev=300,
                            xtol=1e-11, ftol=1e-11, gtol=1e-11)
        if not opt.success:
            raise ValueError(f"Refinement failed: {opt.message}")
        x, y = unpack(opt.x[:6]), unpack(opt.x[6:])
        sv = np.linalg.svd(opt.jac, compute_uv=False)
        optimization = {"enabled": True, "success": bool(opt.success), "cost": float(opt.cost),
                        "jacobian_condition": float(sv[0] / max(sv[-1], 1e-15))}
        if refinement == "reprojection":
            x, y, pixel_optimization = _refine_reprojection(samples, x, y)
            pixel_optimization["closure_initialization"] = optimization
            optimization = pixel_optimization
    training = evaluate(samples, x, y)
    warnings = []
    if diagnostics["second_axis_ratio"] < 0.25:
        warnings.append("Weak rotation diversity; add rotations about another axis")
    for row in training["samples"]:
        if row["translation_mm"] > 10 or row["rotation_deg"] > 2:
            warnings.append(f"Inspect sample {row['id']}: large closure error; no sample automatically removed")
    return {"schema_version": 1, "convention": "T_A_B maps coordinates from B to A; translation in meters",
            "T_base_camera": x.tolist(), "T_camera_base": inverse(x).tolist(), "T_ee_target": y.tolist(),
            "initial_method": method, "optimization": optimization, "diversity": diagnostics,
            "method_comparison": {v[1]: v[4]["summary"] for v in candidates}, "method_failures": failures,
            "training": training, "training_ids": [s["id"] for s in samples], "warnings": warnings,
            "acceptance": "UNVALIDATED: independent validation and physical accuracy check required"}

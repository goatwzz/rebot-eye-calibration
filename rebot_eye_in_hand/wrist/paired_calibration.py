"""Offline wrist reprojection fit with joint2..5 effective offsets.

Promoted from the solver used for the recorded 74-pose experiment. Uses the
existing Pinocchio FK, OpenCV geometry and SciPy optimizer; no hardware I/O.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from .auto_calibration import read_dataset, geometry
from .validation import evaluate, load_result, apply_result_correction
from eyehand.kinematics import CorrectedKinematics


def fit_dataset(path, output, validation_path=None):
    path, data = read_dataset(path)
    samples = data['samples']
    if data.get('synthetic') is not False or data['acquisition']['split'] != 'train' or len(samples) < 15:
        raise ValueError('Need at least 15 real training poses; keep validation separate')
    if len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Duplicate training sample IDs')
    fk = CorrectedKinematics(data['robot'])
    obj, k, dist = np.array(samples[0]['object_points']), np.array(data['K']), np.array(data['D'])
    qs, uvs = [], []
    for s in samples:
        if s['object_points'] != samples[0]['object_points'] or s['K'] != data['K'] or s['D'] != data['D']:
            raise ValueError('Target geometry or intrinsics changed within the dataset')
        q, uv = np.array(s['robot_reads'][-1]['q_rad']), np.array(s['image_points'])
        if q.shape != (6,) or uv.shape != (len(obj), 2) or not np.isfinite(np.r_[q, uv.ravel()]).all():
            raise ValueError('Invalid raw joint/corner observations')
        qs.append(q); uvs.append(uv)
    validation = None
    if validation_path:
        _, validation = read_dataset(validation_path)
        if (validation.get('synthetic') is not False or validation['acquisition']['split'] != 'validation'
                or validation['dataset_id'] == data['dataset_id']
                or {s['id'] for s in samples} & {s['id'] for s in validation['samples']}):
            raise ValueError('Validation must be independently acquired, without reused training IDs')
        for key in ('target', 'K', 'D', 'image_size'):
            if data[key] != validation[key]:
                raise ValueError('Training/validation mismatch: '+key)
        if (data['camera']['serial'] != validation['camera']['serial'] or
                data['robot']['urdf_sha256'] != validation['robot']['urdf_sha256']):
            raise ValueError('Training/validation camera or model mismatch')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output exists; choose a new candidate directory')
    virtual = [dict(s, T_base_ee=geometry.inverse(fk.fk(q)).tolist()) for s, q in zip(samples, qs)]
    seed = geometry.solve(virtual, refinement='reprojection')
    z0 = np.r_[geometry.pack(np.array(seed['T_base_camera'])),
               geometry.pack(np.array(seed['T_ee_target'])), np.zeros(4)]

    def residual(z):
        xi, y = geometry.inverse(geometry.unpack(z[:6])), geometry.unpack(z[6:12])
        offsets = np.r_[0., z[12:], 0.]
        values = []
        for q, uv in zip(qs, uvs):
            c = xi @ geometry.inverse(fk.fk(q, offsets)) @ y
            projected, _ = cv2.projectPoints(obj, cv2.Rodrigues(c[:3, :3])[0], c[:3, 3], k, dist)
            values.extend((projected.reshape(-1, 2)-uv).ravel())
            depths = (obj @ c[:3, :3].T+c[:3, 3])[:, 2]
            values.extend(np.maximum(0., .001-depths)*1e6)
        return values

    lo = np.r_[np.full(12, -np.inf), np.full(4, -np.radians(5.))]
    opt = least_squares(residual, z0, bounds=(lo, -lo), loss='soft_l1', f_scale=1.,
                        max_nfev=200, xtol=1e-11, ftol=1e-11, gtol=1e-11)
    if not opt.success or np.any(np.abs(opt.x[12:]) >= np.radians(4.99)):
        raise ValueError('Fit failed or reached correction bounds: '+str(opt.message))
    x, y, offsets = geometry.unpack(opt.x[:6]), geometry.unpack(opt.x[6:12]), np.r_[0., opt.x[12:], 0.]
    norms = np.linalg.norm(opt.jac, axis=0)
    sv = np.linalg.svd(opt.jac/np.maximum(norms, 1e-15), compute_uv=False)
    rank = int(np.count_nonzero(sv > sv[0]*1e-6))
    if rank < 16:
        raise ValueError('Insufficient observable motion; add independent rotations')
    corrected = copy.deepcopy(data)
    for s, q in zip(corrected['samples'], qs):
        s['T_base_ee'] = fk.fk(q, offsets).tolist()
    training = evaluate(corrected, x, np.eye(4), geometry)
    output.mkdir(parents=True, exist_ok=False)
    candidate = output/'hand_eye_candidate.npz'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with candidate.open('xb') as handle:
        np.savez(handle, T_result=x, mode=np.array(['eye_in_hand']), n_samples=np.array([len(samples)]),
                 method=np.array(['PARK+robust_reprojection+effective_joint_offsets_2_5']),
                 camera_serial=np.array([data['camera']['serial']]),
                 urdf_sha256=np.array([data['robot']['urdf_sha256']]), joint_offsets_rad=offsets,
                 training_dataset_ids=np.array([data['dataset_id']]), training_dataset_sha256=np.array([digest]))
    _, info = load_result(candidate, geometry, allow_joint_offsets=True)
    validation_report = None
    if validation is not None:
        validation_report = evaluate(apply_result_correction(validation, info), x, np.eye(4), geometry)
    report = dict(T_gripper_camera=x.tolist(), T_base_target=y.tolist(),
                  joint_offsets_rad=offsets.tolist(), joint_offsets_deg=np.degrees(offsets).tolist(),
                  training=training, independent_validation=validation_report,
                  dataset_sha256=digest, training_count=len(samples),
                  optimization=dict(success=True, cost=float(opt.cost), nfev=int(opt.nfev),
                                    normalized_rank=rank, normalized_condition=float(sv[0]/sv[-1])),
                  physical_accuracy_verified=False, active_calibration_changed=False)
    correction = dict(type='effective_joint_angle_offsets',
                      joint_names=[f'joint{i}' for i in range(1, 7)],
                      joint_offsets_rad=offsets.tolist(), joint_offsets_deg=np.degrees(offsets).tolist(),
                      original_urdf_sha256=data['robot']['urdf_sha256'],
                      convention='q_model=q_raw+joint_offsets_rad; motor commands remain raw',
                      base_frame='base_link', ee_frame='gripper_end')
    for name, content in [('report.json', report), ('joint_correction.json', correction),
                          ('intrinsics.json', {key: data[key] for key in ('camera', 'image_size', 'K', 'D')})]:
        (output/name).write_text(json.dumps(content, ensure_ascii=False, indent=2)+'\n')
    print(candidate)
    print(json.dumps(dict(training=training['summary'], independent_validation=None if
                         validation_report is None else validation_report['summary']), indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--validation-dataset')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    fit_dataset(args.dataset, args.output, args.validation_dataset)

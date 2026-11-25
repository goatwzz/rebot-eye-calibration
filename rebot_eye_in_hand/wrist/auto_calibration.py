"""Wrist entry/configuration and OpenCV hand-eye solving; shared SDK owns motion."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

from .backend import dependencies
ROOT=Path(__file__).resolve().parents[1]
_,geometry=dependencies(ROOT.parent/'rebot_eye_to_hand')
from eyehand.auto_collection_runtime import main as collect_main
from eyehand.auto_collection import coverage
from eyehand.cli import save
from .validation import evaluate,load_result


def read_dataset(path):
    path=Path(path).resolve()
    if path.is_dir():path/='dataset.json'
    data=json.loads(path.read_text())
    if data.get('acquisition',{}).get('calibration_mode')!='eye_in_hand':
        raise ValueError('Use fixed-target wrist automatic data, not external-camera data')
    if not data['camera'].get('serial'):
        raise ValueError('Missing wrist camera identity')
    if (data['target'].get('marker_m') not in (.06,.1) or
            data['target']!=dict(type='aruco',dictionary='DICT_4X4_50',id=0,marker_m=data['target'].get('marker_m'))):
        raise ValueError('Expected measured 60/100 mm DICT_4X4_50 ID 0')
    if data['robot']['base_frame']!='base_link' or data['robot']['ee_frame']!='gripper_end':
        raise ValueError('Unexpected base/end frames')
    return path,data


def fit_candidate(dataset_path,output,validation_path=None):
    """G X C=Y becomes X C=inv(G) Y: reuse audited ETH OpenCV/refinement code.

    Here X=T_gripper_camera and Y=T_base_target. Only robot transform direction
    changes for the solver; original raw-angle observations remain unchanged.
    """
    path,data=read_dataset(dataset_path)
    if data.get('synthetic') is not False:raise ValueError('Candidate requires real acquisition data')
    if data.get('acquisition',{}).get('split')!='train':
        raise ValueError('Fit requires train split; keep independent validation out of the solver')
    if len(data['samples'])<15:raise ValueError('Need at least 15 wrist poses; prefer 60 or more')
    virtual=[dict(s,T_base_ee=geometry.inverse(geometry.checked(s['T_base_ee'])).tolist())
             for s in data['samples']]
    fitted=geometry.solve(virtual,refinement='reprojection')
    x=geometry.checked(fitted['T_base_camera'])
    training=evaluate(data,x,np.eye(4),geometry)
    validation=None
    if validation_path:
        vpath,v=read_dataset(validation_path)
        if v.get('synthetic') is not False:raise ValueError('Validation must contain real samples')
        if (v['camera'].get('serial')!=data['camera'].get('serial') or v['robot']['urdf_sha256']!=data['robot']['urdf_sha256'] or v['image_size']!=data['image_size']
                or not np.allclose(v['K'],data['K'],rtol=0,atol=1e-6)
                or not np.allclose(v['D'],data['D'],rtol=0,atol=1e-9)):
            raise ValueError('Validation model or camera intrinsics differ')
        if (v['dataset_id']==data['dataset_id'] or
                {s['id'] for s in v['samples']} & {s['id'] for s in data['samples']} or
                v.get('acquisition',{}).get('split')!='validation'):
            raise ValueError('Validation must be independently acquired, not reused training poses')
        validation=evaluate(v,x,np.eye(4),geometry)
        validation.update(dataset=str(vpath),dataset_sha256=hashlib.sha256(vpath.read_bytes()).hexdigest())
    report=dict(mode='eye_in_hand',convention='T_A_B maps B coordinates to A; metres',
        T_gripper_camera=x.tolist(),T_camera_gripper=geometry.inverse(x).tolist(),
        T_base_target=fitted['T_ee_target'],joint_offsets_rad=[0.]*6,
        method=fitted['initial_method']+'+robust_corner_reprojection',
        optimization=fitted['optimization'],training=training,validation=validation,
        method_comparison=fitted['method_comparison'],warnings=fitted['warnings'],
        coverage=coverage(data),training_dataset=str(path),
        training_dataset_id=data['dataset_id'],training_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        camera=data['camera'],target=data['target'],robot=data['robot'],
        previous_calibration_used=False,active_calibration_changed=False,
        physical_accuracy_verified=False,
        acceptance='INDEPENDENT_VALIDATION_REQUIRED' if validation is None else validation['status'])
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    candidate=output/'hand_eye_candidate.npz'
    with candidate.open('xb') as f:
        np.savez(f,T_result=x,mode=np.array(['eye_in_hand']),n_samples=np.array([len(data['samples'])]),
            method=np.array([report['method']]),camera_serial=np.array([data['camera']['serial']]),
            urdf_sha256=np.array([data['robot']['urdf_sha256']]),joint_offsets_rad=np.zeros(6),
            training_dataset_ids=np.array([data['dataset_id']]),
            training_dataset_sha256=np.array([report['training_sha256']]))
    # Existing wrist demo/validator must be able to read the produced NPZ.
    recovered,_=load_result(candidate,geometry)
    if not np.allclose(recovered,x,rtol=0,atol=1e-12):
        raise RuntimeError('Saved candidate transform did not round-trip')
    save(output/'report.json',report)
    save(output/'intrinsics.json',dict(camera=data['camera'],image_size=data['image_size'],K=data['K'],D=data['D']))
    print('[拟合] '+str(candidate),flush=True)
    for label,metrics in [('训练',training),('独立验证',validation)]:
        if metrics is None:continue
        summary=metrics['summary']
        t,r,u=(summary[k] for k in ('translation_mm','rotation_deg','predicted_reprojection_rms_px'))
        print(f"[{label}] 平移 RMS/P95 {t['rms']:.2f}/{t['p95']:.2f} mm；"
              f"旋转 {r['rms']:.2f}/{r['p95']:.2f}°；重投影 {u['rms']:.2f}/{u['p95']:.2f} px。",flush=True)
    print('[结果] 名义运动学基线候选已另存，当前标定未覆盖；外参＋关节角修正需另行拟合，并用独立验证验收。',flush=True)
    return report


def after_collection(output,settings):
    if settings['split']!='train':return
    path=output/'dataset.json'
    if not path.exists():return
    data=json.loads(path.read_text())
    if len(data['samples'])<15:
        print('[拟合未执行] 有效样本少于15组；已采数据仍然保留。',flush=True);return
    try:fit_candidate(path,output/'fit')
    except (ValueError,RuntimeError) as exc:
        print('[拟合未完成] '+str(exc)+'；原始数据已保存，可离线重试。',flush=True)
        save(output/'fit_error.json',dict(error=str(exc)))


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='fit':
        parser=argparse.ArgumentParser(description='Offline wrist hand-eye fit and optional independent validation')
        parser.add_argument('--dataset',required=True)
        parser.add_argument('--validation-dataset')
        parser.add_argument('--output',required=True)
        args=parser.parse_args(argv[1:])
        fit_candidate(args.dataset,args.output,args.validation_dataset)
        return 0
    return collect_main(argv,mount='wrist',project_root=ROOT,
        config_path=ROOT/'config/auto_collection.json',after_collection=after_collection)

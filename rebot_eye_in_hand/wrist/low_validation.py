"""Frozen-result, low-pose validation on the existing SDK collection runtime."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import yaml

from .backend import dependencies
ROOT=Path(__file__).resolve().parents[1]
_,geometry=dependencies(ROOT.parent/'rebot_eye_to_hand')
from .validation import load_result,apply_result_correction,evaluate,stats
from .auto_calibration import read_dataset
from eyehand.auto_collection import load_settings as shared_settings,CollectionPlanner
from eyehand.auto_collection_runtime import main as collect_main
from eyehand.cli import save


def load_settings(path,split='validation',*,mount='wrist'):
    if split!='validation' or mount!='wrist':raise ValueError('This profile is wrist validation only, never training')
    settings,cfg,rows=shared_settings(path,split,mount=mount)
    if settings['target']['marker_m']!=.06:raise ValueError('Low-pose validation uses the existing 60mm board')
    defaults=yaml.safe_load((Path(cfg['_path']).parent/'default.yaml').read_text())
    result=(ROOT/defaults['paths']['result']).resolve()
    _,info=load_result(result,geometry,allow_joint_offsets=True)
    cfg.update(validation_result_source=str(result),validation_result_sha256=info['sha256'],
        collection_planning_offsets_rad=info['joint_offsets_rad'],
        validation_intrinsics_source=str((Path(cfg['_path']).parent/cfg['intrinsics']).resolve()))
    return settings,cfg,rows


class ValidationPlanner(CollectionPlanner):
    def __init__(self,cfg):
        self.result_bytes=Path(cfg['validation_result_source']).read_bytes()
        if hashlib.sha256(self.result_bytes).hexdigest()!=cfg['validation_result_sha256']:
            raise ValueError('Validation result changed during initialization')
        super().__init__(cfg)
        self.intrinsics_bytes=Path(cfg['validation_intrinsics_source']).read_bytes()
        self.intrinsics=json.loads(self.intrinsics_bytes)
        self.robot_metadata.update(validation_calibration=dict(source=cfg['validation_result_source'],
            sha256=cfg['validation_result_sha256'],snapshot='hand_eye_under_test.npz',
            planning_joint_offsets_rad=self.offsets.tolist(),recorded_fk_is_nominal=True),
            validation_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.robot_metadata['gravity_parameters']['scope']='Paired corrected-angle SDK gravity feedforward; saved capture FK remains nominal RAW angles'

    def snapshot_validation_result(self,output):
        for name,content in [('hand_eye_under_test.npz',self.result_bytes),
                             ('intrinsics_under_test.json',self.intrinsics_bytes)]:
            with (output/name).open('xb') as f:f.write(content)

    def check_validation_camera(self,camera):
        intr=self.intrinsics
        if (camera.metadata['serial']!=intr['camera']['serial']
                or [camera.width,camera.height]!=intr['image_size']
                or not np.allclose(camera.k,intr['K'],rtol=0,atol=1e-6)
                or not np.allclose(camera.d,intr['D'],rtol=0,atol=1e-9)):
            raise ValueError('Live RGB identity/profile/intrinsics differ from the frozen calibration')


def report(directory,output=None):
    directory=Path(directory).resolve()
    path,data=read_dataset(directory)
    if data.get('synthetic') is not False:raise ValueError('Require real independent acquisition')
    if data.get('acquisition',{}).get('split')!='validation':raise ValueError('Require validation split')
    x,info=load_result(directory/'hand_eye_under_test.npz',geometry,allow_joint_offsets=True)
    session=json.loads((directory/'session.json').read_text())
    if info['sha256']!=session['control_config']['validation_result_sha256']:
        raise ValueError('Frozen candidate changed since collection')
    if info['urdf_sha256']!=data['robot']['urdf_sha256'] or info['camera_serial']!=data['camera']['serial']:
        raise ValueError('Frozen candidate camera/model mismatch')
    if (data['dataset_id'] in info.get('training_dataset_ids',[]) or
            hashlib.sha256(path.read_bytes()).hexdigest() in info.get('training_dataset_sha256',[])):
        raise ValueError('Candidate training data cannot be its independent validation')
    output=Path(output).resolve() if output else directory/'independent_validation_report.json'
    if output.exists():raise FileExistsError('Report exists; choose a new --output')
    if not data['samples']:
        result=dict(status='NO_SAMPLES',count=0,hand_eye_refitted=False,physical_accuracy_verified=False)
    else:
        corrected=apply_result_correction(data,info)
        result=evaluate(corrected,x,np.eye(4),geometry)
        target=np.array(result['mean_T_base_target_without_compensation'])
        residuals=np.array([(np.array(s['T_base_ee'])@x@np.array(s['T_camera_target']))[:3,3]-target[:3,3]
                           for s in corrected['samples']])*1000
        angles=np.degrees(np.arcsin(np.clip([-np.array(s['T_base_ee'])[2,0] for s in corrected['samples']],-1,1)))
        heights=np.array([s['T_base_ee'][2][3]*1000 for s in corrected['samples']])
        result.update(axis_errors_mm={axis:dict(stats(abs(residuals[:,i])),
            signed_mean=float(residuals[:,i].mean()),signed_min=float(residuals[:,i].min()),signed_max=float(residuals[:,i].max()))
            for i,axis in enumerate('xyz')},actual_tcp_z_range_mm=[float(min(heights)),float(max(heights))],
            actual_down_angle_range_deg=[float(min(angles)),float(max(angles))])
        masks={group:np.array([s['requested_pose']['group']==group for s in data['samples']])
               for group in sorted({s['requested_pose']['group'] for s in data['samples']})}
        masks.update({f'down_{lo}_{hi}_deg':(angles>=lo)&(angles<hi) for lo,hi in [(55,70),(70,80),(80,88),(88,91)]})
        result['z_by_pose_group']={name:dict(count=int(mask.sum()),**stats(abs(residuals[mask,2])),
            signed_mean=float(residuals[mask,2].mean())) for name,mask in masks.items() if mask.any()}
        for row,r,height,angle in zip(result['samples'],residuals,heights,angles):
            row.update(target_position_delta_xyz_mm=r.tolist(),actual_tcp_z_mm=float(height),
                       tool_down_angle_deg=float(angle))
    result.update(dataset=str(path),dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        calibration=info,validation_type='new independent low-pose acquisition, fixed candidate and joint corrections',
        target=data['target'],refit_performed=False,active_calibration_changed=False,
        interpretation='Fixed-board repeatability across poses, not physical fingertip/absolute-Z ground truth; board mean estimated from validation observations')
    save(output,result)
    if 'summary' in result:
        t=result['summary']['translation_mm'];z=result['axis_errors_mm']['z']
        print(f"[验证] {result['count']}组；XYZ RMS/P95 {t['rms']:.2f}/{t['p95']:.2f} mm；Z RMS/P95 {z['rms']:.2f}/{z['p95']:.2f} mm；{result['status']}",flush=True)
    print('[报告] '+str(output),flush=True)
    return result


def after_collection(output,settings):
    if (output/'dataset.json').exists():report(output)


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='report':
        parser=argparse.ArgumentParser(description='Evaluate frozen wrist candidate; never fit')
        parser.add_argument('--dataset',required=True);parser.add_argument('--output')
        args=parser.parse_args(argv[1:]);report(args.dataset,args.output);return 0
    if not any(a=='--split' or a.startswith('--split=') for a in argv):argv+=['--split','validation']
    return collect_main(argv,mount='wrist',project_root=ROOT,
        config_path=ROOT/'config/low_pose_validation.json',settings_loader=load_settings,
        planner_type=ValidationPlanner,after_collection=after_collection)

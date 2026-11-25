"""Offline real-URDF checks, isolated from legacy SDK imports in existing tests."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pinocchio as pin
from eyehand.click_hover import HoverPlanner, load_config, ROOT, in_workspace
from eyehand.kinematics import CorrectedKinematics, measured_joints

cfg=load_config()
# This legacy regression inspects polynomial derivatives of the joint planner.
# The active SDK sampled Cartesian planner is covered by sdk_hover_checks.py.
cfg['planning_mode']='joint'
p=HoverPlanner(cfg)
data=json.loads((ROOT/'runs/final_eye_to_hand_2026-11-20/combined_dataset.json').read_text())
c=p.result['kinematic_correction']
k=CorrectedKinematics({'urdf':c['original_urdf'],'urdf_sha256':c['original_urdf_sha256'],
                       'joint_names':c['joint_names'],'base_frame':'base_link','ee_frame':'gripper_end'},c)
assert p.model.nq == p.model.nv == 6
expected_point_mass = p.gravity['payload_mass_kg'] + p.gravity['wrist_camera_mass_kg']
assert abs(sum(i.mass for i in p.gravity_model.inertias)-sum(i.mass for i in p.model.inertias)-expected_point_mass)<1e-9
count=0
refused=0
for sample in data['samples']:
    raw=measured_joints(sample)
    transform=k.fk(raw)
    expected=transform[:3,:3]@cfg['tcp_offset_m']+transform[:3,3]
    np.testing.assert_allclose(p.tcp(raw),expected,atol=1e-12)
    target=p.tcp(raw)+[.005,0,.005]
    if not in_workspace(expected,cfg) or not in_workspace(target,cfg):
        # Historical calibration allowed fingers below the demo's 30mm floor.
        # Correcting the TCP must not silently relax that motion workspace.
        try:p.plan(raw,target)
        except ValueError as exc:assert 'workspace' in str(exc)
        else:raise AssertionError('Out-of-workspace historical pose accepted')
        refused+=1
        continue
    plan=p.plan(raw,target)
    assert np.linalg.norm(p.tcp(plan.goal)-target)<.0002
    qs,vs=plan.sample(0)
    np.testing.assert_allclose(qs,raw,atol=1e-12)
    np.testing.assert_allclose(vs,0,atol=1e-12)
    grid=np.linspace(0,1,501)
    qs=plan.curve(grid)
    assert np.max(np.abs(plan.curve(grid,1)/plan.duration))<=cfg['max_joint_speed_rad_s']*1.001
    assert np.max(np.abs(plan.curve(grid,2)/plan.duration**2))<=cfg['max_joint_accel_rad_s2']*1.001
    assert np.all(qs>=p.raw_lower) and np.all(qs<=p.raw_upper)
    positions=np.array([p.tcp(q) for q in qs])
    cart_speed=np.linalg.norm(np.diff(positions,axis=0),axis=1)/(plan.duration/500)
    assert cart_speed.max()<=cfg['max_tcp_speed_m_s']*1.001
    q_model=plan.goal+p.offsets
    d=p.model.createData()
    pin.framesForwardKinematics(p.model,d,q_model)
    np.testing.assert_allclose(d.oMf[p.tcp_id].translation,target,atol=.0002)
    count+=1
for target in ([5,5,5],p.tcp(raw)+[.7,0,0],[float('nan'),0,0]):
    try:p.plan(raw,target)
    except ValueError:pass
    else:raise AssertionError('Invalid target accepted')
assert count+refused==85
print(f'85 FK checks; {count} valid nearby plans, {refused} workspace refusals; raw/corrected angle mapping, trajectory bounds, mass checks passed')

# Exercise startup using fake motors and the official callback lifecycle, without CAN.
from types import SimpleNamespace
import time
from unittest.mock import Mock, patch
from eyehand.hover_control import HoverController
import eyehand.native_feedback as feedback_module
import eyehand.hover_control as control_module
names=[f'joint{i}' for i in range(1,7)]+['gripper']
q0=np.r_[raw,.25]
events=[]
motors={}
for i,name in enumerate(names):
    m=Mock()
    m.robstride_get_param_f32.return_value=float(q0[i])
    m.send_mit.side_effect=lambda *args,n=name:events.append(('send',n,args[0]))
    m.enable.side_effect=lambda n=name:events.append(('enable',n))
    motors[name]=m
arm=SimpleNamespace(joint_names=names[:6],mode_mit=Mock(return_value=True),send_mit=Mock(),
                    get_positions=Mock(return_value=q0[:6].copy()))
gripper=SimpleNamespace(joint_names=['gripper'],mode_mit=Mock(return_value=True),send_mit=Mock(),
                        _mit_kp=np.array([1.]),_mit_kd=np.array([.1]),
                        get_positions=Mock(return_value=q0[6:].copy()))
robot=SimpleNamespace(connect=Mock(),arm=arm,gripper=gripper,groups={'gripper':gripper},_motor_map=motors,
                      _all_joints=[SimpleNamespace(name=n,vendor='robstride',motor_id=i+1,feedback_id=0xFD,model='rs-06' if i<3 else 'rs-00')
                                   for i,n in enumerate(names)],start_control_loop=Mock(),
                      stop_control_loop=Mock(),disconnect=Mock(),estop=Mock())
feedback=Mock()
feedback.snapshot.return_value=SimpleNamespace(positions=q0.copy())
# Timestamped native snapshot fixture; repeated lifecycle reads use current fake status.
def fresh_snapshot(max_age):
    feedback.times=np.full(7,time.perf_counter())
    return feedback.snapshot.return_value
feedback.snapshot.side_effect=fresh_snapshot
feedback.sequences=np.ones(7,dtype=int)
feedback.intervals=[[] for _ in range(7)]
with patch.object(feedback_module,'NativeFeedback') as factory, patch.object(control_module.time,'sleep'):
    factory.supported.return_value=True
    factory.return_value=feedback
    controller=HoverController(p,robot)
    controller.start()
    assert controller.enabled
    np.testing.assert_allclose(controller.target,raw)
    np.testing.assert_allclose(controller.gripper,[.25])
    for i,name in enumerate(names):
        assert events[3*i][0:2]==('send',name)
        assert abs(events[3*i][2]-q0[i])<1e-12
        assert events[3*i+1]==('enable',name)
        assert events[3*i+2][0:2]==('send',name)
    robot.start_control_loop.assert_called_once()
    controller._tick(robot,.01)
    assert not controller.fault, controller.fault
    np.testing.assert_allclose(arm.send_mit.call_args.args[0],raw)
    controller.close()
    robot.disconnect.assert_called_once()
print('Mocked startup: mode -> captured current targets -> enable -> official loop; no zero target or hardware access')

# Longer task-sized goals in several directions, not only +5 mm samples.
long_cases = []
for index in (0, 20, 40, 60, 84):
    raw = measured_joints(data['samples'][index])
    for delta in ([.03,0,0],[-.03,0,0],[0,.03,0],[0,0,.03],[.05,.02,.04]):
        target=p.tcp(raw)+delta
        plan=p.plan(raw,target)
        positions=np.array([p.tcp(q) for q in plan.curve(np.linspace(0,1,501))])
        assert np.linalg.norm(p.tcp(plan.goal)-target)<.0002
        speed=np.linalg.norm(np.diff(positions,axis=0),axis=1)/(plan.duration/500)
        assert speed.max()<=cfg['max_tcp_speed_m_s']*1.001
        long_cases.append(float(np.linalg.norm(p.tcp(plan.goal)-target)*1000))
print(f'{len(long_cases)} task-sized direction checks passed, max model endpoint error {max(long_cases):.4f} mm')

# Explicit zero-payload configurations must still create valid Pinocchio gravity data.
import tempfile
with tempfile.TemporaryDirectory() as directory:
    g=json.loads(Path(cfg['gravity_config']).read_text())
    g.update(payload_mass_kg=0.,wrist_camera_mass_kg=0.)
    path=Path(directory)/'gravity.json';path.write_text(json.dumps(g))
    unloaded=HoverPlanner({**cfg,'gravity_config':str(path)})
    assert np.isfinite(unloaded.gravity_torque(raw)).all()
    original=json.loads(Path(cfg['result']).read_text())
    for update in ('camera','transform','synthetic'):
        altered=json.loads(json.dumps(original))
        if update=='camera':altered['provenance']['camera']['serial']='wrong-device'
        if update=='transform':altered['T_base_camera'][0][0]=2.
        if update=='synthetic':altered['provenance']['synthetic']=True
        path=Path(directory)/f'{update}.json';path.write_text(json.dumps(altered))
        try:HoverPlanner({**cfg,'result':str(path)})
        except ValueError:pass
        else:raise AssertionError('Invalid calibration accepted')
try:p.plan(raw,p.tcp(raw)+[.01,0,0],cancelled=lambda:True)
except InterruptedError:pass
else:raise AssertionError('Cancelled planner ran')

# Camera may fail during mode setup: the last pre-enable hook must block every motor.
for motor in motors.values():motor.reset_mock()
with patch.object(feedback_module,'NativeFeedback') as factory, patch.object(control_module.time,'sleep'):
    factory.supported.return_value=True;factory.return_value=feedback
    blocked=HoverController(p,robot,before_enable=Mock(side_effect=RuntimeError('camera expired during setup')))
    try:blocked.start()
    except RuntimeError as exc:assert 'camera expired' in str(exc)
    else:raise AssertionError('Camera failure did not block enable')
    for motor in motors.values():motor.enable.assert_not_called()
    blocked.close()

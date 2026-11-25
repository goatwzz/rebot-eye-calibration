"""Actual calibrated model + mocked controller: zero -> hover -> raw zero, no hardware."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from eyehand.click_hover import HoverPlanner,load_config,in_workspace,ROOT
from eyehand.kinematics import measured_joints
from eyehand.hover_control import HoverController
from scripts.click_hover_eth import offline_check
from eyehand.table_reference import effective_hover_m

cfg=load_config();cfg['planning_mode']='joint'
# Keep the historical reachability suite for the retained joint planner;
# active sampled Cartesian motion has separate public-SDK comparisons.
p=HoverPlanner(cfg)
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'report.json'
    with contextlib.redirect_stdout(io.StringIO()):status=offline_check(p,path)
    report=json.loads(path.read_text())
    # A calibration pose is allowed to be outside this motion demo's workspace.
    # The corrected fingertip reveals one such pose; require refusal, not a wider
    # workspace or ignoring a planner failure.
    dataset=json.loads((ROOT/'runs/final_eye_to_hand_2026-11-20/combined_dataset.json').read_text())
    by_id={s['id']:s for s in dataset['samples']}
    refused=[case for case in report['cases'] if not case['success']]
    assert len(refused)==1 and status==1
    for case in refused:
        point=p.tcp(measured_joints(by_id[case['id']]))
        assert not in_workspace(point,p.cfg)
        assert 'workspace' in case['reason']
    assert report['height_reference']=='table'
    assert len(report['table_height_cases'])==3
    assert all(c['success'] for c in report['table_height_cases'])
    assert all(abs(c['target_base_m'][2]-report['synthetic_table_z_m']-effective_hover_m(p.cfg))<1e-12
               for c in report['table_height_cases'])
    assert len(report['zero_to_hover_and_home'])==85
    assert all(c['success'] for c in report['zero_to_hover_and_home'])
    assert all(c['home_goal_raw_rad']==[0.]*6 for c in report['zero_to_hover_and_home'])

# SDK safe_home minimum-jerk targets, timing and zero MIT velocity reference.
q=np.array([.3,1.1,.5,.4,.2,-.2])
plan=p.plan_home(q)
np.testing.assert_array_equal(plan.goal,np.zeros(6))
np.testing.assert_allclose(plan.sample(0)[0],q)
np.testing.assert_allclose(plan.sample(0)[1],0,atol=1e-12)
np.testing.assert_array_equal(plan.sample(plan.duration)[0],np.zeros(6))
np.testing.assert_array_equal(plan.sample(plan.duration)[1],np.zeros(6))
assert not plan.mit_velocity_feedforward
assert plan.duration==2*np.abs(q).max()/p.cfg['transit']['home_max_vel_rad_s']*p.cfg['transit']['time_scale']
assert len(plan.points)==int((plan.duration/p.cfg['transit']['time_scale'])*50)
# Raw zero must map through the offsets once, not to q_model=0.
np.testing.assert_allclose(plan.target,p.tcp(np.zeros(6)))

# Status feedback quantization near lower limits is tolerated, commands are clamped.
feedback=np.array([0.,-.0002,-.0003,0.,0.,0.])
np.testing.assert_array_equal(p.command_from_feedback(feedback),np.zeros(6))
assert np.isfinite(p.gravity_torque(feedback)).all()
assert p.plan_home(feedback).goal.tolist()==[0.]*6
try:p.validate_feedback([0,-.04,0,0,0,0])
except ValueError:pass
else:raise AssertionError('Truly invalid joint feedback accepted')

robot=SimpleNamespace(arm=Mock(),gripper=Mock(),_motor_map={},estop=Mock(),disconnect=Mock())
controller=HoverController(p,robot)
controller.enabled=True;controller.target=np.zeros(6);controller.gripper=np.array([.25])
controller.feedback=Mock()
controller.feedback.snapshot.return_value=SimpleNamespace(positions=np.r_[np.zeros(6),.25])
controller.plan=plan;controller.started=time.perf_counter()-plan.duration-1
controller.stable_since=time.perf_counter()-.6
controller._tick(robot,.01)
assert controller.fault is None,controller.fault
assert controller.completion['status']=='HOME_REACHED_HOLD'
assert controller.enabled
robot.disconnect.assert_not_called()
np.testing.assert_array_equal(robot.arm.send_mit.call_args.args[0],np.zeros(6))
np.testing.assert_array_equal(robot.gripper.send_mit.call_args.args[0],[.25])

# A wrist rotation can leave TCP position almost unchanged. It is not raw home.
actual=np.zeros(6);actual[5]=.1
assert np.linalg.norm(p.tcp(actual)-p.tcp(np.zeros(6)))<p.cfg['settle_tolerance_m']
controller.feedback.snapshot.return_value.positions[:6]=actual
controller.plan=p.plan_home(actual)
controller.started=time.perf_counter()-controller.plan.duration-1
controller.stable_since=time.perf_counter()-.6;controller.completion=None
controller._tick(robot,.01)
assert controller.fault is None,controller.fault
assert controller.plan is not None and controller.completion is None
controller.started=time.perf_counter()-controller.plan.duration-p.cfg['settle_timeout_s']-1
controller._tick(robot,.01)
assert controller.completion['status']=='HOME_TIMEOUT_HOLD'
assert controller.enabled and controller.plan is None
robot.disconnect.assert_not_called()
print('85 zero-to-clicked-marker-hover-and-home plans; raw-zero mapping, minimum-jerk bounds, quantization, home joint checks, gripper retention and continued hold passed')

from unittest.mock import patch
import eyehand.native_feedback as feedback_module
import eyehand.hover_control as control_module
names=[f'joint{i}' for i in range(1,7)]+['gripper']
measured=np.r_[feedback,.25]
motors={n:Mock() for n in names}
for i,n in enumerate(names):motors[n].robstride_get_param_f32.return_value=float(measured[i])
arm=SimpleNamespace(joint_names=names[:6],mode_mit=Mock(return_value=True),send_mit=Mock(),
                    get_positions=Mock(return_value=measured[:6].copy()))
gripper=SimpleNamespace(joint_names=['gripper'],mode_mit=Mock(return_value=True),send_mit=Mock(),
                        _mit_kp=np.array([50.]),_mit_kd=np.array([4.]),
                        get_positions=Mock(return_value=measured[6:].copy()))
robot=SimpleNamespace(connect=Mock(),arm=arm,gripper=gripper,groups={'gripper':gripper},_motor_map=motors,
                      _all_joints=[SimpleNamespace(name=n,vendor='robstride',motor_id=i+1,feedback_id=0xFD,
                                                   model='rs-06' if i<3 else 'rs-00') for i,n in enumerate(names)],
                      start_control_loop=Mock(),stop_control_loop=Mock(),disconnect=Mock(),estop=Mock())
fresh=Mock();fresh.snapshot.return_value=SimpleNamespace(positions=measured.copy())
# Timestamped native snapshot fixture; repeated lifecycle reads use current fake status.
def fresh_snapshot(max_age):
    fresh.times=np.full(7,time.perf_counter())
    return fresh.snapshot.return_value
fresh.snapshot.side_effect=fresh_snapshot
fresh.sequences=np.ones(7,dtype=int)
fresh.intervals=[[] for _ in range(7)]
with patch.object(feedback_module,'NativeFeedback') as factory,patch.object(control_module.time,'sleep'):
    factory.supported.return_value=True;factory.return_value=fresh
    startup=HoverController(p,robot);startup.start()
    np.testing.assert_array_equal(startup.target,np.zeros(6))
    startup._tick(robot,.01)
    assert startup.fault is None,startup.fault
    np.testing.assert_array_equal(robot.arm.send_mit.call_args.args[0],np.zeros(6))
    startup.close()
print('Near-zero quantized startup captures zero command and continues holding without a manual-drag step')

# Replay the observed zero-rest position: joint3 crossed the former -0.003rad
# application threshold. The official SDK policy accepts this feedback, while
# negative target angles are still rejected. No motor zero/calibration is edited.
observed=np.array([-.0282508526,.0084368922,-.0031105811,.0191747677,-.0579461120,.0041800975])
np.testing.assert_array_equal(p.validate_feedback(observed),observed)
assert np.isfinite(p.gravity_torque(observed)).all()
try:p.validate_raw(observed)
except ValueError:pass
else:raise AssertionError('Out-of-limit target accepted')
expected=np.clip(observed,p.raw_lower,p.raw_upper)
arm.get_positions.return_value=observed.copy()
fresh.snapshot.return_value.positions=np.r_[observed,.25]
for motor in motors.values():motor.reset_mock()
with patch.object(feedback_module,'NativeFeedback') as factory,patch.object(control_module.time,'sleep'):
    factory.supported.return_value=True;factory.return_value=fresh
    startup=HoverController(p,robot);startup.start()
    np.testing.assert_array_equal(startup.target,expected)
    startup._tick(robot,.01)
    assert startup.fault is None,startup.fault
    startup.close()
arm.get_positions.assert_called_with(fresh=True,strict=True,timeout_ms=150)
gripper.get_positions.assert_called_with(fresh=True,strict=True,timeout_ms=150)

# A real feedback violation is rejected before mode changes or motor enable.
arm.mode_mit.reset_mock();gripper.mode_mit.reset_mock()
arm.get_positions.return_value=observed.copy();arm.get_positions.return_value[2]=-.04
for motor in motors.values():motor.reset_mock()
blocked=HoverController(p,robot)
try:blocked.start()
except ValueError as exc:
    assert 'joint3=-2.2918deg' in str(exc),str(exc)
    assert 'motors not enabled' in str(exc),str(exc)
else:raise AssertionError('Official feedback violation accepted')
arm.mode_mit.assert_not_called();gripper.mode_mit.assert_not_called()
for motor in motors.values():
    motor.enable.assert_not_called();motor.send_mit.assert_not_called()
blocked.close()

# SDK JointMotion produces all safe_home samples; execution holds each target
# for duration/N as in the SDK, rather than resplining/replanning endpoints.
from reBotArm_control_py.trajectory import JointMotion
plan=p.plan_home(observed)
raw=p.command_from_feedback(observed)
duration=max(2*np.abs(raw).max()/p.cfg['transit']['home_max_vel_rad_s'],.04)
motion=JointMotion(raw,np.zeros(6),duration)
expected=np.array([motion.sample(t)[0] for t in np.linspace(0,duration,len(plan.points))])
np.testing.assert_allclose(plan.points,expected,atol=1e-13)
for i in range(len(expected)):
    actual_q,actual_v=plan.sample((i+.25)*plan.duration/len(expected))
    np.testing.assert_allclose(actual_q,expected[i],atol=1e-13)
    np.testing.assert_array_equal(actual_v,np.zeros(6))
assert plan.diagnostics['source']=='SDK JointMotion with RebotArmEndPose.safe_home timing'
print('Measured joint3 boundary case, SDK feedback policy, strict target limits, fresh group reads, no-enable rejection, and SDK safe_home sample playback passed')

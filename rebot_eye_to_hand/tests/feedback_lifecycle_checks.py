"""Native SDK command/reply lifecycle: fake motor handles, no CAN/camera access."""
from pathlib import Path
import sys,time,json,itertools
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import eyehand
from eyehand.click_hover import import_sdk,load_config
cfg=load_config(ROOT/'click_hover_config.json');import_sdk(cfg['sdk_root'])
from eyehand.hover_control import HoverController
from eyehand.native_feedback import NativeFeedback,FeedbackUnavailable
names=[f'joint{i}' for i in range(1,7)]+['gripper']
Q=np.array([.1,.2,.3,.1,.2,.3,.25])

class Motor:
 def __init__(self,index,events):
  self.index=index;self.events=events;self.seq=0;self.received_at=-float('inf');self.enabled=False
  self.reporting=None;self.pos=Q[index];self.stale=False;self.reply_delay=0.
 def robstride_set_active_report(self,value):
  if value:raise AssertionError('Duplicate periodic reports must never be enabled')
  self.reporting=value;self.events.append(('report_off',self.index))
 def disable(self):self.enabled=False;self.events.append(('disable',self.index))
 def enable(self):self.enabled=True;self.events.append(('enable',self.index))
 def send_mit(self,pos,vel,kp,kd,tau):
  self.events.append(('send',self.index,kp))
  if not self.stale:self.seq+=1;self.received_at=time.perf_counter()+self.reply_delay
 def get_state_snapshot(self):
  if not self.seq:return None
  return NS(state=NS(pos=self.pos,status_code=0,vel=0.,torq=0.),
   feedback_age_s=time.perf_counter()-self.received_at,feedback_sequence=self.seq)


def fixture():
 events=[];motors={n:Motor(i,events) for i,n in enumerate(names)}
 def group(selected):
  def send(pos,vel=None,kp=None,kd=None,tau=None,**kwargs):
   if not any(m.enabled for m in motors.values()):
    assert np.array_equal(kp,np.zeros(len(selected)))
    assert np.array_equal(kd,np.zeros(len(selected)))
    assert np.array_equal(tau,np.zeros(len(selected)))
   for i,name in enumerate(selected):
    if kwargs.get('cancelled',lambda:False)():raise InterruptedError('cancelled')
    motors[name].send_mit(float(pos[i]),float(0 if vel is None else vel[i]),
     float(50 if kp is None else kp[i]),float(4 if kd is None else kd[i]),float(0 if tau is None else tau[i]))
  return NS(joint_names=selected,mode_mit=Mock(return_value=True),send_mit=Mock(side_effect=send),
   get_positions=Mock(return_value=Q[[names.index(n) for n in selected]].copy()),
   _mit_kp=np.array([50.]),_mit_kd=np.array([4.]))
 arm,grip=group(names[:6]),group(names[6:])
 robot=NS(connect=Mock(),arm=arm,gripper=grip,groups={'gripper':grip},_motor_map=motors,
  _ctrl_map={'rs':NS(poll_feedback_once=Mock(),close=Mock())},
  _all_joints=[NS(name=n,vendor='robstride',motor_id=i+1,feedback_id=0xFD,model='rs-06' if i<3 else 'rs-00') for i,n in enumerate(names)],
  start_control_loop=Mock(),stop_control_loop=Mock(),disconnect=Mock())
 planner=NS(cfg=cfg,validate_feedback=lambda x:x,validate_raw=lambda x:x,
  command_from_feedback=lambda x:x.copy(),gravity_torque=lambda x:np.zeros(6))
 c=HoverController(planner,robot)
 return c,robot,events


class LifecycleTests(unittest.TestCase):
 def manual_fixture(self):
  c,r,e=fixture();c.cfg=dict(c.cfg,manual_disable_only=True)
  with patch('eyehand.hover_control.time.sleep'):c.start()
  return c,r,e

 def test_manual_policy_fault_freezes_target_and_continues_sdk_hold_without_disable(self):
  c,r,e=self.manual_fixture();target=c.target.copy();before=sum(v[0]=='disable' for v in e)
  c.feedback.snapshot=Mock(side_effect=RuntimeError('injected feedback fault'))
  c._tick(r,.002);self.assertIn('injected',c.fault)
  c._tick(r,.002)
  self.assertTrue(all(m.enabled for m in r._motor_map.values()))
  self.assertEqual(sum(v[0]=='disable' for v in e),before)
  c.feedback.snapshot.assert_called_once()
  np.testing.assert_array_equal(r.arm.send_mit.call_args.args[0],target)
  np.testing.assert_array_equal(r.arm.send_mit.call_args.kwargs['vel'],np.zeros(6))
  with self.assertRaisesRegex(RuntimeError,'Still holding'):c.close()
  self.assertFalse(c.stopped);r.disconnect.assert_not_called();r.stop_control_loop.assert_not_called()
  c.authorize_manual_disable();c.close()
  self.assertTrue(all(not m.enabled for m in r._motor_map.values()))

 def test_manual_policy_transport_failure_never_disables_or_auto_resumes(self):
  c,r,e=self.manual_fixture();before=sum(v[0]=='disable' for v in e)
  original=r.arm.send_mit.side_effect
  r.arm.send_mit.side_effect=RuntimeError('injected CAN error')
  c.enter_fault_hold('feedback error');c._tick(r,.002)
  self.assertIn('CAN error',c.hold_transport_error)
  self.assertTrue(c.enabled);self.assertEqual(sum(v[0]=='disable' for v in e),before)
  r.arm.send_mit.side_effect=original;c.last_hold_attempt=-float('inf');c._tick(r,.002)
  self.assertIsNone(c.hold_transport_error);self.assertEqual(c.fault,'feedback error')
  self.assertIsNone(c.plan)

 def test_manual_policy_home_does_not_itself_authorize_disable(self):
  c,r,e=self.manual_fixture();c.home_confirmed=True
  with self.assertRaisesRegex(RuntimeError,'manual release'):c.close()
  self.assertTrue(c.enabled);self.assertFalse(c.stopped)
  c.authorize_manual_disable();c.close()

 def test_manual_terminal_release_requires_no_home_or_fault_recovery(self):
  from eyehand.hover_runtime import release_supported
  c,r,e=self.manual_fixture();c.enter_fault_hold('injected error')
  c.planner.plan_home=Mock(side_effect=AssertionError('No extra home'))
  release_supported(c,input_fn=lambda _:'release')
  c.planner.plan_home.assert_not_called();self.assertTrue(c.manual_disable_authorized)
  self.assertEqual(c.fault,'injected error');self.assertFalse(c.home_confirmed)
  self.assertTrue(c.enabled)  # Actual disable only occurs after authorized close.
  c.close()

 def test_manual_terminal_eof_and_ctrl_c_do_not_authorize_disable(self):
  from eyehand.hover_runtime import release_supported
  c,r,e=self.manual_fixture();calls=[0]
  def read(**kwargs):
   calls[0]+=1
   self.assertFalse(c.manual_disable_authorized)
   if calls[0]==1:return EOFError('terminal closed')
   if calls[0]==2:raise KeyboardInterrupt()
   c.home_confirmed=True;return 'release'
  commands=Mock();commands.get.side_effect=read
  with patch('eyehand.hover_runtime.queue.Queue',return_value=commands),patch('eyehand.hover_runtime.threading.Thread.start'):
   release_supported(c,input_fn=lambda _:'unused')
  self.assertEqual(calls[0],3);self.assertTrue(c.manual_disable_authorized);self.assertTrue(c.enabled)
  c.close()

 def test_disabled_probes_then_native_stream_and_no_periodic_report(self):
  c,r,e=fixture()
  with patch('eyehand.hover_control.time.sleep'):c.start()
  self.assertIsInstance(c.feedback,NativeFeedback)
  self.assertIn('get_state_snapshot',c.feedback_source)
  first_enable=min(i for i,v in enumerate(e) if v[0]=='enable')
  self.assertTrue(all(('report_off',i) in e[:first_enable] for i in range(7)))
  self.assertTrue(all(('disable',i) in e[:first_enable] for i in range(7)))
  self.assertTrue(all(any(v[:2]==('send',i) for v in e[:first_enable]) for i in range(7)))
  self.assertTrue(all(m.seq>=3 for m in r._motor_map.values()))
  self.assertTrue(c.enabled);self.assertFalse(c.disabled_confirmed)
  c.close();self.assertTrue(all(not m.enabled and m.reporting is False for m in r._motor_map.values()))
  json.dumps(c.feedback_diagnostics(),allow_nan=False)

 def test_startup_hook_delay_refreshes_using_disabled_status_before_enable(self):
  c,r,e=fixture()
  def slow_ui():
   for m in r._motor_map.values():m.received_at-=1.
  c.before_enable=slow_ui
  with patch('eyehand.hover_control.time.sleep'):c.start()
  self.assertTrue(c.enabled)
  self.assertTrue(all(time.perf_counter()-m.received_at<.15 for m in r._motor_map.values()))
  c.close()

 def test_moved_pose_during_hook_still_refuses_enable(self):
  c,r,e=fixture()
  c.before_enable=lambda:setattr(r._motor_map['joint2'],'pos',Q[1]+.02)
  with patch('eyehand.hover_control.time.sleep'):
   with self.assertRaisesRegex(RuntimeError,'moved during startup'):c.start()
  self.assertFalse(any(m.enabled for m in r._motor_map.values()))
  c.close()

 def test_missing_gripper_status_still_refuses_enable(self):
  c,r,e=fixture();r._motor_map['gripper'].stale=True
  ticks=itertools.count(1000.,.02)
  with patch('eyehand.hover_control.time.sleep'),patch('eyehand.hover_control.time.perf_counter',side_effect=lambda:next(ticks)):
   with self.assertRaisesRegex(RuntimeError,'No fresh feedback before enable'):c.start()
  self.assertFalse(any(m.enabled for m in r._motor_map.values()))
  c.close()

 def test_probe_rejected_after_enable_and_if_disable_unconfirmed(self):
  c,r,e=fixture()
  with self.assertRaisesRegex(RuntimeError,'confirmed disabled'):c._disabled_probe()
  c.disabled_confirmed=True;c.enabled=True
  with self.assertRaisesRegex(RuntimeError,'confirmed disabled'):c._disabled_probe()
  self.assertFalse(any(v[0]=='send' for v in e))

 def test_runtime_gripper_50hz_arm_500hz_and_expiry_not_relaxed(self):
  c,r,e=fixture()
  c.enabled=True;c.feedback=NativeFeedback(r,names);c.target=Q[:6].copy();c.gripper=Q[6:].copy()
  for m in r._motor_map.values():m.enabled=True;m.seq=1;m.received_at=100.;m.reply_delay=.000001
  for i in range(51):
   now=100.+i*.002;c.ui_seen=now
   with patch('eyehand.hover_control.time.perf_counter',return_value=now):c._tick(r,.002)
   self.assertIsNone(c.fault,c.fault)
  self.assertEqual(c.send_counts,dict(arm_batches=51,gripper_batches=6))
  # All seven motors remain checked. A repeated cached read never refreshes age.
  gripper=r._motor_map['gripper'];gripper.stale=True;last=gripper.received_at
  for i in range(1,77):
   now=100.1+i*.002;c.ui_seen=now
   for name,m in r._motor_map.items():
    if name!='gripper':m.received_at=now;m.seq+=1
   with patch('eyehand.hover_control.time.perf_counter',return_value=now):c._tick(r,.002)
   if c.fault:break
  self.assertIn('gripper',c.fault);self.assertEqual(gripper.received_at,last)
  self.assertEqual(c.cfg['feedback_max_age_s'],.15)
  self.assertFalse(any(m.enabled for m in r._motor_map.values()))

 def test_partial_report_setup_failure_never_enables_any_motor(self):
  c,r,e=fixture()
  def failed(value):raise RuntimeError('report off setup failed')
  r._motor_map['joint3'].robstride_set_active_report=failed
  with self.assertRaisesRegex(RuntimeError,'report off setup failed'):c.start()
  self.assertFalse(any(v[0]=='enable' for v in e))
  self.assertFalse(any(v[0]=='send' for v in e))
  r._motor_map['joint3'].robstride_set_active_report=lambda value:None
  c.close()

if __name__=='__main__':unittest.main(verbosity=2)

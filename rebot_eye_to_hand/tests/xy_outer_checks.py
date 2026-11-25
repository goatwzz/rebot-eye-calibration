"""XY-only outer loop with actual calibrated SDK/FK and a mock lag plant."""
from pathlib import Path
import sys,json
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eyehand.click_hover import HoverPlanner,MotionPlan,load_config
from eyehand.cartesian_settle import CartesianSettler,validate_settings
from eyehand.hover_control import HoverController
from eyehand.table_reference import effective_hover_m


class XYOuterChecks(unittest.TestCase):
    def make_case(self, delta=(.003,-.002,.006)):
        cfg=load_config();cfg['cartesian_outer_loop']['axes']='xy'
        cfg['hover_m']=.088;cfg['height_compensation_m']=-.015  # Historical XY-only regression fixture.
        # Regression for the retained XY-only option. The active external
        # configuration now uses XYZ tracking, covered by sdk_hover_checks.py.
        p=HoverPlanner(cfg)
        rows=[json.loads(x) for x in (Path(__file__).resolve().parents[1]/'runs/click_hover/20261122_222950_b1d44af7.jsonl').read_text().splitlines()]
        click=next(r for r in rows if r.get('event')=='click')
        target=np.array(click['hover_base_m']);target[2]=p.cfg['table_reference']['table_z_m']+effective_hover_m(p.cfg)
        goal=p.plan(np.array(click['raw_start_rad']),target).goal
        target=p.tcp(goal)
        out=p.ik.solve_ik(p.model,p.model.createData(),p.tcp_id,p.pin.SE3(np.eye(3),target+delta),goal+p.offsets,
                          p.ik.IKParams(max_iter=1000,tolerance=1e-8),controlled_joints=6,position_only=True)
        self.assertTrue(out.success)
        actual=p.validate_raw(out.q-p.offsets);bias=goal-actual
        plan=MotionPlan([goal,goal],p.cfg,0.,joint_space=True);plan.target=target
        robot=SimpleNamespace(arm=Mock(),gripper=Mock(),estop=Mock(),_motor_map={})
        c=HoverController(p,robot);c.enabled=True;c.target=goal.copy();c.gripper=np.array([.25])
        c.feedback=Mock();c.feedback.snapshot.return_value=SimpleNamespace(positions=np.r_[actual,.25])
        c.plan=plan;c.started=0.;c.stable_since=None
        return p,c,goal,target,actual,bias

    def tick(self,c,now,raw):
        c.feedback.snapshot.return_value.positions[:6]=raw;c.ui_seen=now
        with patch('eyehand.hover_control.time.perf_counter',return_value=now):c._tick(c.robot,.01)
        self.assertIsNone(c.fault,c.fault)

    def run_plant(self,case,quantized=False,tau=.15):
        p,c,goal,target,actual,bias=case
        actual=actual.copy();commands=[];xy=[];z=[];start=c.plan.duration+.01
        for i in range(1300):
            measured=actual.copy()
            if quantized:
                quantum=.0003835069006732022;measured=quantum*np.round(measured/quantum)
            self.tick(c,start+.01*i,measured)
            command=c.robot.arm.send_mit.call_args.args[0].copy();commands.append(command)
            error=p.tcp(measured)-target;xy.append(float(np.linalg.norm(error[:2])));z.append(float(error[2]))
            actual+=(1-np.exp(-.01/tau))*(command-bias-actual)
            if not c.busy:break
        return c.take_completion(),np.array(commands),xy,z,actual,start+.01*i

    def test_external_settings_and_xy_converge_while_z_bias_remains(self):
        for quantized,tau in ((False,.15),(True,.5)):
            with self.subTest(quantized=quantized,tau=tau):
                case=self.make_case();p,c,goal,target,actual,bias=case
                self.assertEqual(p.cfg['cartesian_outer_loop']['axes'],'xy')
                self.assertAlmostEqual(p.cfg['height_compensation_m'],-.015)
                self.assertAlmostEqual(effective_hover_m(p.cfg),.073)
                done,commands,xy,z,actual,now=self.run_plant(case,quantized,tau)
                self.assertEqual(done['status'],'SETTLED_MODEL_ONLY')
                self.assertEqual(done['settled_axes'],'xy')
                self.assertLessEqual(done['controlled_fk_error_mm'],1.)
                self.assertLessEqual(done['fk_error_xy_mm'],1.)
                self.assertGreater(done['fk_error_mm'],3.)
                self.assertGreater(abs(done['fk_error_xyz_mm'][2]),5.)
                self.assertFalse(done['cartesian_outer_loop']['z_feedback_enabled'])
                self.assertFalse(done['physical_accuracy_verified'])
                self.assertLessEqual(np.abs(commands-goal).max(),.015+1e-10)
                command_z=[p.tcp(command)[2] for command in commands]
                self.assertLessEqual(np.max(np.abs(np.array(command_z)-p.tcp(goal)[2])),.0005+1e-10)
                self.assertLessEqual(done['cartesian_outer_loop']['peak_joint_command_speed_rad_s'],.02+1e-10)
                self.assertLessEqual(done['cartesian_outer_loop']['peak_tcp_command_speed_m_s'],.003+1e-10)
                held=c.target.copy();np.testing.assert_array_equal(c.planning_start(),held)
                next_plan=p.plan(held,target+[.003,0,0]);c.ui_seen=now+.01
                with patch('eyehand.hover_control.time.perf_counter',return_value=now+.01):c.move(next_plan)
                self.tick(c,now+.01,actual)
                np.testing.assert_allclose(c.robot.arm.send_mit.call_args.args[0],held,atol=1e-12)

    def test_pure_z_error_does_not_integrate_or_prevent_xy_arrival(self):
        case=self.make_case((0.,0.,.020));p,c,goal,target,actual,bias=case
        with patch.object(p.ik,'solve_ik',wraps=p.ik.solve_ik) as solve:
            done,commands,xy,z,actual,now=self.run_plant(case)
        self.assertEqual(done['status'],'SETTLED_MODEL_ONLY')
        self.assertGreater(done['fk_error_xyz_mm'][2],19.)
        self.assertLess(done['fk_error_xy_mm'],.01)
        self.assertEqual(solve.call_count,0)
        np.testing.assert_allclose(commands,np.tile(goal,(len(commands),1)),atol=1e-12)

    def test_official_ik_sees_current_measured_z_and_corrected_angles_once(self):
        p,c,goal,target,actual,bias=self.make_case()
        s=CartesianSettler(p,goal,target);s.step(actual,0.)
        with patch.object(p.ik,'solve_ik',wraps=p.ik.solve_ik) as solve:s.step(actual,.01)
        self.assertEqual(solve.call_count,1)
        args=solve.call_args.args
        np.testing.assert_allclose(args[3].translation[:2],target[:2])
        self.assertAlmostEqual(args[3].translation[2],p.tcp(actual)[2],places=12)
        np.testing.assert_allclose(args[4],actual+p.offsets)
        self.assertTrue(solve.call_args.kwargs['position_only'])

    def test_xy_capture_hold_and_z_drift_bounds(self):
        p,c,goal,target,actual,bias=self.make_case()
        s=CartesianSettler(p,goal,target)
        s.target[:2]+=[.03,0]
        np.testing.assert_array_equal(s.step(actual,0.),goal)
        self.assertEqual(s.stop_reason,'capture_radius_exceeded')
        p,c,goal,target,actual,bias=self.make_case()
        s=CartesianSettler(p,goal,target);p.cfg['cartesian_outer_loop']['max_uncontrolled_z_drift_m']=1e-12
        for i in range(100):s.step(actual,.01*i)
        self.assertLessEqual(abs(s.fk(s.command)[2]-s.nominal_tcp[2]),1e-12)
        self.assertGreater(s.saturation_count,0)
        self.tick(c,c.plan.duration+.01,actual);self.tick(c,c.plan.duration+.02,actual)
        held=c.target.copy();c.hold('USER_HOLD');self.tick(c,3.06,actual)
        np.testing.assert_array_equal(c.target,held);self.assertIsNone(c.cartesian_settler)

    def test_axis_configuration_is_explicit_and_defaults_to_xyz(self):
        cfg=load_config()
        self.assertEqual(validate_settings({'enabled':True},cfg)['axes'],'xyz')
        for bad in ({'axes':'z'},{'axes':['x','y']},{'max_uncontrolled_z_drift_m':.01}):
            with self.assertRaises(ValueError):validate_settings(bad,cfg)


if __name__=='__main__':unittest.main()

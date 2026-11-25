"""Console keeps actionable status; diagnostic payloads remain in JSONL."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from scripts import click_hover_eth as app
import test_click_hover as fault_fixtures
import test_hover_gui_review as gui_fixtures


class TerminalOutputTests(unittest.TestCase):
    def test_brief_startup_and_full_startup_logging_callback(self):
        with tempfile.TemporaryDirectory() as folder:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                _, _, _, factory = gui_fixtures.GUIReviewTests().run_fake(folder)
            startup_log = factory.call_args.kwargs['startup_log']
            detail = {'event': 'robot_startup', 'raw_q_rad': [0.]*6,
                      'feedback_source': 'SDK timestamp snapshot'}
            startup_log(detail)
            path = next(Path(folder).glob('*.jsonl'))
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(events[-1], {'unix_s': events[-1]['unix_s'], **detail})
        lines = output.getvalue().splitlines()
        self.assertLessEqual(len(lines), 6)
        self.assertIn('[操作]', output.getvalue())
        self.assertIn('q/Esc', output.getvalue())
        self.assertNotIn('raw_q_rad', output.getvalue())
        self.assertNotIn('cartesian_outer_loop', output.getvalue())

    def test_success_summarizes_error_but_home_timeout_remains_a_warning(self):
        settled = {'status': 'SETTLED_MODEL_ONLY', 'fk_error_mm': .229038,
                   'elapsed_s': 23.526, 'raw_q_rad': [1.]*6,
                   'cartesian_outer_loop': {'updates': 2020}}
        text = app.completion_summary(settled)
        self.assertIn('FK', text)
        self.assertIn('0.23', text)
        self.assertNotIn('2020', text)
        warning = app.completion_summary({'status': 'HOME_TIMEOUT_HOLD',
            'fk_error_mm': 8.2359, 'home_joint_error_deg': .79104})
        self.assertIn('警告', warning)
        self.assertIn('未达标', warning)
        self.assertIn('8.24', warning)
        self.assertIn('0.79', warning)
        self.assertIsNone(app.completion_summary({'status': 'HOME_REQUESTED_HOLD'}))

    def test_planned_click_is_one_line_with_target_and_duration(self):
        text = app.click_summary({'hover_base_m': [.3655, -.1411, .10755],
            'execution_requested': True, 'duration_s': 19.337, 'raw_goal_rad': [1.]*6})
        self.assertIn('365.5', text)
        self.assertIn('19.3', text)
        self.assertNotIn('raw_goal_rad', text)
        self.assertNotIn('\n', text)

    def test_requested_shutdown_cancel_does_not_report_a_control_fault(self):
        c, robot = fault_fixtures.ControlFaultTests().make_controller()
        def cancel_send(*args, **kwargs):
            c.stopped = True  # close() races with an in-flight SDK send.
            raise InterruptedError('arm MIT batch cancelled')
        robot.arm.send_mit.side_effect = cancel_send
        c._tick(robot, .002)
        self.assertIsNone(c.fault)
        robot.estop.assert_not_called()
        c.close()
        robot.stop_control_loop.assert_called_once()
        robot.estop.assert_called_once()  # Normal shutdown still disables.

    def test_unrequested_send_cancellation_is_still_a_fault(self):
        c, robot = fault_fixtures.ControlFaultTests().make_controller()
        robot.arm.send_mit.side_effect = InterruptedError('unexpected cancellation')
        with contextlib.redirect_stdout(io.StringIO()): c._tick(robot, .002)
        self.assertIn('unexpected cancellation', c.fault)
        robot.estop.assert_called_once()


if __name__ == '__main__': unittest.main()

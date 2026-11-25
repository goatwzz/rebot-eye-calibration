"""Exercise the upstream snapshot API adapter without native motors or sockets."""
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
import numpy as np
from eyehand.native_feedback import NativeFeedback,FeedbackUnavailable


class NativeFeedbackTests(unittest.TestCase):
    def fixture(self):
        clock=[10.];sample=[SimpleNamespace(state=SimpleNamespace(pos=.2,status_code=0),
            feedback_age_s=.001,feedback_sequence=1)]
        motor=Mock();motor.get_state_snapshot.side_effect=lambda:sample[0]
        controller=Mock();robot=SimpleNamespace(_motor_map={'joint1':motor},_ctrl_map={'rs':controller})
        f=NativeFeedback(robot,['joint1'])
        return f,clock,sample,controller

    def test_typed_upstream_api_gives_age_and_sequence_without_protocol_access(self):
        f,clock,sample,controller=self.fixture()
        with patch('eyehand.native_feedback.time.perf_counter',side_effect=lambda:clock[0]):s=f.snapshot(.03)
        np.testing.assert_allclose(s.positions,[.2]);np.testing.assert_allclose(s.times,[9.999])
        np.testing.assert_array_equal(s.sequences,[1]);controller.poll_feedback_once.assert_called_once()

    def test_rereading_cached_sequence_cannot_refresh_age(self):
        f,clock,sample,_=self.fixture()
        with patch('eyehand.native_feedback.time.perf_counter',side_effect=lambda:clock[0]):
            first=f.snapshot(.03);clock[0]+=.01;sample[0].feedback_age_s=.011
            second=f.snapshot(.03);np.testing.assert_array_equal(first.times,second.times)
            clock[0]+=.03;sample[0].feedback_age_s=.001
            with self.assertRaisesRegex(FeedbackUnavailable,'expired'):f.snapshot(.03)

    def test_sequence_reversal_or_changed_duplicate_is_not_hidden(self):
        for change in ('sequence','position'):
            f,clock,sample,_=self.fixture();sample[0].feedback_sequence=2
            with patch('eyehand.native_feedback.time.perf_counter',side_effect=lambda:clock[0]):
                f.snapshot(.03);clock[0]+=.01
                if change=='sequence':sample[0].feedback_sequence=1
                else:sample[0].state.pos=.3
                with self.assertRaisesRegex(RuntimeError,'reversed|changed position'):f.snapshot(.03)

    def test_missing_stale_nonfinite_and_faulty_upstream_samples(self):
        for bad in ('missing','stale','nan','fault','negative_age'):
            f,clock,sample,_=self.fixture()
            if bad=='missing':sample[0]=None
            elif bad=='stale':sample[0].feedback_age_s=.1
            elif bad=='nan':sample[0].state.pos=float('nan')
            elif bad=='fault':sample[0].state.status_code=8
            else:sample[0].feedback_age_s=-.001
            with patch('eyehand.native_feedback.time.perf_counter',side_effect=lambda:clock[0]):
                with self.assertRaises((FeedbackUnavailable,RuntimeError)):f.snapshot(.03)

    def test_timing_distinguishes_frame_rate_from_host_observed_gap(self):
        f,clock,sample,_=self.fixture()
        with patch('eyehand.native_feedback.time.perf_counter',side_effect=lambda:clock[0]):
            for i in range(6):
                clock[0]=10.+i*.01;sample[0].feedback_sequence=1+i*5;f.snapshot(.03)
        self.assertAlmostEqual(f.rate_capacity(),500.)
        self.assertAlmostEqual(f.max_observed_period(),.01)
        f.reset_intervals()
        with self.assertRaises(FeedbackUnavailable):f.rate_capacity()

    def test_snapshot_timestamp_regression_remains_fatal(self):
        f,clock,sample,_=self.fixture()
        with patch('eyehand.native_feedback.time.perf_counter',side_effect=lambda:clock[0]):
            f.snapshot(.03);clock[0]+=.001;sample[0].feedback_sequence=2;sample[0].feedback_age_s=.01
            with self.assertRaisesRegex(RuntimeError,'time reversed'):f.snapshot(.03)

    def test_delayed_api_call_with_newer_packet_does_not_false_trip(self):
        f,clock,sample,_=self.fixture()
        with patch('eyehand.native_feedback.time.perf_counter',side_effect=lambda:clock[0]):
            f.snapshot(.03);previous=f.times.copy()
            clock[0]=10.001;sample[0].feedback_sequence=2;sample[0].feedback_age_s=.003
            # SDK samples age at10.004: true packet time10.001 is NEWER than
            # the first packet9.999, despite the conservative lower bound9.998.
            f.motors[0].get_state_snapshot.side_effect=lambda:(clock.__setitem__(0,10.004) or sample[0])
            accepted=f.snapshot(.03)
            np.testing.assert_array_equal(accepted.times,previous)
            self.assertEqual(accepted.sequences[0],2)
            self.assertEqual(f.timestamp_bound_adjustments[0],1)
            self.assertEqual(len(f.intervals[0]),0)
            f.motors[0].get_state_snapshot.side_effect=lambda:sample[0]
            clock[0]=10.006;sample[0].feedback_sequence=3;sample[0].feedback_age_s=.001
            f.snapshot(.03)
            self.assertEqual(f.intervals[0][-1][1],2)  # No lost frame count.

    def test_slow_api_call_cannot_make_an_expired_stream_fresh(self):
        f,clock,sample,_=self.fixture()
        with patch('eyehand.native_feedback.time.perf_counter',side_effect=lambda:clock[0]):
            f.snapshot(.03);clock[0]=10.001;sample[0].feedback_sequence=2
            f.motors[0].get_state_snapshot.side_effect=lambda:(clock.__setitem__(0,10.05) or sample[0])
            with self.assertRaisesRegex(FeedbackUnavailable,'expired'):f.snapshot(.03)

    def test_timing_reset_excludes_cross_window_10_911ms_without_refreshing_cache(self):
        f,clock,sample,_=self.fixture()
        with patch('eyehand.native_feedback.time.perf_counter',side_effect=lambda:clock[0]):
            first=f.snapshot(.08);old_times=f.times.copy();old_seq=f.sequences.copy()
            clock[0]=10.009;f.reset_intervals()
            np.testing.assert_array_equal(f.times,old_times);np.testing.assert_array_equal(f.sequences,old_seq)
            clock[0]=10.009911309;sample[0].feedback_age_s=0.;sample[0].feedback_sequence=2
            f.snapshot(.08)
            self.assertEqual(len(f.intervals[0]),0)
            for i in range(1,6):
                clock[0]=10.009911309+i*.002;sample[0].feedback_sequence=2+i;f.snapshot(.08)
        self.assertAlmostEqual(f.max_observed_period(),.002)
        self.assertAlmostEqual(f.rate_capacity(),500.)
        details=f.timing_details()[0]
        self.assertEqual(details['first_update']['read_index'],1)
        self.assertEqual(details['interval_count'],5)

    def test_reset_does_not_count_cached_reads_as_new_baselines_or_hide_steady_gaps(self):
        f,clock,sample,_=self.fixture()
        with patch('eyehand.native_feedback.time.perf_counter',side_effect=lambda:clock[0]):
            f.snapshot(.08);f.reset_intervals()
            for i in range(2):
                clock[0]+=.002;sample[0].feedback_age_s+=.002;f.snapshot(.08)
            self.assertFalse(f.timing_anchor[0])
            sample[0].feedback_age_s=0.;sample[0].feedback_sequence=2;f.snapshot(.08)
            for i in range(4):
                clock[0]+=.002;sample[0].feedback_sequence+=1;f.snapshot(.08)
            clock[0]+=.010911309;sample[0].feedback_sequence+=6;f.snapshot(.08)
        self.assertAlmostEqual(f.max_observed_period(),.010911309)
        details=f.timing_details()[0]
        self.assertEqual(details['first_update']['read_index'],3)
        self.assertEqual(details['worst_gap']['sequence_delta'],6)
        self.assertGreater(f.rate_capacity(),475.)

    def test_real_pinned_upstream_runtime_exposes_snapshot_capability_without_connecting(self):
        import motorbridge
        from motorbridge.abi import get_abi
        self.assertTrue(callable(motorbridge.Motor.get_state_snapshot))
        self.assertTrue(hasattr(get_abi().lib,'motor_handle_get_state_snapshot'))


if __name__=='__main__':unittest.main()

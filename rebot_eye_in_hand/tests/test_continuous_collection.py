"""60mm profile and capture-failure skip policy; no hardware access."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wrist.auto_calibration import geometry
from eyehand.auto_collection import load_settings
from eyehand.auto_collection_runtime import AutoSequence


class ContinuousCollectionTests(unittest.TestCase):
    def test_both_new_splits_use_60mm_id0_and_disable_failure_pause(self):
        for split in ['train','validation']:
            s,c,rows=load_settings(ROOT/'config/auto_collection.json',split,mount='wrist')
            self.assertEqual(s['target'],dict(type='aruco',dictionary='DICT_4X4_50',id=0,marker_m=.06))
            self.assertEqual(s['max_consecutive_capture_failures'],0)
            self.assertEqual(c['collection_gravity_link_mass_reductions_kg'],dict(gripper_left=.018,gripper_right=.018))
            self.assertTrue(c['manual_disable_only'])
            self.assertEqual(s['fps'],30)
            self.assertTrue(any(r['xyz_rpy'][2]<.1 for r in rows))

    def sequence(self,settings,rows):
        c=Mock();c.motion_epoch=1;c.take_completion.return_value=None
        emit=Mock();seq=AutoSequence(Mock(),c,rows,settings,emit)
        seq.worker=Mock(active=False);seq.worker.poll.return_value=None
        seq._submit=Mock()
        return seq,c,emit

    def test_more_than_three_failed_captures_skip_and_never_auto_pause(self):
        s,_,rows=load_settings(ROOT/'config/auto_collection.json',mount='wrist')
        seq,c,emit=self.sequence(s,rows)
        for _ in range(8):
            seq.phase='CAPTURING';seq.deadline=0;seq.pump()
        self.assertEqual(seq.index,8);self.assertEqual(len(seq.skipped),8)
        self.assertEqual(seq._submit.call_count,8)
        c.hold.assert_not_called()
        self.assertNotIn('capture_paused',[call.args[0]['event'] for call in emit.call_args_list])
        # Explicit user pause remains effective.
        seq.pause();self.assertEqual(seq.phase,'PAUSED');c.hold.assert_called_once()

    def test_other_profiles_keep_their_existing_three_failure_pause(self):
        s,_,rows=load_settings(ROOT.parent/'rebot_eye_to_hand/auto_collection_config.json')
        self.assertEqual(s['max_consecutive_capture_failures'],3)
        seq,c,emit=self.sequence(s,rows)
        for _ in range(3):
            seq.phase='CAPTURING';seq.deadline=0;seq.pump()
        self.assertEqual(seq.phase,'PAUSED');self.assertEqual(seq._submit.call_count,2)
        c.hold.assert_called_once()
        self.assertIn('capture_paused',[call.args[0]['event'] for call in emit.call_args_list])


if __name__=='__main__':unittest.main()

from pathlib import Path
import copy,hashlib,tempfile,unittest
from unittest.mock import patch
import numpy as np
import pinocchio as pin
from eyehand.kinematics import CorrectedKinematics,export_corrected_urdf
from eyehand.paths import PROJECT_ROOT,artifact_path

class RelocatedPathsTests(unittest.TestCase):
    def test_legacy_urdf_paths_do_not_require_old_symlink(self):
        from test_diagnostic_kinematics import KinematicsDiagnosticTests
        data,_,_,_=KinematicsDiagnosticTests().make_data(11,25)
        original=Path(data['robot']['urdf']);expected=CorrectedKinematics(data['robot'])
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);new=root/'saved.urdf';new.write_bytes(original.read_bytes())
            for old in ['/home/zzw/rebot_eye_to_hand','/home/zzw/robot_sim/rebot_eye_to_hand']:
                robot=dict(data['robot'],urdf=old+'/saved.urdf')
                with patch('eyehand.paths.PROJECT_ROOT',root):
                    model=CorrectedKinematics(robot)
                    np.testing.assert_allclose(model.fk(np.zeros(6)),expected.fk(np.zeros(6)),atol=1e-12)
                    correction=dict(original_urdf=old+'/saved.urdf',original_urdf_sha256=robot['urdf_sha256'],joint_names=data['robot']['joint_names'],joint_offsets_rad=[0.]*6)
                    export_corrected_urdf(correction,root/'export.urdf')
                    self.assertTrue((root/'export.urdf').is_file())

if __name__=='__main__':unittest.main()

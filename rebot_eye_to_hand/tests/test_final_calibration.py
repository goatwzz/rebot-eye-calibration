from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from eyehand.cli import save, validate
from scripts.finalize_handeye import compatible_groups, finalize
import test_diagnostic_kinematics as fixtures


class FinalCalibrationTests(unittest.TestCase):
    def write_runs(self, root):
        paths=[]
        factory=fixtures.KinematicsDiagnosticTests()
        for seed in (11,35,82):
            data,_,_,offsets=factory.make_data(seed,20)
            path=root/f"run_{seed}"/"dataset.json"
            save(path,data)
            paths.append(path)
        return paths,offsets

    def test_grouped_predictions_exclude_entire_test_run_and_inputs_stay_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            paths,offsets=self.write_runs(root)
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            out=root/"final"
            with contextlib.redirect_stdout(io.StringIO()):
                result,report=finalize(paths,out)
            self.assertEqual(len(result["training_ids"]),60)
            np.testing.assert_allclose(result["kinematic_correction"]["joint_offsets_rad"],offsets,atol=1e-7)
            self.assertEqual(report["aggregate"]["corrected"]["status"],"PASS_INTERNAL_CONSISTENCY")
            self.assertLess(report["aggregate"]["corrected"]["summary"]["translation_mm"]["max"],1e-4)
            self.assertFalse(result["physical_accuracy_verified"])
            for p in paths:
                data=json.loads(p.read_text())
                fold=json.loads((out/"cross_validation_models"/(p.parent.name+"_held_out.json")).read_text())
                held_ids={s["id"] for s in data["samples"]}
                self.assertFalse(held_ids&set(fold["training_ids"]))
                self.assertEqual(len(fold["training_ids"]),40)
            self.assertEqual(before,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
            args=argparse.Namespace(result=str(out/"result.json"),dataset=str(paths[0]),output=str(root/"invalid_validation.json"),
                                    translation_mm=5,rotation_deg=1,pixel_px=3)
            with self.assertRaisesRegex(ValueError,"overlap"):
                validate(args)
            with self.assertRaisesRegex(ValueError,"Output exists"):
                finalize(paths,out)

    def test_incompatible_intrinsics_and_duplicate_samples_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            paths,_=self.write_runs(root)
            original=json.loads(paths[1].read_text())
            changed=copy.deepcopy(original)
            changed["K"][0][0]+=1.
            save(paths[1],changed)
            with self.assertRaisesRegex(ValueError,"Incompatible K"):
                compatible_groups(paths)
            changed=copy.deepcopy(original)
            changed["samples"][0]["id"]=json.loads(paths[0].read_text())["samples"][0]["id"]
            save(paths[1],changed)
            with self.assertRaisesRegex(ValueError,"overlapping"):
                compatible_groups(paths)


if __name__=="__main__":unittest.main()

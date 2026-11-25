#!/usr/bin/env python3
"""Consolidate compatible runs, test by whole-run holdout, then fit all poses.

Fixed metric target, K/D and link geometry; identify only X/Y and joint2..5
effective angle offsets. Offline only. Original data/control files stay intact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyehand.cli import save
from eyehand.geometry import evaluate, inverse, solve
from eyehand.kinematics import corrected_dataset, export_corrected_urdf, identify_kinematics


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compatible_groups(paths):
    groups = []
    for name in paths:
        path = Path(name).expanduser().resolve()
        if path.is_dir(): path /= "dataset.json"
        data = json.loads(path.read_text())
        if not data["samples"]:
            raise ValueError(f"Empty pose dataset: {path}")
        groups.append({"name": path.parent.name, "path": path, "sha256": file_hash(path), "data": data})
    if len(groups) < 2 or len({g["name"] for g in groups}) != len(groups):
        raise ValueError("Need at least two separately named collection runs")
    reference = groups[0]["data"]
    all_ids, dataset_ids = set(), set()
    for group in groups:
        data = group["data"]
        if data["dataset_id"] in dataset_ids: raise ValueError("Repeated dataset ID")
        dataset_ids.add(data["dataset_id"])
        for key in ("target", "image_size", "K", "D", "synthetic"):
            if data[key] != reference[key]: raise ValueError(f"Incompatible {key}: {group['name']}")
        for field, keys in [("camera", ["kind", "serial", "device"]),
                            ("robot", ["base_frame", "ee_frame", "urdf_sha256", "joint_names"])]:
            for key in keys:
                if data[field].get(key) != reference[field].get(key):
                    raise ValueError(f"Incompatible {field}.{key}: {group['name']}")
        for sample in data["samples"]:
            if sample["id"] in all_ids: raise ValueError("Duplicate/overlapping sample ID")
            all_ids.add(sample["id"])
            if sample["K"] != data["K"] or sample["D"] != data["D"]:
                raise ValueError("Per-image intrinsics differ from dataset intrinsics")
    return groups


def merge_groups(groups):
    source = groups[0]["data"]
    digest = hashlib.sha256("|".join(g["sha256"] for g in groups).encode()).hexdigest()
    samples = []
    for group in groups:
        for sample in group["data"]["samples"]:
            item = {**sample, "source_run": group["name"], "source_dataset_id": group["data"]["dataset_id"]}
            if "image" in sample:
                item["image"] = str((group["path"].parent / sample["image"]).resolve())
            samples.append(item)
    return {**source, "dataset_id": "merged_" + digest[:24], "samples": samples,
            "source_runs": [{"name":g["name"],"path":str(g["path"]),"sha256":g["sha256"],
                             "dataset_id":g["data"]["dataset_id"],"count":len(g["data"]["samples"]),
                             "gravity_parameters":g["data"]["robot"].get("gravity_parameters")}
                            for g in groups]}


def aggregate(rows):
    summary = {}
    for key in ("translation_mm", "rotation_deg", "predicted_reprojection_rms_px"):
        values = np.array([row[key] for row in rows])
        summary[key] = {"rms":float(np.sqrt(np.mean(values**2))),"median":float(np.median(values)),
                        "p95":float(np.percentile(values,95)),"max":float(values.max())}
    return {"count":len(rows),"summary":summary,"samples":rows}


def acceptance(report):
    summary = report["summary"]
    checks = {"translation_p95":summary["translation_mm"]["p95"]<=5,
              "rotation_p95":summary["rotation_deg"]["p95"]<=1,
              "reprojection_p95":summary["predicted_reprojection_rms_px"]["p95"]<=3}
    return {"status":"PASS_INTERNAL_CONSISTENCY" if all(checks.values()) else "FAIL", "checks":checks}


def finalize(paths, output):
    output = Path(output).expanduser().resolve()
    if output.exists(): raise ValueError("Output exists; use a new final-calibration directory")
    groups = compatible_groups(paths)
    for held in groups:
        if len(held["data"]["samples"]) < 8: raise ValueError("Each held-out run needs at least eight poses")
        if sum(len(g["data"]["samples"]) for g in groups if g is not held) < 20:
            raise ValueError("Each training split needs at least twenty poses")
    output.mkdir(parents=True)
    cv_rows = {"nominal":[],"corrected":[]}
    folds = []
    for held in groups:
        train = merge_groups([g for g in groups if g is not held])
        test = held["data"]
        train_ids = {s["id"] for s in train["samples"]}
        assert not train_ids & {s["id"] for s in test["samples"]}
        nominal = solve(train["samples"],refinement="reprojection")
        nominal["training_dataset_id"] = train["dataset_id"]
        correction, result = identify_kinematics(train,initial=nominal)
        corrected_test = corrected_dataset(test,correction)
        reports = {"nominal":evaluate(test["samples"],np.array(nominal["T_base_camera"]),np.array(nominal["T_ee_target"])),
                   "corrected":evaluate(corrected_test["samples"],np.array(result["T_base_camera"]),np.array(result["T_ee_target"]))}
        fold = {"held_out_run":held["name"],"held_out_dataset_id":test["dataset_id"],
                "training_runs":[g["name"] for g in groups if g is not held],
                "training_count":len(train_ids),"test_count":len(test["samples"]),
                "joint_offsets_deg":correction["joint_offsets_deg"],
                "optimization":result["optimization"],
                "reports":{k:{**r,**acceptance(r)} for k,r in reports.items()}}
        for kind,report in reports.items():
            cv_rows[kind].extend([{**row,"source_run":held["name"],"prediction_from":"model_excluding_this_run"}
                                  for row in report["samples"]])
        save(output/"cross_validation_models"/(held["name"]+"_held_out.json"),result)
        folds.append(fold)
        print("Held out",held["name"],"train",len(train_ids),"test",len(test["samples"]),
              "corrected P95 mm/deg/px:",*[round(reports["corrected"]["summary"][k]["p95"],3)
                  for k in ("translation_mm","rotation_deg","predicted_reprojection_rms_px")],flush=True)
    cross_validation = {kind:{**aggregate(rows),**acceptance(aggregate(rows))} for kind,rows in cv_rows.items()}
    cv_report = {"scheme":"leave-one-collection-run-out; fit X/Y and offsets afresh in each fold",
                 "thresholds":{"translation_p95_mm":5,"rotation_p95_deg":1,"reprojection_p95_px":3},
                 "physical_accuracy_verified":False,
                 "interpretation":"Grouped cross-validation after prior model exploration; not an untouched independent acceptance test.",
                 "folds":folds,"aggregate":cross_validation}
    save(output/"cross_validation.json",cv_report)

    combined = merge_groups(groups)
    save(output/"combined_dataset.json",combined)
    correction, result = identify_kinematics(combined)
    result["training_dataset_sha256"] = file_hash(output/"combined_dataset.json")
    result["training_dataset_ids"] = [g["data"]["dataset_id"] for g in groups]
    result["source_runs"] = combined["source_runs"]
    urdf = output/"corrected_arm.urdf"
    export_corrected_urdf(correction,urdf)
    correction.update(exported_urdf=str(urdf),exported_urdf_sha256=file_hash(urdf),
                      training_dataset_ids=result["training_dataset_ids"])
    status = cross_validation["corrected"]["status"]
    result["acceptance"] = ("FAIL_CURRENT_THRESHOLDS: grouped cross-validation; physical accuracy unverified"
                            if status=="FAIL" else "GROUPED_CROSS_VALIDATION_PASS: physical accuracy unverified")
    result["test_report"] = "cross_validation.json"
    result["physical_accuracy_verified"] = False
    save(output/"kinematic_correction.json",correction)
    save(output/"result.json",result)
    corrected_all = corrected_dataset(combined,correction)
    same_data = {}
    for group in groups:
        ids = {s["id"] for s in group["data"]["samples"]}
        report = evaluate([s for s in corrected_all["samples"] if s["id"] in ids],
                          np.array(result["T_base_camera"]),np.array(result["T_ee_target"]))
        same_data[group["name"]] = {**report,"interpretation":"IN_SAMPLE: this run was included in final fitting"}
    save(output/"in_sample_by_run.json",same_data)
    with (output/"held_out_samples.csv").open("w",newline="") as stream:
        columns = ["id","source_run","translation_mm","rotation_deg","predicted_reprojection_rms_px","prediction_from"]
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(cv_rows["corrected"])
    report = {"date":"2026-11-20","pose_count":len(combined["samples"]),"run_count":len(groups),
              "source_runs":combined["source_runs"],"fixed_parameters":["camera K/D","marker 0.060m","URDF link lengths and axes"],
              "identified_parameters":["T_base_camera","T_ee_target","effective joint2..5 angle offsets"],
              "static_camera_bursts_excluded":True,"samples_removed":0,
              "training":result["training"]["summary"],"grouped_cross_validation":cross_validation["corrected"]["summary"],
              "grouped_cross_validation_nominal":cross_validation["nominal"]["summary"],
              "status":status,"physical_accuracy_verified":False,
              "original_inputs_unchanged":all(file_hash(g["path"])==g["sha256"] for g in groups),
              "use":"result.json embeds the paired FK correction. Validation on a newly collected nominal/raw-joint dataset applies it automatically. Old source runs now overlap training and cannot be called independent validation.",
              "hardware_accessed":False}
    save(output/"final_report.json",report)
    text = ["FINAL EYE-TO-HAND PARAMETER BUNDLE",f"Camera serial: {combined['camera'].get('serial')}",
            "p_base = T_base_camera @ p_color_optical_camera; translation in metres.",
            "T_base_camera:",np.array2string(np.array(result["T_base_camera"]),precision=10),
            "T_camera_base:",np.array2string(np.array(result["T_camera_base"]),precision=10),
            "T_ee_target:",np.array2string(np.array(result["T_ee_target"]),precision=10),
            "K:",np.array2string(np.array(combined["K"]),precision=10),"D: "+str(combined["D"]),
            "q_model = q_measured + offsets; degrees joint1..joint6:",str(correction["joint_offsets_deg"]),
            "Offsets are effective FK corrections, not measured motor zeros. Use original URDF + offsets OR corrected URDF + raw q.",
            "No controller, motor zero, dynamics or gravity settings changed.",
            "In-sample residuals:",json.dumps(report["training"],indent=2),
            "Grouped held-out prediction errors:",json.dumps(report["grouped_cross_validation"],indent=2),
            "Acceptance: "+result["acceptance"],
            f"All {len(combined['samples'])} poses are used for final fitting; grouped CV is not an independent physical accuracy measurement."]
    (output/"parameters_and_tests.txt").write_text("\n".join(text)+"\n")
    print(json.dumps({"output":str(output),"poses":len(combined["samples"]),"joint_offsets_deg":correction["joint_offsets_deg"],
                      "T_base_camera":result["T_base_camera"],"training":report["training"],
                      "grouped_cross_validation":report["grouped_cross_validation"],"acceptance":result["acceptance"]},indent=2))
    return result,cv_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets",nargs="+",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    finalize(args.datasets,args.output)


if __name__=="__main__": main()

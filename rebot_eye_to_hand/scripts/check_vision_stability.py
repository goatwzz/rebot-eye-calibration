#!/usr/bin/env python3
"""Camera-only stationary-marker repeatability check. Never opens robot/CAN.

Physically support the arm/target so they remain still. This checks short-term
repeatability, not absolute accuracy, intrinsics or robot kinematic truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyehand.cli import save
from eyehand.geometry import average, errors
from eyehand.vision import Camera, Target


def repeatability(observations):
    if not observations:
        raise ValueError("No valid target observations")
    poses = [np.array(s["T_camera_target"]) for s in observations]
    centre = average(poses)
    deviations = np.array([errors(t, centre) for t in poses])
    def stats(values):
        return {"rms": float(np.sqrt(np.mean(values ** 2))), "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values))}
    count = len(poses)
    block = max(1, count // 3)
    early, late = average(poses[:block]), average(poses[-block:])
    drift = errors(early, late)
    return {"valid_frames": count, "mean_T_camera_target": centre.tolist(),
            "translation_jitter_mm": stats(deviations[:, 0]),
            "rotation_jitter_deg": stats(deviations[:, 1]),
            "first_vs_last_third_drift_mm_deg": list(drift),
            "note": "Repeatability about the sample mean, not physical accuracy. Jitter includes target/camera motion."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial")
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--output", required=True)
    parser.add_argument("--intrinsics")
    parser.add_argument("--target", default=str(ROOT / "target_aruco_60mm.json"))
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.seconds <= 60:
        parser.error("--seconds must be between 2 and 60")
    output = Path(args.output)
    if output.exists():
        parser.error("Output exists; choose a new report filename")
    serial = args.serial or json.loads((ROOT / "camera_realsense.json").read_text())["serial"]
    target = Target(json.loads(Path(args.target).read_text()))
    camera = None
    observations, rejected = [], Counter()
    try:
        print("CAMERA ONLY: no robot/CAN access. Physically support the arm/marker; keep camera and marker still.",flush=True)
        camera = Camera("realsense",serial=serial,intrinsics=args.intrinsics)
        warmup = time.monotonic()
        while time.monotonic() - warmup < 2.:
            camera.read()
        start = time.monotonic()
        while time.monotonic() - start < args.seconds:
            frame = camera.read()
            try:
                obs = target.detect(frame,camera.k,camera.d)
                observations.append({"elapsed_s":time.monotonic()-start,
                                     "T_camera_target":obs["T_camera_target"], "pnp_rms_px":obs["pnp_rms_px"],
                                     "ippe_gap_px":obs["ippe_second_solution_gap_px"],
                                     "frame":dict(camera.last_frame_metadata)})
                label = f"{len(observations)} valid | PnP {obs['pnp_rms_px']:.2f}px"
            except ValueError as exc:
                reason = str(exc)
                rejected[reason.split(" (")[0].split(";")[0]] += 1
                label = reason
            if not args.no_window:
                cv2.putText(frame,label[:100],(15,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,230,255),1)
                cv2.imshow("Camera-only repeatability",frame)
                key = cv2.waitKey(1)&0xff
                if key in (ord("q"),27) or cv2.getWindowProperty("Camera-only repeatability",cv2.WND_PROP_VISIBLE)<1:
                    break
        summary = repeatability(observations) if observations else {"valid_frames":0}
        report = {"camera":camera.metadata,"K":camera.k.tolist(),"D":camera.d.tolist(),
                  "target":target.config,"duration_s":time.monotonic()-start,"hardware_robot_accessed":False,
                  "status":"RECORDED" if len(observations)>=30 else "INSUFFICIENT_VALID_FRAMES",
                  "physical_accuracy_verified":False,"summary":summary,"rejected":dict(rejected),"observations":observations}
        save(output,report)
        print(json.dumps({"output":str(output),"status":report["status"],"summary":summary,"rejected":dict(rejected)},indent=2))
    finally:
        if camera is not None:
            camera.close()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__=="__main__":
    main()

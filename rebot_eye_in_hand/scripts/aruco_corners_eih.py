#!/usr/bin/env python3
"""Wrist camera: key 1..4 -> initial observation pose -> RGB ArUco corner -> hover."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wrist.click_hover import WristPlanner, load_config
from eyehand.corner_demo import main

if __name__ == '__main__':
    raise SystemExit(main('wrist', ROOT/'config/aruco_corners.json', load_config, WristPlanner))

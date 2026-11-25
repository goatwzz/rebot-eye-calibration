#!/usr/bin/env python3
"""External camera: key 1..4 -> RAW home -> RGB ArUco corner -> hover."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eyehand.click_hover import HoverPlanner, load_config
from eyehand.corner_demo import main

if __name__ == '__main__':
    raise SystemExit(main('external', ROOT/'aruco_corners_config.json', load_config, HoverPlanner))

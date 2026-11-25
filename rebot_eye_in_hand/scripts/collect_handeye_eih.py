#!/usr/bin/env python3
"""Compatibility entry: manual wrist acquisition using the shared backend."""
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wrist.validation import main

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not any(a == "--output" or a.startswith("--output=") for a in arguments):
        name = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("collection_%Y%m%d_%H%M%S")
        arguments += ["--output", str(ROOT / "runs" / name)]
    main(["collect", *arguments])

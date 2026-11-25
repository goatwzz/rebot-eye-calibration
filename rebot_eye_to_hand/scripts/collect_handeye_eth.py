#!/usr/bin/env python3
"""Eye-to-hand entry point; --manual enables integrated gravity hand guiding."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eyehand.cli import main


if __name__ == "__main__":
    sys.argv.insert(1, "collect")
    main()

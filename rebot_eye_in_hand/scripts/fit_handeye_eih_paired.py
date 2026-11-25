#!/usr/bin/env python3
"""Offline wrist extrinsic + effective joint2..5 corrections; never activates results."""
from pathlib import Path
import os
import sys
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wrist.paired_calibration import main
if __name__ == '__main__':
    main()

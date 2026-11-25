#!/usr/bin/env python3
"""Automatic wrist hand-eye calibration with the shared official-SDK collector."""
from pathlib import Path
import os
import sys

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wrist.auto_calibration import main

if __name__=='__main__':
    raise SystemExit(main())

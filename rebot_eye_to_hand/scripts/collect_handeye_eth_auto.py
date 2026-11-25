#!/usr/bin/env python3
"""Automatic ETH collection: official SDK trajectory, stationary RGB captures."""
from pathlib import Path
import os
import sys
# Process-local numerical worker limits, BEFORE importing NumPy/OpenCV/SDK.
# Match the offline checks and leave CPU time for the sole motor sender.
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eyehand.auto_collection_runtime import main
if __name__=='__main__':
    raise SystemExit(main())

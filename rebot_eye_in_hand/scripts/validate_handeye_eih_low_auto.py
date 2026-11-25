#!/usr/bin/env python3
"""Low-height/high-pitch independent wrist validation; no refitting."""
from pathlib import Path
import os
import sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wrist.low_validation import main
if __name__=='__main__':raise SystemExit(main())

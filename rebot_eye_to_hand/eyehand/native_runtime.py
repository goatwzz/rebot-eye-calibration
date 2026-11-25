"""Select the project's verified upstream MotorBridge before importing the SDK.

Process-local dependency selection only: no global package installation, SDK
modification, native ABI declaration, SocketCAN reader or hardware access.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def activate():
    project = Path(__file__).resolve().parents[1]
    contract = json.loads((project/'runtime/motorbridge_contract.json').read_text())
    for name, expected in contract['files'].items():
        path = project/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Pinned upstream MotorBridge file changed: '+str(path))
    python = project/'runtime/python'
    package = python/'motorbridge'
    existing = sys.modules.get('motorbridge')
    if existing is not None and Path(existing.__file__).resolve().parent != package:
        raise RuntimeError('Another MotorBridge is already imported; start this demo in a fresh Python process')
    library = package/'lib/libmotor_abi.so'
    if str(python) not in sys.path:
        sys.path.insert(0, str(python))
    os.environ['MOTORBRIDGE_LIB'] = str(library)
    module = importlib.import_module('motorbridge')
    if (Path(module.__file__).resolve().parent != package
            or not callable(getattr(module.Motor, 'get_state_snapshot', None))):
        raise RuntimeError('Pinned MotorBridge public timestamped snapshot API unavailable')
    return dict(version=module.__version__, python_path=str(module.__file__),
                native_library=str(library), upstream_commit=contract['upstream_commit'])

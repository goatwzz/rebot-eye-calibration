"""Single explicit bridge to the sibling project's shared acquisition backend."""
from pathlib import Path
import sys


def dependencies(path):
    path = Path(path).resolve()
    if not (path / "eyehand/cli.py").is_file():
        raise ValueError(f"Shared acquisition backend missing: {path}")
    existing = sys.modules.get("eyehand")
    if existing is not None and Path(existing.__file__).resolve().parent.parent != path:
        raise ValueError("A different eyehand backend is already imported; start a fresh process")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from eyehand import cli, geometry
    return cli, geometry

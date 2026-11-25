"""Resolve project paths embedded before relocation without modifying provenance."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_ROOTS = (Path('/home/zzw/robot_sim/rebot_eye_to_hand'), Path('/home/zzw/rebot_eye_to_hand'))


def artifact_path(value):
    if str(value).startswith("repo:/"):
        return PROJECT_ROOT.parent / str(value).split(":", 1)[1].lstrip("/")
    path = Path(value)
    for previous in PREVIOUS_ROOTS:
        try:
            relative = path.relative_to(previous)
        except ValueError:
            continue
        return PROJECT_ROOT / relative
    return path

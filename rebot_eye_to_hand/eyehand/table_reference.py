"""Fixed horizontal table height, bound to camera/calibration and saved once."""
import json
from pathlib import Path
import uuid
import numpy as np


def effective_hover_m(cfg):
    """Requested table clearance plus explicit signed base-Z compensation."""
    offset = cfg.get('height_compensation_m', 0.)
    if not isinstance(offset, (int, float)) or not np.isfinite(offset):
        raise ValueError('height_compensation_m must be a finite signed number')
    height = cfg['hover_m']+offset
    if not np.isfinite(height) or height <= 0:
        raise ValueError('Compensated hover height must be positive')
    return float(height)


def validate_table_settings(cfg):
    cfg.setdefault('height_reference', 'table')
    cfg.setdefault('height_compensation_m', 0.)
    effective_hover_m(cfg)
    if cfg['height_reference'] != 'table':
        raise ValueError('Click demo requires height_reference=table')
    reference = cfg.get('table_reference')
    if reference is not None:
        if not isinstance(reference, dict):
            raise ValueError('Invalid saved table_reference')
        checked_table_z(cfg, reference.get('table_z_m'))


def checked_table_z(cfg, z):
    if (not isinstance(z, (int, float)) or not np.isfinite(z)
            or not cfg['workspace_min_m'][2] <= z+effective_hover_m(cfg) <= cfg['workspace_max_m'][2]):
        raise ValueError('Invalid table height or table hover height outside workspace')
    return float(z)


def matching_table_reference(cfg, calibration_sha256):
    saved = cfg.get('table_reference')
    if (saved and saved.get('camera_serial') == cfg['serial']
            and saved.get('calibration_sha256') == calibration_sha256):
        checked_table_z(cfg, saved.get('table_z_m'))
        return saved
    return None


def record_table_reference(cfg, surface, *, pixel, source_log, calibration_sha256):
    z = checked_table_z(cfg, float(np.asarray(surface)[2]))
    saved = dict(table_z_m=z, table_reference_id=uuid.uuid4().hex,
                 source_log=str(source_log), source_pixel=list(pixel),
                 camera_serial=cfg['serial'], calibration_sha256=calibration_sha256,
                 table_assumption='parallel_to_base_XY')
    save_table_reference(cfg, saved)
    return saved


def save_table_reference(cfg, reference):
    """Persist only the chosen table reference; preserve other user settings."""
    import tempfile
    path = Path(cfg['_path'])
    temporary = None
    try:
        raw = json.loads(path.read_text())
        raw['table_reference'] = reference
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.'+path.name+'.', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(raw, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(path)
        cfg['table_reference'] = reference
    except OSError as exc:
        raise RuntimeError(f'Table reference was not saved: {exc}') from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)



def table_hover_target(T_base_camera, point_camera, hover_m, *, table_z_m):
    from .click_hover import hover_target
    if not np.isscalar(table_z_m) or not np.isfinite(table_z_m):
        raise ValueError('Select a bare-table reference before clicking a motion target')
    surface, _ = hover_target(T_base_camera, point_camera, hover_m)
    target = surface.copy()
    target[2] = float(table_z_m)+hover_m
    return surface, target

"""Wrist camera transforms and shared SDK planning; no robot or camera I/O here."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from .backend import dependencies
from .validation import load_result

ROOT = Path(__file__).resolve().parents[1]
_, geometry = dependencies(ROOT.parent / 'rebot_eye_to_hand')
from eyehand.click_hover import HoverPlanner, import_sdk, in_workspace, load_config as load_shared_config


def load_config(path=ROOT / 'config/click_hover.json'):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    wrist_defaults = (path.parent / raw['calibration_config']).resolve()
    defaults = yaml.safe_load(wrist_defaults.read_text())
    project = wrist_defaults.parent.parent
    # Defaults remain the sole active hand-eye selection, including future activations.
    effective = {**raw, 'result': str((project / defaults['paths']['result']).resolve())}
    for key in ('sdk_root', 'gravity_config', 'log_dir', 'intrinsics', 'urdf'):
        effective[key] = str((path.parent / raw[key]).resolve())
    # Reuse existing validation of workspace, gains, speeds, depth and TCP definitions.
    cfg = load_shared_config(path, overrides=effective)
    cfg['_path'] = str(path)
    cfg['calibration_config'] = str(wrist_defaults)
    if cfg.get('height_reference') != 'table':
        raise ValueError('Wrist click demo requires height_reference=table')
    reference = cfg.get('table_reference')
    if reference is not None:
        if not isinstance(reference, dict):
            raise ValueError('Invalid saved table_reference')
        z = reference.get('table_z_m')
        if (not isinstance(z, (int, float)) or not np.isfinite(z)
                or not cfg['workspace_min_m'][2] <= z+cfg['hover_m'] <= cfg['workspace_max_m'][2]):
            raise ValueError('Invalid saved table height or hover height outside workspace')
    # Observation pose is controlled by this project's independent configuration.
    ready = raw['ready_pose']
    cfg['ready_pose'] = {k: float(ready.get(k, d)) for k, d in
                         dict(x=.25, y=0., z=.35, roll=0., pitch=1.2, yaw=0., duration=3.).items()}
    if not np.isfinite(list(cfg['ready_pose'].values())).all() or cfg['ready_pose']['duration'] <= 0:
        raise ValueError('Invalid reference ready pose')
    # Observation is a repeatable viewing pose, not the clicked TCP target.
    # Its camera transform always uses fresh measured joint angles.
    cfg.setdefault('ready_position_tolerance_m', .005)
    cfg.setdefault('ready_joint_tolerance_rad', .01)
    for key, maximum in (('ready_position_tolerance_m', .01), ('ready_joint_tolerance_rad', .02)):
        value = cfg[key]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(f'Invalid {key}')
    if not in_workspace([cfg['ready_pose'][k] for k in ('x', 'y', 'z')], cfg):
        raise ValueError('Reference ready pose outside workspace')
    if cfg['camera_fps'] != defaults['camera']['fps'] or cfg['serial'] != defaults['camera']['serial']:
        raise ValueError('Wrist RGB-D profile must match wrist calibration configuration')
    if (not np.isfinite([cfg['click_stationary_s'], cfg['click_joint_span_rad']]).all()
            or cfg['click_stationary_s'] < cfg['camera_max_age_s']+.1
            or not 0 < cfg['click_joint_span_rad'] <= np.radians(.2)):
        raise ValueError('Invalid stationary image/robot pose acquisition settings')
    if cfg.get('base_compensation_m') != [0., 0., 0.]:
        raise ValueError('This wrist demo requires zero extra base compensation')
    transit = cfg['transit'] = {**dict(time_scale=1., home_max_vel_rad_s=.5, home_target_hz=50.),
                                 **raw.get('transit', {})}
    if (not np.isfinite(list(transit.values())).all() or transit['time_scale'] < 1.
            or not 0 < transit['home_max_vel_rad_s'] <= .5 or transit['home_target_hz'] != 50.):
        raise ValueError('Invalid SDK transit timing; time_scale must be >= 1, home speed <= 0.5')
    return cfg


from eyehand.table_reference import save_table_reference


class WristPlanner(HoverPlanner):
    """Shared MIT/IK, with paired wrist extrinsic and effective joint corrections."""

    def __init__(self, cfg):
        import pinocchio as pin
        self.pin, self.cfg = pin, cfg
        self.ik = import_sdk(cfg['sdk_root'])
        self.T_gripper_camera, info = load_result(cfg['result'], geometry, allow_joint_offsets=True)
        if info.get('camera_serial') != cfg['serial']:
            raise ValueError('Wrist extrinsic camera identity mismatch')
        actual_urdf_hash = hashlib.sha256(Path(cfg['urdf']).read_bytes()).hexdigest()
        if info.get('urdf_sha256') != actual_urdf_hash:
            raise ValueError('Wrist extrinsic URDF fingerprint mismatch')
        intr = json.loads(Path(cfg['intrinsics']).read_text())
        if intr['camera']['serial'] != cfg['serial'] or intr['image_size'] != [1280, 720]:
            raise ValueError('Wrist intrinsics camera or image profile mismatch')
        k, d = np.asarray(intr['K'], float), np.asarray(intr['D'], float)
        if k.shape != (3, 3) or d.shape != (5,) or not np.isfinite(k).all() or not np.isfinite(d).all() or np.any(d != 0):
            raise ValueError('This RGB-D demo requires calibrated zero-distortion RGB')
        self.calibration = intr
        self.offsets = np.asarray(info['joint_offsets_rad'],float)
        self._initialize_robot_model(SimpleNamespace(model=pin.buildModelFromUrdf(cfg['urdf'])))
        self.result_sha256 = info['sha256']
        self.result_info = info
        self.camera_data = self.model.createData()
        self.ee_id = self.model.getFrameId('gripper_end')

    def camera_pose(self, raw):
        q = self.validate_feedback(raw) + self.offsets
        self.pin.forwardKinematics(self.model, self.camera_data, q)
        self.pin.updateFramePlacements(self.model, self.camera_data)
        return geometry.checked(self.camera_data.oMf[self.ee_id].homogeneous.copy()) @ self.T_gripper_camera

    def project_base(self, point, raw):
        import cv2
        pose = geometry.inverse(self.camera_pose(raw))
        camera_point = pose[:3, :3] @ np.asarray(point, float) + pose[:3, 3]
        if camera_point[2] <= .001:
            return None
        uv, _ = cv2.projectPoints(camera_point.reshape(1, 3), np.zeros(3), np.zeros(3),
                                  np.asarray(self.calibration['K']), np.asarray(self.calibration['D']))
        u, v = uv.reshape(2)
        if not np.isfinite([u, v]).all() or not (0 <= u < 1280 and 0 <= v < 720):
            return None
        return int(round(u)), int(round(v))

    def plan_ready(self, raw, cancelled=lambda: False):
        from .sdk_transit import ready_plan
        return ready_plan(self, raw, cancelled)

    def plan_home(self, raw, cancelled=lambda: False):
        from .sdk_transit import home_plan
        return home_plan(self, raw, cancelled)


def freeze_click(planner, point_camera, raw, hover_m, *, table_z_m):
    """Keep clicked XY; Z is always the saved horizontal table + hover height."""
    from eyehand.table_reference import table_hover_target
    return table_hover_target(planner.camera_pose(raw), point_camera, hover_m, table_z_m=table_z_m)

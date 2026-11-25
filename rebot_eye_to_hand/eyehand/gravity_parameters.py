"""Independent gravity parameter API; never reads the Demo 10/SDK profile."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as et

import numpy as np
DEFAULT_GRAVITY_CONFIG = Path(__file__).resolve().parents[1] / "gravity_config.json"
PARAMETER_KEYS = ("tau_scale", "kp", "kd", "linear_release_threshold", "angular_release_threshold",
                  "payload_mass_kg", "payload_frame", "payload_com_m",
                  "wrist_camera_mass_kg", "wrist_camera_frame", "wrist_camera_com_m")


def _validate_point_mass(settings, prefix, default_frame):
    mass = float(settings.get(f"{prefix}_mass_kg", 0.0))
    com = np.asarray(settings.get(f"{prefix}_com_m", [0., 0., 0.]), dtype=float)
    frame = settings.get(f"{prefix}_frame", default_frame)
    if not np.isfinite(mass) or mass < 0:
        raise ValueError(f"{prefix}_mass_kg must be finite and nonnegative")
    if com.shape != (3,) or not np.isfinite(com).all():
        raise ValueError(f"{prefix}_com_m must contain three finite coordinates in metres")
    if not isinstance(frame, str) or not frame.strip():
        raise ValueError(f"{prefix}_frame must be a nonempty frame name")
    return {f"{prefix}_mass_kg": mass, f"{prefix}_frame": frame, f"{prefix}_com_m": com.tolist()}


def validate_gravity_parameters(settings):
    scale = np.asarray(settings["tau_scale"], dtype=float)
    if scale.ndim == 0:
        scale = np.full(6, float(scale))
    if scale.shape != (6,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("tau_scale must contain six positive finite values")
    values = {key: float(settings[key]) for key in [
        "kp", "kd", "linear_release_threshold", "angular_release_threshold"]}
    if not all(np.isfinite(v) for v in values.values()) or values["kp"] <= 0 or values["kd"] < 0:
        raise ValueError("Invalid gravity gains")
    if min(values["linear_release_threshold"], values["angular_release_threshold"]) <= 0:
        raise ValueError("Invalid gravity release thresholds")
    return {**values, "tau_scale": scale.tolist(),
            **_validate_point_mass(settings, "payload", "gripper_end"),
            **_validate_point_mass(settings, "wrist_camera", "link6")}


def load_gravity_parameters(config_path=DEFAULT_GRAVITY_CONFIG):
    path = Path(config_path).expanduser().resolve()
    raw = path.read_bytes()
    parameters = validate_gravity_parameters(json.loads(raw))
    return {**parameters, "config_path": str(path), "config_sha256": hashlib.sha256(raw).hexdigest(),
            "scope": "ArUco and wrist camera point masses added separately before gravity scaling; original integral/lock law retained"}


def set_gravity_parameters(config_path=DEFAULT_GRAVITY_CONFIG, **updates):
    """Validate and persist local parameters. Does not contact a running robot."""
    unknown = set(updates) - set(PARAMETER_KEYS)
    if unknown:
        raise ValueError(f"Unknown gravity parameters: {sorted(unknown)}")
    path = Path(config_path).expanduser().resolve()
    current = json.loads(path.read_text())
    current.update(updates)
    parameters = validate_gravity_parameters(current)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(parameters, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return load_gravity_parameters(path)


def _apply_payload_model(controller, parameters):
    """Add two point masses to a private model; reconfiguration never stacks mass.

    The payload COM is expressed in payload_frame, whose placement is relative
    to its parent joint. Kinematics and the URDF on disk remain unchanged.
    """
    if parameters["payload_mass_kg"] == 0 and parameters["wrist_camera_mass_kg"] == 0 and not hasattr(controller, "_model_without_payload"):
        return
    import pinocchio as pin

    bare_model = getattr(controller, "_model_without_payload", controller._model)
    model = bare_model.copy()
    # A later switch to the main project's camera-equipped URDF must not add
    # another 160 g. Refuse ambiguous duplicate mass rather than silently skip.
    if parameters["wrist_camera_mass_kg"] > 0 and any(
            model.existFrame(name) for name in ("camera_mount_link", "camera_link")):
        raise ValueError("Model already has wrist-camera frames; verify its inertias and set wrist_camera_mass_kg=0 if camera mass is already included")
    for prefix in ("payload", "wrist_camera"):
        mass = parameters[f"{prefix}_mass_kg"]
        if mass == 0:
            continue
        frame_name = parameters[f"{prefix}_frame"]
        frame_id = model.getFrameId(frame_name)
        if frame_id >= model.nframes:
            raise ValueError(f"Unknown payload frame: {frame_name}")
        frame = model.frames[frame_id]
        if frame.parentJoint == 0:
            raise ValueError("Payload frame must be attached to a moving joint")
        inertia = pin.Inertia(mass, np.asarray(parameters[f"{prefix}_com_m"]), np.zeros((3, 3)))
        model.appendBodyToJoint(frame.parentJoint, inertia, frame.placement)
    controller._model_without_payload = bare_model
    controller._model = model
    controller._data = model.createData()


def validate_gravity_link_mass_reductions(values=None):
    """Measured removable finger portions, not complete gripper link removal."""
    values={} if values is None else values
    if not isinstance(values,dict) or not set(values)<= {'gripper_left','gripper_right'}:
        raise ValueError('Mass reduction only supports removable finger portions')
    result={}
    for name,value in values.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
            raise ValueError('Removed finger mass must be positive finite kg')
        result[name]=float(value)
    return result


def configure_gravity_model_exclusions(controller,urdf,links,link_mass_reductions=None):
    """Exclude absent camera inertias or measured finger mass from private gravity.

    Reused by collection and the external ArUco demo. The URDF on disk and
    the calibrated model used for FK/IK remain unchanged.
    """
    if (not isinstance(links,list) or any(not isinstance(name,str) for name in links) or
            len(set(links))!=len(links) or
            not set(links)<= {'camera_mount_link','camera_link'}):
        raise ValueError('Only absent wrist-camera URDF links may be excluded from gravity')
    reductions=validate_gravity_link_mass_reductions(link_mass_reductions)
    controller.gravity_exclusion=dict(links=list(links),removed_mass_kg=0.,
        scope='Private gravity model only; original URDF and nominal FK unchanged')
    if not links and not reductions:return
    if links and controller.gravity['wrist_camera_mass_kg']!=0:
        raise ValueError('Removed-camera gravity model requires zero added wrist-camera mass')
    from .kinematics import JOINT_NAMES
    pin=controller.pin
    tree=et.fromstring(Path(urdf).read_bytes());removed=0.
    for name in links:
        link=tree.find(f"link[@name='{name}']")
        if link is None:raise ValueError('Unknown excluded gravity link: '+name)
        inertial=link.find('inertial')
        if inertial is None:raise ValueError('Excluded gravity link has no inertia: '+name)
        mass=float(inertial.find('mass').get('value'))
        if not np.isfinite(mass) or mass<0:raise ValueError('Invalid excluded link mass')
        removed+=mass;link.remove(inertial)
    for name,amount in reductions.items():
        link=tree.find(f"link[@name='{name}']")
        if link is None:raise ValueError('Unknown reduced gravity link: '+name)
        inertial=link.find('inertial')
        if inertial is None:raise ValueError('Reduced gravity link has no inertia: '+name)
        mass_node=inertial.find('mass');mass=float(mass_node.get('value'))
        if not np.isfinite(mass) or amount>=mass:
            raise ValueError('Measured removed mass must be below the URDF link mass: '+name)
        # The removed distal parts have not had their COM measured. Approximate
        # at the existing link COM, retaining the slider/carriage and frames.
        factor=(mass-amount)/mass;mass_node.set('value',str(mass-amount))
        inertia=inertial.find('inertia')
        for key,value in list(inertia.attrib.items()):inertia.set(key,str(float(value)*factor))
        removed+=amount
    if reductions:
        controller.gravity_exclusion.update(link_mass_reductions_kg=reductions,
            removed_mass_com_approximation='existing URDF link COM; inertia scaled proportionally')
    xml=et.tostring(tree,encoding='unicode')
    full=pin.buildModelFromXML(xml)
    lock=[i for i in range(1,full.njoints) if full.names[i] not in JOINT_NAMES]
    bare=pin.buildReducedModel(full,lock,pin.neutral(full))
    if bare.nq!=6 or list(bare.names[1:])!=JOINT_NAMES:
        raise ValueError('Excluded gravity model changed the six arm joints')
    controller._model_without_payload=bare
    controller._model=bare;controller._data=bare.createData()
    _apply_payload_model(controller,controller.gravity)
    controller.gravity_model,controller.gravity_data=controller._model,controller._data
    controller.sdk_law.model,controller.sdk_law.data=controller.gravity_model,controller.gravity_data
    controller.gravity_exclusion.update(removed_mass_kg=removed,
        private_urdf_xml_sha256=hashlib.sha256(xml.encode()).hexdigest())


def apply_gravity_parameters(controller, parameters=None):
    parameters = load_gravity_parameters() if parameters is None else parameters
    parameters = {**parameters, **validate_gravity_parameters(parameters)}
    _apply_payload_model(controller, parameters)
    controller.KP = parameters["kp"]
    controller.KD = parameters["kd"]
    controller.V_THRESH = parameters["linear_release_threshold"]
    controller.W_THRESH = parameters["angular_release_threshold"]
    scale = np.asarray(parameters["tau_scale"], dtype=float)
    # Change this controller instance only. Re-applying must not multiply twice.
    unscaled = getattr(controller, "_unscaled_gravity", controller._compute_gravity)
    controller._unscaled_gravity = unscaled

    def scaled_gravity(*args, **kwargs):
        torque = np.asarray(unscaled(*args, **kwargs), dtype=float).copy()
        if torque.ndim != 1 or len(torque) < 6 or not np.isfinite(torque).all():
            raise ValueError("Invalid model gravity torque")
        torque[:6] *= scale
        return torque

    controller._compute_gravity = scaled_gravity
    return parameters

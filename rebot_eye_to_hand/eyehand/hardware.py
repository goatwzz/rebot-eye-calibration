from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .geometry import checked

# The legacy collector keeps its original SDK after project relocation. The
# click-hover program selects the modern SDK separately in click_hover_config.json.
DEFAULT_SDK = Path(os.environ.get("REBOT_LEGACY_SDK",
    str(Path(__file__).resolve().parents[2] / "vendor/sdks/legacy")))


def write_pose(path, t_base_ee, *, base_frame="base_link", ee_frame="gripper_end", acquired_unix_s):
    """Call in the existing controller AFTER a successful fresh feedback read + FK.

    acquired_unix_s must be the feedback acquisition time, never a heartbeat
    timestamp wrapped around an old cached pose. Does not access hardware.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"T_base_ee": checked(t_base_ee).tolist(), "timestamp_unix_s": acquired_unix_s,
               "base_frame": base_frame, "ee_frame": ee_frame}
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(target)


class PoseFile:
    def __init__(self, path, base_frame, ee_frame, max_age=0.3):
        self.path, self.base_frame, self.ee_frame, self.max_age = Path(path), base_frame, ee_frame, max_age
        self.metadata = {"source": "pose-file", "base_frame": base_frame, "ee_frame": ee_frame}

    def read(self):
        value = json.loads(self.path.read_text())
        age = time.time() - value["timestamp_unix_s"]
        if not 0 <= age <= self.max_age:
            raise ValueError(f"Robot pose stale or clock mismatch: age={age:.3f}s")
        for key in ["base_frame", "ee_frame"]:
            if value[key] != getattr(self, key):
                raise ValueError(f"Pose file {key} mismatch")
        return checked(value["T_base_ee"]), {"feedback_unix_s": value["timestamp_unix_s"], "age_s": age}

    def close(self):
        pass


class RebotReader:
    """RS-only parameter queries. No RebotArm connect/disconnect or mode commands.

    Independent socket client: controller must already hold the robot steady.
    Prefer PoseFile if the existing controller also reads parameter responses.
    """
    def __init__(self, sdk=DEFAULT_SDK, channel=None, ee_frame="gripper_end", base_frame="base_link"):
        import pinocchio as pin
        import yaml
        from .native_runtime import activate
        activate()
        from motorbridge import Controller

        sdk = Path(sdk).resolve()
        config = yaml.safe_load((sdk / "config/rebotarm_rs.yaml").read_text())
        urdf = sdk / config["urdf_path"]
        self.pin, self.model = pin, pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(ee_frame)
        if self.frame_id >= self.model.nframes:
            raise ValueError(f"Unknown end frame: {ee_frame}")
        # Pinocchio FK is relative to the URDF root, not a scene/world frame.
        import xml.etree.ElementTree as et
        root = et.parse(urdf).getroot()
        children = {n.get("link") for n in root.findall("joint/child")}
        roots = {n.get("name") for n in root.findall("link")} - children
        if roots != {base_frame}:
            raise ValueError(f"URDF root is {roots}, requested {base_frame}")
        self.ctrl = Controller(channel or config["channel"])
        self.motors = []
        self.joint_indices = []
        try:
            names = config["groups"]["arm"]["joints"]
            if names != [f"joint{i}" for i in range(1, 7)]:
                raise ValueError("RS reader requires joint1..joint6 in order (matches gain/torque limit order)")
            for name in names:
                jc = next(j for j in config["joints"] if j["name"] == name)
                if jc["vendor"] != "robstride":
                    raise ValueError("This reader supports only RobStride RS motors")
                jid = self.model.getJointId(name)
                if jid >= self.model.njoints or self.model.joints[jid].nq != 1:
                    raise ValueError(f"Unexpected joint model: {name}")
                self.joint_indices.append(self.model.joints[jid].idx_q)
                self.motors.append(self.ctrl.add_robstride_motor(jc["motor_id"], jc["feedback_id"], jc["model"]))
            if len(names) != 6:
                raise ValueError("Expected six arm joints")
        except Exception:
            self.close()
            raise
        self.metadata = {"source": "robstride-mechPos-0x7019", "base_frame": base_frame,
                         "ee_frame": ee_frame, "channel": channel or config["channel"],
                         "urdf": str(urdf), "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
                         "joint_names": names}

    def wait_for_feedback(self, timeout_s=15.):
        """Require two consecutive full, fresh reads before any activation.

        This only queries mechPos. Startup may tolerate a temporarily unready
        link; the runtime read() timeout and freshness rules are unchanged.
        Call the base reader explicitly because ManualGravityReader.read()
        exposes a snapshot only after its control thread has started.
        """
        if not 0 < timeout_s <= 60:
            raise ValueError("Feedback startup timeout must be in (0,60] seconds")
        deadline = time.monotonic() + timeout_s
        consecutive = 0
        last_error = "No complete feedback read"
        print(f"Waiting for six-joint feedback, up to {timeout_s:g}s; motors NOT enabled.", flush=True)
        while time.monotonic() < deadline:
            try:
                pose, info = RebotReader.read(self)
                consecutive += 1
                if consecutive >= 2:
                    elapsed = info["read_ended_monotonic_s"] - info["read_started_monotonic_s"]
                    print(f"Feedback ready: two consecutive six-joint reads; last read {elapsed * 1000:.1f}ms.", flush=True)
                    return pose, info
            except Exception as exc:
                consecutive = 0
                last_error = str(exc)
                print(f"Feedback not ready: {last_error}", flush=True)
            time.sleep(0.05)
        raise RuntimeError(f"Six-joint feedback not ready within {timeout_s:g}s; motors NOT enabled. "
                           f"Last error: {last_error}")

    def read(self):
        start = time.time()
        monotonic_start = time.monotonic()
        # Never substitute zeros or cached states when a register read fails.
        values = []
        for name, motor in zip(self.metadata["joint_names"], self.motors):
            try:
                values.append(motor.robstride_get_param_f32(0x7019, timeout_ms=150))
            except Exception as exc:
                raise RuntimeError(f"{name} position read failed on {self.metadata['channel']}: {exc}") from exc
        values = np.asarray(values, dtype=float)
        end = time.time()
        monotonic_end = time.monotonic()
        if monotonic_end - monotonic_start > 0.3 or not np.isfinite(values).all():
            raise ValueError("Joint read invalid or too slow for stationary sampling")
        q = self.pin.neutral(self.model)
        q[self.joint_indices] = values
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        return checked(self.data.oMf[self.frame_id].homogeneous), {
            "q_rad": values.tolist(), "read_started_unix_s": start, "read_ended_unix_s": end,
            "read_started_monotonic_s": monotonic_start, "read_ended_monotonic_s": monotonic_end}

    def close(self):
        if getattr(self, "ctrl", None) is not None:
            # close_bus/free only; do not call shutdown(), disable_all(), or SDK disconnect().
            try:
                self.ctrl.close_bus()
            finally:
                self.ctrl.close()
                self.ctrl = None

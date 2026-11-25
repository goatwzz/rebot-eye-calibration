"""Shared original gravity controller plus a separate fresh-feedback reader.

Normal module imports replace dynamic AST compilation. The controller lives in
sibling rebot_eye_in_hand/robot; SDK motor control and the gravity law are reused.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .hardware import RebotReader, DEFAULT_SDK
from .gravity_parameters import DEFAULT_GRAVITY_CONFIG, apply_gravity_parameters, load_gravity_parameters

DEFAULT_REFERENCE = Path(__file__).resolve().parents[2] / "rebot_eye_in_hand/robot/gravity_controller.py"


def load_reference_class(path=DEFAULT_REFERENCE):
    path = Path(path).resolve()
    # Old --reference-script paths may now point to the compatibility launcher.
    if path.name == "collect_handeye_eih.py":
        candidate = path.parents[1] / "robot/gravity_controller.py"
        if candidate.is_file():
            path = candidate
    spec = importlib.util.spec_from_file_location("rebot_shared_gravity", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load gravity controller: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GravityCompController, hashlib.sha256(path.read_bytes()).hexdigest()


class ReferenceManualReader(RebotReader):
    uses_explicit_modes = False

    def __init__(self, *, reference_script=DEFAULT_REFERENCE, channel=None,
                 ee_frame="gripper_end", base_frame="base_link", gravity_config=DEFAULT_GRAVITY_CONFIG,
                 sdk_root=None):
        if not sys.stdin.isatty():
            raise ValueError("--manual requires an interactive terminal")
        self._active = False
        self.command_queue = None
        # Select the checked-out dependency without requiring a global SDK install.
        from .native_runtime import activate
        activate()
        if "reBotArm_control_py" not in sys.modules:
            sys.path.insert(0, str(Path(sdk_root or DEFAULT_SDK).resolve()))
        import reBotArm_control_py
        from reBotArm_control_py.actuator import RebotArm
        import yaml

        sdk = Path(reBotArm_control_py.__file__).resolve().parent.parent
        global_config = yaml.safe_load((sdk / "config/rebotarm.yaml").read_text())
        if Path(global_config["hardware_yaml"]).name != "rebotarm_rs.yaml":
            raise ValueError(f"Original command's SDK is not configured for RS: {sdk}")
        self.arm = RebotArm()
        actual_channel = self.arm._channel
        if channel is not None and channel != actual_channel:
            raise ValueError(f"Original SDK uses {actual_channel}; --channel {channel} would differ")
        cls, digest = load_reference_class(reference_script)
        self.controller = cls(self.arm)
        self.gravity_parameters = apply_gravity_parameters(self.controller, load_gravity_parameters(gravity_config))
        super().__init__(sdk, actual_channel, ee_frame, base_frame)
        self.metadata.update(source="shared-gravity-controller-plus-readonly-mechPos",
                             reference_script=str(Path(reference_script).resolve()),
                             reference_controller_sha256=digest,
                             sdk_path=str(sdk), sdk_control_rate_hz=float(self.arm.rate),
                             control_algorithm="Shared original integral/velocity-lock law; SDK mode verification checked; private payload model and independent gravity settings",
                             gravity_parameters=self.gravity_parameters,
                             calibration_reader="Separate read-only controller; no register reads in SDK control callback",
                             gripper_controlled=True, automatic_homing=False)

    def start(self):
        print("[Gravity parameters] " + json.dumps(self.gravity_parameters, ensure_ascii=False), flush=True)
        print("REFERENCE MANUAL: using original GravityCompController and original installed SDK.\n"
              "Stop other controllers and support the arm during startup.\n"
              "Drag by hand; the original velocity lock follows movement and locks when you stop.\n"
              "Enter/s=capture; pos=FK; q/c=finish. Gripper control follows the original script.\n"
              "Exit does NOT auto-home: support the arm, then type release to disable.", flush=True)
        self._active = True  # original startup can enable before a later error
        self.controller.start()

    @property
    def status(self):
        return "REFERENCE MANUAL: automatic velocity lock"

    def read(self):
        if not self._active:
            raise RuntimeError("Reference controller has not started")
        if not self.arm.control_loop_active:
            raise RuntimeError("Original SDK control loop is not running; sampling refused")
        # The original callback remains unchanged and owns its own bus/SDK.
        # Calibration reads do not hold its I/O lock or block that callback.
        pose, info = super().read()
        return pose, {**info, "manual_mode": "original_auto_velocity_lock"}

    def release_supported(self):
        try:
            self.controller._gc_running.clear()
            self.arm.stop_control_loop()
            self.arm.disconnect()  # original SDK disables arm AND gripper
        finally:
            self._active = False
            super().close()

    def close(self):
        if not self._active:
            super().close()
            return
        running = self.arm.control_loop_active
        state = "Gravity loop is running." if running else "Gravity loop is NOT running; holding is not established."
        print(f"\nCollection ended. {state}\n"
              "No automatic home. Support arm AND gripper/target, then type release to disable and exit.", flush=True)
        while True:
            try:
                command = input() if self.command_queue is None else self.command_queue.get(timeout=.2)
                if command is None:
                    raise EOFError
                if command.strip().lower() == "release":
                    self.release_supported()
                    return
                print("Support arm/target, then type release. No automatic homing.", flush=True)
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                print("Ctrl+C does not disable; support arm/target and type release.", flush=True)
            except EOFError:
                print("Terminal closed; use physical support/stop for the robot.", flush=True)
                time.sleep(1.)

#!/usr/bin/env python3
"""Set explicit SDK paths and camera serials in this checkout; never accesses hardware."""
import argparse
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def update(path, **values):
    data = json.loads(path.read_text())
    data.update(values)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modern-sdk', type=Path, default=ROOT/'vendor/sdks/modern')
    parser.add_argument('--legacy-sdk', type=Path, default=ROOT/'vendor/sdks/legacy')
    parser.add_argument('--wrist-serial')
    parser.add_argument('--external-serial')
    args = parser.parse_args()
    modern, legacy = args.modern_sdk.resolve(), args.legacy_sdk.resolve()
    for sdk in (modern, legacy):
        if not (sdk/'reBotArm_control_py/actuator/rebotarm.py').is_file():
            parser.error(f'SDK missing: {sdk}; run bootstrap_sdks.py first')
    wrist = ROOT/'rebot_eye_in_hand'; external = ROOT/'rebot_eye_to_hand'
    update(wrist/'config/click_hover.json', sdk_root=str(modern),
           urdf=str(legacy/'urdf/RS/urdf/ReBot_Arm_RS.urdf'))
    for name in ('click_hover_config.json', 'aruco_corners_control_config.json'):
        update(external/name, sdk_root=str(modern))
    for name in ('auto_collection_config.json', 'auto_validation_config.json'):
        update(external/name, urdf=str(modern/'urdf/RS/urdf/ReBot_Arm_RS.urdf'))
    for name in ('auto_collection.json', 'low_pose_validation.json'):
        update(wrist/'config'/name, urdf=str(legacy/'urdf/RS/urdf/ReBot_Arm_RS.urdf'))
    defaults = wrist/'config/default.yaml'; data = yaml.safe_load(defaults.read_text())
    data['paths']['sdk'] = str(legacy)
    if args.wrist_serial:
        data['camera']['serial'] = args.wrist_serial
        for name in ('click_hover.json','auto_collection.json','low_pose_validation.json'):
            update(wrist/'config'/name, serial=args.wrist_serial)
    defaults.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    if args.external_serial:
        for name in ('click_hover_config.json','aruco_corners_control_config.json',
                     'auto_collection_config.json','auto_validation_config.json','camera_realsense.json'):
            update(external/name, serial=args.external_serial)
    print('Local configuration updated. Existing result/intrinsics camera identities were NOT changed.')
    print('A new camera/mount needs its own intrinsics, calibration, and independent validation.')


if __name__ == '__main__':
    main()

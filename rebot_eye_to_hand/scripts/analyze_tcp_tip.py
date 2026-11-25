#!/usr/bin/env python3
"""Inspect URDF finger collision STL distal planes; no camera or robot access."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin


def binary_stl_vertices(path):
    raw = Path(path).read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0]
    if len(raw) != 84+50*count:
        raise ValueError('Expected binary STL')
    dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3,3)), ('attribute', '<u2')])
    return np.unique(np.frombuffer(raw, dtype=dtype, offset=84)['vertices'].reshape(-1,3), axis=0)


def analyze(urdf):
    urdf = Path(urdf).resolve()
    xml = ET.parse(urdf).getroot()
    # Derive the extension direction from the wrist motor to the tool frame.
    # A coordinate minimum is NOT necessarily the fingertip: here the motor is
    # at local X=-166.21 mm, the wide slider/rail at X=-88.61 mm, and tips at X=0.
    mount = xml.find("joint[@name='j_gripper_end']")
    if mount is None or mount.find('parent').get('link') != 'link6':
        raise ValueError('Expected gripper_end fixed to wrist motor link6')
    origin = mount.find('origin')
    T_motor_tool = pin.SE3(pin.rpy.rpyToMatrix(np.fromstring(origin.get('rpy'), sep=' ')),
                          np.fromstring(origin.get('xyz'), sep=' '))
    motor_in_tool = T_motor_tool.inverse().translation
    if np.linalg.norm(motor_in_tool) < 1e-6:
        raise ValueError('Cannot establish fingertip direction from wrist mount')
    outward = -motor_in_tool/np.linalg.norm(motor_in_tool)
    fingers = []
    for side, joint_name in [('left','gripper_joint1'), ('right','gripper_joint2')]:
        joint = xml.find(f"joint[@name='{joint_name}']")
        origin = joint.find('origin')
        xyz = np.fromstring(origin.get('xyz'), sep=' ')
        rpy = np.fromstring(origin.get('rpy'), sep=' ')
        collision = xml.find(f"link[@name='gripper_{side}']/collision")
        co = collision.find('origin')
        mesh = collision.find('geometry/mesh')
        path = (urdf.parent/mesh.get('filename')).resolve()
        p = binary_stl_vertices(path) * np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
        p = p @ pin.rpy.rpyToMatrix(np.fromstring(co.get('rpy','0 0 0'),sep=' ')).T + np.fromstring(co.get('xyz','0 0 0'),sep=' ')
        p = p @ pin.rpy.rpyToMatrix(rpy).T + xyz
        extension = p @ outward
        cap = p[extension > extension.max()-5e-5]
        centre = (cap.min(0)+cap.max(0))/2
        fingers.append({'side': side, 'mesh': str(path), 'mesh_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'bounds_gripper_end_m': [p.min(0).tolist(),p.max(0).tolist()],
                        'distal_x_m': float(centre[0]), 'distal_cap_bbox_center_m': centre.tolist(),
                        'distal_extension_m':float(extension.max())})
    offset = np.mean([f['distal_cap_bbox_center_m'] for f in fingers], axis=0)
    # CAD roundoff is sub-micrometre; don't turn it into a physical TCP claim.
    offset[np.abs(offset) < 1e-6] = 0.
    return {'urdf': str(urdf), 'urdf_sha256': hashlib.sha256(urdf.read_bytes()).hexdigest(), 'fingers': fingers,
            'wrist_motor_center_gripper_end_m':motor_in_tool.tolist(),
            'fingertip_outward_axis_gripper_end':outward.tolist(),
            'tcp_definition': 'Midpoint between distal fingertip ends at symmetric opening; outward from wrist motor, local +X nose near gripper_end origin',
            'recommended_tcp_offset_m': np.round(offset,7).tolist(),
            'supersedes': 'Previous analysis incorrectly selected local minimum X (slider/rail at -88.6096 mm) as the distal end',
            'assumption': 'Symmetric finger travel and original modeled fingers. No mapping of gripper motor angle to finger travel is present in URDF. Tips are not a single contact point.',
            'physical_tcp_verified': False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf',default=str(Path(__file__).resolve().parents[2]/'vendor/sdks/legacy/urdf/RS/urdf/ReBot_Arm_RS.urdf'))
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists():
        parser.error('Output exists')
    report=analyze(args.urdf)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

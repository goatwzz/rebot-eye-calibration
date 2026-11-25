#!/usr/bin/env python3
"""Fetch pinned official SDKs, apply recorded compatibility changes; no hardware I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run(list(map(str, args)), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modern-source', help='Optional local official SDK Git checkout (offline)')
    parser.add_argument('--legacy-source', help='Optional local official SDK Git checkout (offline)')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'vendor/sdk-manifest.json').read_text())
    for name, spec in manifest['sdks'].items():
        target = ROOT/'vendor/sdks'/name
        if target.exists():
            raise SystemExit(f'{target} exists; leave it intact. Use an empty checkout for bootstrap.')
        source = getattr(args, name+'_source') or manifest['repository']
        target.parent.mkdir(parents=True, exist_ok=True)
        run('git', 'clone', '--no-checkout', '--no-hardlinks', source, target)
        run('git', '-C', target, 'checkout', '--detach', spec['commit'])
        patch = ROOT/spec['patch']
        if hashlib.sha256(patch.read_bytes()).hexdigest() != spec['patch_sha256']:
            raise SystemExit('SDK patch checksum mismatch')
        if patch.stat().st_size:
            run('git', '-C', target, 'apply', '--check', patch)
            run('git', '-C', target, 'apply', patch)
        for rel in spec['overlay_files']:
            src = ROOT/'vendor/sdk-overlays'/name/rel
            dst = target/rel
            if dst.exists():
                raise SystemExit(f'Overlay would replace an upstream file: {dst}')
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        urdf = target/'urdf/RS/urdf/ReBot_Arm_RS.urdf'
        if hashlib.sha256(urdf.read_bytes()).hexdigest() != spec['urdf_sha256']:
            raise SystemExit(f'{name} URDF differs from the recorded model')
        print(f'[OK] {name}: {spec["commit"]}; model checksum verified')
    print('SDKs ready. No SDK connect/enable method was called.')


if __name__ == '__main__':
    main()

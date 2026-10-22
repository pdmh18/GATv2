#!/usr/bin/env python3
"""Remove reproducible dataset payloads, preserving checkpoints and reports."""
import argparse
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def candidates(root):
    paths=[]
    for run in root.iterdir() if root.exists() else ():
        if not run.is_dir() or run.is_symlink():continue
        if run.name=='archive':
            for name in ('data','migration-originals'):
                if (run/name).is_dir():paths.append(run/name)
            continue
        for seed in run.glob('seed-*'):
            if not seed.is_dir() or seed.is_symlink():continue
            for name in ('raw','splits','features','features-v3'):
                if (seed/name).is_dir():paths.append(seed/name)
            paths.extend(p for p in seed.glob('features.previous-*') if p.is_dir())
            paths.extend(p for p in seed.rglob('contexts') if p.is_dir())
            paths.extend(p for p in seed.rglob('diagnostic_arrays.npz') if p.is_file())
    return sorted(set(paths))


def size(path):
    files=[path] if path.is_file() else path.rglob('*')
    return sum(p.stat().st_size for p in files if p.is_file() and not p.is_symlink())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--apply',action='store_true',help='Delete payloads; default prints a preview')
    args=parser.parse_args()
    root=ROOT/'outputs';paths=candidates(root);total=0
    for path in paths:
        if path.is_symlink() or root.resolve() not in path.resolve().parents:
            raise ValueError('Refusing cleanup outside outputs')
        amount=size(path);total+=amount
        print(f'{amount/1024**3:.3f} GiB  {path.relative_to(ROOT)}')
    if args.apply:
        receipt={'removed':[str(p.relative_to(ROOT)) for p in paths], 'payload_bytes':total,
                 'preserved':'Checkpoints, reports, figures, saved evaluation predictions. Raw/splits/features/context caches must be regenerated before further training.'}
        for path in paths:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        root.mkdir(exist_ok=True)
        (root/'cleanup.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(f'{"Removed" if args.apply else "Would remove"}: {total/1024**3:.3f} GiB')


if __name__=='__main__':main()

"""Apply the reference-standard finish to the existing 25 results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'common')]
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '8'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--refresh-existing', action='store_true')
    args = parser.parse_args()
    from utils.animation_parts import load_layers
    from utils.animation_clean import export_compact
    from utils.animation_finish import (read_snapshot, read_delivery, finish, add_whole_alternatives,
                                        clean_watermarks, repair_engineer_right_boot, finish_engineer_reference,
                                        repair_engineer_mechanical_root, ENGINEER)
    source = ROOT/'立绘测试结果_OursFull_20261027'
    output = ROOT/'立绘拆分结果'
    with zipfile.ZipFile(ROOT/'workspace/finish_audit/before_finish.zip') as archive:
        names = sorted({p.split('/')[0] for p in archive.namelist()})
        if args.only:
            names = [n for n in names if n.startswith(tuple(args.only))]
        for i, name in enumerate(names, 1):
            print('FINISH_START', i, len(names), name, flush=True)
            if args.refresh_existing:
                layers, info = read_delivery(output/name)
                layers = [p for p in layers if p.meta.get('default_visible', True)]
                if name == 'f8bdb4d2d17cc6a56bbeb911b2864245':
                    layers = clean_watermarks(name, layers)
                if name == ENGINEER:
                    _, originals = load_layers(source/name)
                    original = next(p for p in originals if p.name == 'legwear')
                    repair_engineer_right_boot(layers, info, original)
                    finish_engineer_reference(layers, info, original)
                    handwear = next(p for p in originals if p.name == 'handwear')
                    repair_engineer_mechanical_root(layers, handwear)
                layers = add_whole_alternatives(layers, info)
                export_compact(layers, output/name, info)
                print('REFRESH_DONE', name, flush=True)
                continue
            layers, info = read_snapshot(archive, name)
            _, originals = load_layers(source/name)
            layers, info = finish(name, layers, info, originals)
            export_compact(layers, output/name, info)
            print('FINISH_DONE', i, len(names), name, 'layers', len(layers), flush=True)


if __name__ == '__main__':
    main()

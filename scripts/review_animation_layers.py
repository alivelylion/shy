"""Apply reviewed repairs to existing compact results, without rerunning diffusion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'common')]
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='8'


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'立绘拆分结果')
    parser.add_argument('--only',nargs='+',required=True)
    parser.add_argument('--no-lama',action='store_true')
    args=parser.parse_args()
    import numpy as np
    from PIL import Image
    from utils.animation_finish import read_delivery
    from utils.animation_clean import export_compact
    from utils.animation_review import apply_reviewed_repairs
    for name in args.only:
        folder=args.output_dir/name
        layers,info=read_delivery(folder)
        source=ROOT/'立绘测试结果_OursFull_20261027'/name/'src_img.png'
        src=np.array(Image.open(source).convert('RGBA'))
        print('REVIEW_START',name,flush=True)
        layers=apply_reviewed_repairs(name,src,layers,not args.no_lama)
        names={p.name for p in layers}
        for key in ('arms','legs'):
            info[key]=[r for r in info.get(key,[]) if all(p in names for p in r.get('parts',[]))]
        info['whole_limb_layers']=[p.name for p in layers if p.role in {'arm_whole','leg_whole'}]
        info['reviewed_semantic_cleanup']='source edges, facial underpainting, true garment/prop ownership and complete ground shadow'
        export_compact(layers,folder,info)
        print('REVIEW_DONE',name,'layers',len(layers),flush=True)


if __name__=='__main__':
    main()

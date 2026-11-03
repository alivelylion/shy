"""Update the existing delivery with reviewed per-character anatomy."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import json
from io import BytesIO
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'common')]
os.environ['TORCH_HOME'] = str(ROOT/'workspace/.cache/torch')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '8'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--no-lama', action='store_true')
    args = parser.parse_args()
    from PIL import Image
    import numpy as np
    from psd_tools import PSDImage
    from utils.animation_parts import Layer, composite
    from utils.animation_finish import read_delivery, add_whole_alternatives, ENGINEER
    from utils.animation_clean import export_compact
    from utils.animation_character_repairs import rebuild_pikachu, rebuild_doraemon
    from utils.animation_detail_repairs import rebuild_seated_cat, repair_satchel
    from utils.animation_standard import (split_farm_sheet, split_configured,
        split_pixel_sheet, split_paired_existing, separate_warrior_ribbon,
        finish_charizard, finish_humanoid_roots, REVISION)
    names = sorted(p.name for p in (ROOT/'立绘拆分结果').iterdir() if p.is_dir()) if args.all else args.only or ['45-meowa-watermark']
    archive = zipfile.ZipFile(ROOT/'workspace/reference_all_20261102/before.zip')
    def baseline(name: str) -> tuple[list[Layer], dict]:
        prefix=name+'/'
        info=json.loads(archive.read(prefix+'关节与图层.json'))
        psd=PSDImage.open(BytesIO(archive.read(prefix+'动画拆层.psd')))
        order={p.name:i for i,p in enumerate(psd)}; stored={p.name:p for p in psd}
        core={'name','file','source','role','pivot_xy','parent','bbox_xyxy'}
        layers=[]; step=max(1,240//len(psd))
        for e in info['layers']:
            rgba=np.array(Image.open(BytesIO(archive.read(prefix+e['file']))).convert('RGBA'))
            visibility=None; item=stored[e['name']]
            if item.has_mask():
                visibility=np.zeros(rgba.shape[:2],np.uint8)
                patch=np.array(item.mask.topil().convert('L')); y,x=item.mask.top,item.mask.left
                visibility[y:y+patch.shape[0],x:x+patch.shape[1]]=patch
            layers.append(Layer(e['name'],rgba,np.full(rgba.shape[:2],(len(psd)-1-order[e['name']])*step,np.uint8),
                e['source'],pivot=e.get('pivot_xy'),parent=e.get('parent'),role=e['role'],
                meta={k:v for k,v in e.items() if k not in core},visibility_mask=visibility))
        return layers,info
    for name in names:
        folder = ROOT/'立绘拆分结果'/name
        layers, info = baseline(name)
        layers=[p for p in layers if p.meta.get('default_visible',True)]
        src = np.array(Image.open(ROOT/'立绘测试结果_OursFull_20261027'/name/'src_img.png').convert('RGBA'))
        if name == '45-meowa-watermark':
            layers, info = split_farm_sheet(src, info, not args.no_lama)
        elif name=='078f35a9b6e4f0285f2d2b883e52d57d':
            layers,info=rebuild_seated_cat(src,layers,info,not args.no_lama)
        elif name=='62a3ea887f36c3d5ed44081cf4bdcb1b':
            layers,info=rebuild_doraemon(src,layers,info)
        elif name in {'025皮卡丘','025皮卡丘小智版'}:
            layers,info=rebuild_pikachu(src,layers,info,name)
        elif name=='sc_gen_7245-meowa-watermark':
            layers,info=split_pixel_sheet(src,layers,info,not args.no_lama)
        elif name=='JRPG-meowa-watermark':
            layers,info=split_configured(src,info,name,use_lama=not args.no_lama)
        elif name=='tangka':
            target=np.logical_or.reduce([p.mask for p in layers if p.name in {'topwear','objects'}]) & (src[...,3]>10)
            layers,info=split_configured(src,info,name,target,not args.no_lama)
        elif name=='303401_xingzhiyan_p':
            from utils.animation_standard import _keep_connected
            from utils.animation_parts import _complete_rgb
            from utils.animation_integrity import polygon
            cow=[p for p in layers if p.name.startswith('cow_')]
            for p in cow:_keep_connected(p)
            cow_mask=np.logical_or.reduce([p.mask for p in cow])
            layers,info=split_configured(src,info,name,(src[...,3]>10)&~cow_mask,not args.no_lama)
            beam=next(p for p in layers if p.name=='scene_beam')
            known=beam.mask.copy()
            support=known|(cow_mask&polygon(known.shape,[[434,329],[603,329],[671,762],[397,762]]))
            beam.rgba,beam.completed=_complete_rgb(src,known,support,not args.no_lama)
            beam.visibility_mask=known.astype(np.uint8)*255
            for p in cow:
                p.depth[:]=40 if p.role=='animal_head' else 50
            layers+=cow
        elif name=='16-1-meowa-watermark':
            layers=split_paired_existing(layers,'handwear',['arm_image_left','arm_image_right'],636,'clothing',
                                         ['画面左完整手臂','画面右完整手臂'],not args.no_lama)
            layers=split_paired_existing(layers,'legwear',['leg_image_left','leg_image_right'],620,'clothing',
                                         ['画面左腿','画面右腿'],not args.no_lama)
        elif name.startswith('006'):
            layers=split_paired_existing(layers,'legwear',['leg_image_left','leg_image_right'],675,'topwear',
                                         ['画面左腿','画面右腿'],not args.no_lama)
            layers,info=finish_charizard(src,layers,info,not args.no_lama)
        elif name=='05':
            layers=separate_warrior_ribbon(src,layers)
        if name!=ENGINEER and (name=='05' or name.startswith(('300','301','302','f8b','jimeng','sc_gen_45206'))):
            finish_humanoid_roots(layers,info)
        if name=='300602_xiaochunqueer_p':
            layers,info=repair_satchel(src,layers,info,not args.no_lama)
        layers=add_whole_alternatives(layers,info)
        if name!=ENGINEER:
            info.update({'standard_revision':REVISION,
                         'visibility_policy':'omit completely unseen parts; complete observed attachments only'})
        info['whole_limb_layers']=[p.name for p in layers if p.role in {'arm_whole','leg_whole'}]
        info['input']=str(ROOT/'立绘测试结果_OursFull_20261027'/name)
        export_compact(layers, folder, info)
        print('STANDARD_DONE', name, 'layers', len(layers), flush=True)


if __name__ == '__main__':
    main()

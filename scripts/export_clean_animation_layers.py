"""Build a new compact delivery from original Ours Full semantic layers."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='8'
sys.path[:0]=[str(ROOT),str(ROOT/'common')]


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--previous-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--only',nargs='+')
    args=parser.parse_args()
    source=args.input_dir.resolve(strict=True)
    previous=args.previous_dir.resolve(strict=True)
    output=args.output_dir.resolve()
    if output==source:
        parser.error('Keep the original semantic predictions in a separate directory')
    import numpy as np
    import cv2
    from PIL import Image
    from utils.animation_parts import load_layers,Layer,_remove_duplicate_cloth
    from utils.animation_integrity import read_hints,polygon
    from utils.animation_clean import split_existing_limb,split_reviewed_crossing_limbs,previous_chains,restore_named_regions,separate_selected_objects,remove_reviewed_duplicates,export_compact
    from utils.animation_special import split_knight_skirts,split_rear_car
    from utils.animation_review import apply_reviewed_repairs
    hints=json.loads((ROOT/'workspace/animation_joint_hints.json').read_text(encoding='utf-8'))
    extra_path=ROOT/'workspace/animation_clean_hints.json'
    overrides=json.loads(extra_path.read_text(encoding='utf-8')) if extra_path.is_file() else {}
    items=sorted(p for p in source.iterdir() if p.is_dir() and (p/'info.json').is_file())
    if args.only:
        items=[p for p in items if p.name.startswith(tuple(args.only))]
    for index,item in enumerate(items,1):
        print(f'CLEAN_START {index}/{len(items)} {item.name}',flush=True)
        src,originals=load_layers(item)
        layers=copy.deepcopy(originals)
        old=json.loads((previous/item.name/'关节与图层.json').read_text(encoding='utf-8'))
        settings={**hints.get(item.name,{}),**overrides.get(item.name,{})}
        info={'input':str(item),'method':'Ours Full original layers plus compact joint subdivision',
              'arms':[],'legs':[],'cleanup':[],'warnings':[],
              'subdivision_policy':'only clearly visible joints; short, round, occluded or skeleton-only estimates remain whole'}
        # Remove a repeated dress, without auto-splitting arbitrary fur/coat layers.
        garment=next((p for p in layers if p.name=='bottomwear'),None)
        if garment is not None:
            change=_remove_duplicate_cloth(layers,garment)
            if change:info['cleanup'].append(change)
        if item.name.startswith('jimeng-2025-09-20-6385'):
            cloth=[e for e in old['layers'] if e['role'] in {'cloth_inner','cloth_outer'} and e['source']=='bottomwear']
            if len(cloth)==2:
                layers=[p for p in layers if p.name!='bottomwear']
                for e in cloth:
                    rgba=np.array(Image.open(previous/item.name/e['file']).convert('RGBA'))
                    depth=np.array(Image.open(previous/item.name/f"{e['name']}_depth.png").convert('L'))
                    layers.append(Layer(e['name'],rgba,depth,'bottomwear',role=e['role'],pivot=e.get('pivot_xy')))
        for tag,key,kind in [('handwear','arms','arm'),('legwear','legs','leg')]:
            limb=next((p for p in layers if p.name==tag),None)
            chains=copy.deepcopy(settings.get(key,previous_chains(old,key)))
            if limb is None or not chains:continue
            allowed=settings.get('subdivide_arms' if kind=='arm' else 'subdivide_legs')
            if allowed is not None:
                for chain in chains:chain['subdivide']=allowed
            # Reuse original leg textures once, rather than adding a second shoe copy.
            if kind=='leg':
                shoe=next((p for p in layers if p.name=='footwear'),None)
                if shoe is not None:
                    yy,xx=np.indices(shoe.mask.shape)
                    xy=np.stack([xx,yy],axis=-1)
                    distal=np.zeros(shoe.mask.shape,bool)
                    for c in chains:
                        j=np.array(c['points'],float)
                        endpoint=j[-1] if len(j)>3 else j[2]+(j[2]-j[1])*.25
                        distal|=np.linalg.norm(xy-j[2],axis=-1)<max(25,np.linalg.norm(endpoint-j[2])*2)
                    useful=shoe.mask & distal & ~cv2.dilate(limb.mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
                    if settings.get('integrate_footwear') is False:
                        useful[:]=False
                    limb.rgba[useful]=shoe.rgba[useful]
                    limb.depth[useful]=shoe.depth[useful]
                    layers=[p for p in layers if p is not shoe]
                    info['cleanup'].append({'removed_duplicate':'footwear','integrated_into':'leg_subparts'})
            crossing=settings.get('crossing_legs') if kind=='leg' else None
            if crossing:
                split,events=split_reviewed_crossing_limbs(limb,chains,kind,
                    crossing['front_outline'],crossing['back_outline'])
            else:
                split,events=split_existing_limb(limb,chains,kind)
            if events:
                layers=[p for p in layers if p is not limb]+split
                info[key]=events
        integrity_hints,_=read_hints(item.name)
        layers,info['named_body_repairs']=restore_named_regions(src,originals,layers,integrity_hints,previous/item.name,settings)
        if settings.get('instances'):
            # Preserve the previously reviewed cow parts; no duplicate background patch.
            for inst in settings['instances']:
                scope=polygon(src.shape[:2],inst['polygon'])
                for p in layers:p.rgba[scope]=0
                for e in old['layers']:
                    if e['source']==inst['name'] and e['role'] in {'animal_body','animal_head','animal_tail'}:
                        rgba=np.array(Image.open(previous/item.name/e['file']).convert('RGBA'))
                        layers.append(Layer(e['name'],rgba,np.full(src.shape[:2],145,np.uint8),e['source'],
                                            role=e['role'],pivot=e.get('pivot_xy'),parent=e.get('parent')))
                for chain in inst['legs']:
                    chain=dict(chain)
                    if settings.get('subdivide_legs') is not None:
                        chain['subdivide']=settings['subdivide_legs']
                    mask=polygon(src.shape[:2],chain['polygon']) & scope & (src[...,3]>10)
                    rgba=src.copy();rgba[~mask]=0
                    split,events=split_existing_limb(Layer(chain['name'],rgba,np.full(mask.shape,145,np.uint8),inst['name']),[chain],'leg')
                    for p in split:
                        if p.role in {'leg_thigh','leg_whole'}:p.parent=f"{inst['name']}_body"
                    layers.extend(split);info['legs'].extend(events)
        layers=separate_selected_objects(item.name,layers)
        layers,changes=remove_reviewed_duplicates(item.name,layers)
        info['cleanup'].extend(changes)
        if item.name=='sc_gen_45206-meowa-watermark':
            layers=split_knight_skirts(src,layers)
            info['reviewed_parts']='front skirt armour, side skirt panels, visible lining and separate leg armour'
        elif item.name=='Keep-the-car-exactly-the-same-same-flat-straight-on-rear-meowa-watermark':
            layers=split_rear_car(src)
            info['reviewed_parts']='rear tail lamps, mirrors, visible tyres and completed body under lamps'
        layers=apply_reviewed_repairs(item.name,src,layers)
        layers=[p for p in layers if p.mask.any()]
        names={p.name for p in layers}
        for key in ('arms','legs'):
            info[key]=[r for r in info.get(key,[]) if all(p in names for p in r.get('parts',[]))]
        info['whole_limb_layers']=[p.name for p in layers if p.role in {'arm_whole','leg_whole'}]
        export_compact(layers,output/item.name,info)
        print(f'CLEAN_DONE {index}/{len(items)} layers={len(layers)} arms={len(info["arms"])} legs={len(info["legs"])}',flush=True)
    print(f'CLEAN_COMPLETE images={len(items)} output={output}',flush=True)


if __name__=='__main__':
    main()

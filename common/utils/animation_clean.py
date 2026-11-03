"""Compact animation delivery built from the original semantic layers.

Joint cuts retain the author's existing pixels. Overlap stays inside the limb
silhouette; no circular caps or full-source residual layer are introduced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from utils.animation_parts import Layer, _bbox, _cut, _complete_rgb, _segment_distance, _remove_duplicate_cloth, composite, save_previews
from utils.animation_integrity import foreground_mask, polygon, read_hints


def _keep_whole_reason(layer: Layer, mask: np.ndarray, chain: dict, joints: np.ndarray) -> str | None:
    if chain.get('keep_whole_reason'):
        return chain['keep_whole_reason']
    if chain.get('subdivide') is False:
        return 'joint_not_clearly_visible_in_reviewed_image'
    if chain.get('provenance') == 'completed_mask_skeleton_estimate' and not chain.get('joint_visible',False):
        return 'mask_skeleton_does_not_verify_a_visible_joint'
    xy = np.round(joints[1]).astype(int)
    h,w = mask.shape
    if not (0 <= xy[0] < w and 0 <= xy[1] < h):
        return 'joint_outside_existing_limb'
    distance = ndi.distance_transform_edt(~mask)
    if distance[xy[1],xy[0]] > 3:
        return 'joint_outside_existing_limb'
    lengths = np.linalg.norm(np.diff(joints[:3],axis=0),axis=1)
    width = ndi.distance_transform_edt(mask)
    radius = float(np.percentile(width[mask],80))
    if lengths.min() < max(6,radius*2):
        return 'short_round_part_without_two_clear_limb_segments'
    return None


def split_existing_limb(layer: Layer, chains: list[dict], kind: str) -> tuple[list[Layer], list[dict]]:
    """Partition into two bones, with narrow, source-coloured joint underlap."""
    if not chains or not layer.mask.any():
        return [layer], []
    if all(c.get('subdivide') is False for c in chains):
        layer.role = f'{kind}_whole'
        layer.meta.update({'label':'手臂整体' if kind=='arm' else '腿部整体',
                           'kept_whole_reason':'joint_not_clearly_visible_in_reviewed_image'})
        return [layer], []
    ys, xs = np.where(layer.mask)
    pixels = np.column_stack([xs, ys])
    joints = [np.asarray(c['points'], float) for c in chains]
    fields = [np.minimum.reduce([_segment_distance(pixels, a, b) for a, b in zip(j[:-1], j[1:])]) for j in joints]
    assignment = np.argmin(np.stack(fields), axis=0)
    # Preserve an already disconnected leg as one limb. A flared trouser edge
    # can be closer to the other skeleton and otherwise become a floating tile.
    if kind=='leg' and len(chains)>1:
        _,labels,stats,_=cv2.connectedComponentsWithStats(layer.mask.astype(np.uint8),8)
        main=[i for i in range(1,len(stats)) if stats[i,cv2.CC_STAT_AREA]>=max(24,layer.mask.sum()*.03)]
        if len(main)==len(chains):
            ids=labels[ys,xs]
            costs=np.array([[f[ids==i].mean() for f in fields] for i in main])
            owners=costs.argmin(axis=1)
            if len(set(owners.tolist()))==len(chains):
                for component,owner in zip(main,owners):
                    assignment[ids==component]=owner
    roles = ('upper', 'forearm') if kind == 'arm' else ('thigh', 'shin')
    pieces, events = [], []
    for ci, (chain, joint) in enumerate(zip(chains, joints)):
        chosen = assignment == ci
        if chosen.sum() < 24:
            continue
        mask = np.zeros(layer.mask.shape, bool)
        mask[ys[chosen], xs[chosen]] = True
        reason = _keep_whole_reason(layer,mask,chain,joint)
        if reason:
            piece = _cut(layer,mask,chain['name'],pivot=joint[0].tolist(),role=f'{kind}_whole')
            piece.meta.update({'label':'手臂整体' if kind=='arm' else '腿部整体','kept_whole_reason':reason,
                               'texture':'original_layer_pixels'})
            pieces.append(piece)
            events.append({'name':chain['name'],'points':joint.tolist(),'parts':[piece.name],
                           'mode':'whole','reason':reason})
            continue
        incoming = joint[1]-joint[0]
        outgoing = joint[2]-joint[1]
        normal = incoming/max(np.linalg.norm(incoming), 1)+outgoing/max(np.linalg.norm(outgoing), 1)
        if np.linalg.norm(normal) < 0.1:
            normal = outgoing
        normal /= max(np.linalg.norm(normal), 1)
        signed = (pixels-joint[1])@normal
        # 3-8 px of existing texture, never a newly painted disk outside a limb.
        width = ndi.distance_transform_edt(mask)
        overlap = float(np.clip(np.percentile(width[mask], 80)*0.18, 2, 8))
        made = []
        for bi, role in enumerate(roles):
            # A partly transparent boundary must have one owner, otherwise its
            # alpha increases when the two copies are composited together.
            boundary = overlap*(layer.rgba[ys,xs,3] == 255)
            select = chosen & ((signed <= boundary) if bi == 0 else (signed > -boundary))
            if select.sum() < 12:
                continue
            support = np.zeros_like(mask)
            support[ys[select], xs[select]] = True
            name = f"{chain['name']}_{role}"
            piece = _cut(layer, support, name, pivot=joint[bi].tolist(), role=f'{kind}_{role}',
                         parent=f"{chain['name']}_{roles[0]}" if bi else None)
            piece.meta.update({'joint_provenance':chain.get('provenance','previous_joint_coordinates'),
                               'joint_overlap_px':overlap, 'texture':'original_layer_pixels'})
            made.append(piece)
        if len(made) == 1:
            made[0].parent = None
        pieces.extend(made)
        events.append({'name':chain['name'], 'points':joint.tolist(), 'parts':[p.name for p in made],
                       'provenance':chain.get('provenance','previous_joint_coordinates')})
    if not pieces:
        return [layer], []
    union = np.logical_or.reduce([p.mask for p in pieces])
    if not np.array_equal(union, layer.mask):
        raise ValueError(f'Limb partition lost pixels: {layer.name}')
    # Underlap is a copy of the same source pixel; the rest pose is unchanged.
    before, after = composite([layer]), composite(pieces)
    if not np.array_equal(before, after):
        raise ValueError(f'Limb partition changed original colours: {layer.name}')
    return pieces, events


def split_reviewed_crossing_limbs(layer: Layer, chains: list[dict], kind: str,
                                  front_outline: list[list[int]], back_outline: list[list[int]],
                                  use_lama: bool = True) -> tuple[list[Layer], list[dict]]:
    """Separate reviewed crossing limbs before cutting either at its joint.

    A nearest-bone partition can swap ownership halfway down a crossing thigh.
    The first chain is the foreground limb; the second is completed only within
    a reviewed outline covered by fully opaque foreground pixels.
    """
    if len(chains)!=2:
        raise ValueError('A reviewed crossing requires exactly two limb chains')
    front_mask=layer.mask & polygon(layer.mask.shape,front_outline)
    back_known=layer.mask & ~front_mask
    if not front_mask.any() or not back_known.any():
        raise ValueError('Both reviewed limbs need visible source pixels')
    hidden=polygon(layer.mask.shape,back_outline) & front_mask & (layer.rgba[...,3]>=220)
    back_support=back_known|hidden
    back_rgba,completed=_complete_rgb(layer.rgba,back_known,back_support,use_lama)
    front=_cut(layer,front_mask,chains[0]['name'])
    back=Layer(chains[1]['name'],back_rgba,layer.depth.copy(),layer.source,completed=completed)
    back.depth[completed]=np.minimum(layer.depth[completed].astype(np.int16)+1,255).astype(np.uint8)
    pieces,events=[],[]
    for limb,chain in zip((front,back),chains):
        split,records=split_existing_limb(limb,[chain],kind)
        for p in split:
            p.meta['limb_assignment']='reviewed_crossing_boundary'
            if limb is back:
                p.meta.update({'texture':'original_visible_pixels_with_bounded_crossing_completion',
                               'inferred_hidden_pixels':int((completed & p.mask).sum())})
        pieces.extend(split);events.extend(records)
    rendered,reference=composite(pieces),composite([layer])
    unchanged=~hidden | (layer.rgba[...,3]==255)
    if not np.array_equal(rendered[unchanged],reference[unchanged]):
        raise ValueError('Reviewed crossing changed the observed limb appearance')
    return pieces,events


def previous_chains(info: dict, key: str) -> list[dict]:
    chains=[]
    for e in info.get(key,[]):
        name=e.get('limb') or e.get('arm') or e.get('name')
        points=e.get('joints') or e.get('points')
        if not name or not points:
            continue
        chain={'name':name,'points':points,
               'provenance':e.get('provenance','human_pose_with_mask_alignment')}
        if e.get('mode')=='whole':
            chain['keep_whole_reason']=e.get('reason','joint_not_clearly_visible_in_reviewed_image')
        chains.append(chain)
    return chains


def restore_named_regions(src: np.ndarray, originals: list[Layer], layers: list[Layer], hints: dict,
                          previous: Path, limb_policy: dict | None = None) -> tuple[list[Layer], list[dict]]:
    """Reuse named, reviewed missing-body repairs, without exporting residuals."""
    target, _ = foreground_mask(src, originals, hints)
    records = []
    for instance in hints.get('source_instances', []):
        scope = target.copy()
        if instance.get('box'):
            scope[:] = False
            x1,y1,x2,y2 = instance['box']
            scope[y1:y2,x1:x2] = target[y1:y2,x1:x2]
        retained = [_cut(p, p.mask & scope, p.name, role=p.role) for p in originals if p.name in instance.get('retain_tags', [])]
        if instance.get('replace_all'):
            layers = []
        else:
            for p in layers:
                p.rgba[scope] = 0
        definitions = instance.get('regions', [])
        slots = [polygon(target.shape, d['polygon']) & scope if d.get('polygon') else scope.copy() for d in definitions]
        chains = [dict(c) for c in instance.get('chains', [])]
        for c in chains:
            allowed = (limb_policy or {}).get('subdivide_arms' if c['kind']=='arm' else 'subdivide_legs')
            if allowed is not None:
                c['subdivide'] = allowed
        slots += [polygon(target.shape, d['polygon']) & scope for d in chains]
        slots += [p.mask for p in retained]
        owner = np.full(target.shape, -1, np.int16)
        for i,m in enumerate(slots):
            owner[m] = i
        pieces = []
        for i,d in enumerate(definitions):
            path = previous / f"{d['name']}.png"
            if path.is_file():
                rgba = np.array(Image.open(path).convert('RGBA'))
            else:
                rgba = src.copy(); rgba[owner != i] = 0
            dp = previous / f"{d['name']}_depth.png"
            depth = np.array(Image.open(dp).convert('L')) if dp.is_file() else np.full(target.shape,230-i*10,np.uint8)
            p = Layer(d['name'], rgba, depth, f"source:{instance['name']}",
                      role=d.get('role','source_region'), pivot=d.get('pivot'), parent=d.get('parent'),
                      meta={'label':d.get('label',d['name']), 'provenance':'reviewed_source_regions'})
            pieces.append(p)
        for ci,c in enumerate(chains):
            known = owner == len(definitions)+ci
            rgba = src.copy(); rgba[~known] = 0
            limb = Layer(c['name'], rgba, np.full(target.shape,100,np.uint8), f"source:{instance['name']}")
            split,_ = split_existing_limb(limb,[c],c['kind'])
            for p in split:
                if p.role in {'arm_upper','leg_thigh','arm_whole','leg_whole'}:
                    p.parent = instance.get('body_name')
                if instance['name'] in {'pikachu','cat'} and p.role in {'arm_whole','leg_whole'}:
                    p.meta['label'] = '前肢整体' if p.role=='arm_whole' else '后肢整体'
            pieces.extend(split)
        pieces.extend(p for p in retained if p.mask.any())
        pieces = [p for p in pieces if p.mask.any()]
        labels = np.zeros(target.shape,np.int16)
        for i,p in enumerate(pieces,1):
            labels[p.mask & scope] = i
        missing = scope & (labels == 0)
        if missing.any():
            nearest = ndi.distance_transform_edt(labels == 0,return_distances=False,return_indices=True)
            assigned = labels[tuple(nearest)]
            for i,p in enumerate(pieces,1):
                selected = missing & (assigned == i)
                p.rgba[selected] = src[selected]
        # These are the explicitly reviewed missing-body cases. Keep their
        # observed appearance, including eye contours, without an extra overlay.
        from utils.animation_integrity import _owner_map
        visible = _owner_map(pieces,scope)
        if (visible[scope] < 0).any():
            raise ValueError(f"Named source repair is incomplete: {instance['name']}")
        edge = scope & (src[...,3] < 255)
        for i,p in enumerate(pieces):
            selected = visible == i
            p.rgba[selected] = src[selected]
            p.rgba[edge & ~selected,3] = 0
        rendered = composite(pieces)
        if not np.array_equal(rendered[scope,3],src[scope,3]):
            raise ValueError(f"Named body repair changed source alpha: {instance['name']}")
        if scope.any() and np.abs(rendered[scope,:3].astype(int)-src[scope,:3].astype(int)).max()>1:
            raise ValueError(f"Named body repair changed source colours: {instance['name']}")
        layers.extend(pieces)
        records.append({'name':instance['name'], 'method':'reuse_named_reviewed_body_parts_and_recropped_limbs',
                        'edge_pixels_assigned_to_existing_parts':int(missing.sum())})
    return [p for p in layers if p.mask.any()],records


def split_umbrella_from_clothing(layers: list[Layer], coat_outline: list[list[int]],
                                 canopy_outline: list[list[int]], shaft_outline: list[list[int]],
                                 use_lama: bool = True, shaft_path: list[list[int]] | None = None) -> list[Layer]:
    """Separate a reviewed umbrella and complete only its bounded occlusions."""
    top = next(p for p in layers if p.name == 'topwear')
    shape = top.mask.shape
    garment = top.mask & polygon(shape,coat_outline)
    # A few coat-coloured pixels along an uncertain edge can otherwise seed a
    # second coat in the inpainted canopy. Exclude a narrow context band too.
    canopy_known = top.mask & ~cv2.dilate(garment.astype(np.uint8),np.ones((17,17),np.uint8)).astype(bool)
    canopy_support = (polygon(shape,canopy_outline) & ndi.binary_fill_holes(top.mask)) | (top.mask & ~garment)
    canopy_rgba,completed = _complete_rgb(top.rgba,canopy_known,canopy_support,use_lama)
    canopy_depth = np.full(shape,230,np.uint8)
    shaft = polygon(shape,shaft_outline)
    shaft_cleanup = cv2.dilate(shaft.astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool)
    rod_rgba = np.zeros_like(top.rgba)
    rod_depth = top.depth.copy()
    for p in layers:
        if p.name not in {'topwear','bottomwear'}:
            continue
        if p is top:
            p.rgba[~garment] = 0
        visible_rod = p.mask & shaft
        cut = p.mask & shaft_cleanup
        rod_rgba[visible_rod] = p.rgba[visible_rod]
        rod_depth[visible_rod] = p.depth[visible_rod]
        if cut.any():
            known = p.mask & ~cut
            if use_lama:
                # A thin rod needs local colour continuation, rather than a
                # newly invented lapel/pocket from generative completion.
                nearest = ndi.distance_transform_edt(~known,return_distances=False,return_indices=True)
                context = p.rgba[...,:3][tuple(nearest)].copy()
                repaired = cv2.inpaint(context,cut.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
                p.rgba[cut,:3] = repaired[cut]
            else:
                p.rgba,_ = _complete_rgb(p.rgba,known,p.mask,False)
            p.meta.update({'umbrella_shaft_removed_pixels':int(cut.sum()),
                           'occlusion_completion':'local_edge_inpaint' if use_lama else 'nearest_material_test'})
    if shaft_path:
        # The model's tube includes bits of the lapel texture. Trace its observed
        # straight centre line and retain the original lower handle below.
        alpha = np.zeros(shape,np.uint8)
        cv2.polylines(alpha,[np.asarray(shaft_path,np.int32)],False,255,6,cv2.LINE_AA)
        rod_rgba[:] = 0
        rod_rgba[alpha>0,:3] = [28,27,30]
        rod_rgba[...,3] = alpha
        rod_depth[alpha>0] = 100
    # This image's original objects layer is the lower umbrella handle only.
    old_rod = next((p for p in layers if p.name == 'objects'),None)
    if old_rod is not None:
        rod_rgba[old_rod.mask] = old_rod.rgba[old_rod.mask]
        rod_depth[old_rod.mask] = old_rod.depth[old_rod.mask]
    if shaft_path:
        rod_depth[rod_rgba[...,3]>0] = 100
    canopy = Layer('umbrella',canopy_rgba,canopy_depth,'objects',role='prop',pivot=[609,405],
                   meta={'label':'伞面','provenance':'reviewed_umbrella_clothing_boundary',
                         'inferred_hidden_pixels':int(completed.sum())})
    handle = Layer('umbrella_handle',rod_rgba,rod_depth,'objects',role='prop',parent='umbrella',pivot=[609,405],
                   meta={'label':'伞杆与伞柄','provenance':'observed_shaft_path_and_original_lower_handle' if shaft_path else 'original_umbrella_shaft_pixels'})
    return [p for p in layers if p is not old_rod]+[canopy]+([handle] if handle.mask.any() else [])


def split_handheld_sword(layers: list[Layer], weapon_outline: list[list[int]],
                         grip_outline: list[list[int]], pivot: list[float]) -> list[Layer]:
    """Separate the reviewed pixel-art sword without cutting off its fingers."""
    hand=next(p for p in layers if p.name=='handwear')
    before=composite([hand])
    rgb=hand.rgba[...,:3].astype(np.int16)
    fingers=(rgb[...,0]>20)&(rgb[...,0]>rgb[...,1]+10)&(rgb[...,0]>rgb[...,2]+10)
    visible=hand.mask & polygon(hand.mask.shape,weapon_outline) & ~fingers
    if not visible.any():
        raise ValueError('No visible sword inside the reviewed outline')
    hidden=polygon(hand.mask.shape,grip_outline) & fingers & (hand.rgba[...,3]==255)
    rgba,completed=_complete_rgb(hand.rgba,visible,visible|hidden,False)
    sword=Layer('sword',rgba,hand.depth.copy(),'handwear',role='prop',parent='handwear',pivot=pivot,
                completed=completed,meta={'label':'剑','provenance':'reviewed_pixel_art_weapon_boundary',
                                         'inferred_hidden_grip_pixels':int(completed.sum())})
    sword.depth[:]=np.minimum(hand.depth.astype(np.int16)+1,255).astype(np.uint8)
    hand.rgba[visible]=0
    if not np.array_equal(composite([hand,sword]),before):
        raise ValueError('Sword separation changed the observed hand or weapon')
    return [*layers,sword]


def separate_selected_objects(name: str, layers: list[Layer]) -> list[Layer]:
    if name=='16-1-meowa-watermark':
        hand=next(p for p in layers if p.name=='handwear');hand.role='arm_whole'
        hand.meta['label']='手臂整体'
        leg=next(p for p in layers if p.name=='legwear');leg.role='leg_whole';leg.meta['label']='腿部整体'
        shoe=next((p for p in layers if p.name=='footwear'),None)
        if shoe is not None:
            extra=shoe.mask & ~leg.mask
            leg.rgba[extra]=shoe.rgba[extra];leg.depth[extra]=shoe.depth[extra]
            layers=[p for p in layers if p is not shoe]
        return split_handheld_sword(layers,
            [[425,605],[495,605],[510,650],[495,690],[320,849],[248,849],[260,813],[420,651]],
            [[451,628],[463,639],[500,601],[489,590]],[480,629])
    if name == '301802_wuxingzhe_p':
        return split_umbrella_from_clothing(layers,
            [[548,283],[570,281],[579,275],[606,278],[624,296],[644,319],[670,340],
             [692,357],[686,366],[673,375],[670,409],[674,450],[690,500],[720,900],[440,925],[440,775],
             [540,537],[568,483],[573,464],[563,430],[552,399],[542,363],[535,339],
             [535,319],[544,316],[552,315],[548,302],[542,291]],
            [[436,10],[278,129],[245,323],[360,447],[519,462],[692,365],[731,212],[663,60]],
            [[496,233],[504,227],[671,478],[677,480],[680,489],[677,495],[671,498],
             [664,496],[660,491],[662,487]],shaft_path=[[500,230],[619,418]])
    if name == '006喷火龙Ymega':
        layer = next(p for p in layers if p.name == 'hair')
        head = polygon(layer.mask.shape, [[597,532],[659,536],[702,490],[714,463],[765,449],[814,455],[851,424],
                                         [818,363],[786,331],[756,285],[719,173],[704,174],[733,284],[692,264],
                                         [688,279],[728,329],[671,371],[624,424]]) & layer.mask
        xx = np.indices(layer.mask.shape)[1]
        masks = [head,layer.mask & ~head & (xx<660),layer.mask & ~head & (xx>=660)]
        names = ['head_neck','wing_image_left','wing_image_right']
        labels = ['头与颈部','画面左翼','画面右翼']
        parts = [_cut(layer,m,n,role='source_region',meta={'label':label}) for m,n,label in zip(masks,names,labels)]
        return [p for p in layers if p is not layer]+parts
    if name == 'f8bdb4d2d17cc6a56bbeb911b2864245':
        layer = next(p for p in layers if p.name == 'objects')
        yy,xx = np.indices(layer.mask.shape)
        text = yy>810
        flag = (xx<490)&~text
        medals = (xx>=480)&(xx<680)&(yy>600)&(yy<750)
        masks = [flag,medals,~text & ~flag & ~medals,text]
        names = ['flag_and_horn','badge_accessories','fishing_rod','poster_text']
        labels = ['旗帜与号角配件','徽章配件','钓竿','原图文字']
        parts = [_cut(layer,layer.mask&m,n,role='source_region',meta={'label':label}) for m,n,label in zip(masks,names,labels)]
        # The poster lettering is also repeated in the original bottomwear/shoes.
        for p in layers:
            if p.source in {'bottomwear','footwear'}:
                p.rgba[text] = 0
        return [p for p in layers if p is not layer]+[p for p in parts if p.mask.any()]
    return layers


def remove_reviewed_duplicates(name: str, layers: list[Layer]) -> tuple[list[Layer], list[dict]]:
    records = []
    # These original predictions visibly repeat trouser legs / fur sleeves in
    # the garment layer. Long knight coats remain separate occluding garments.
    rules = {'05':{'bottomwear':{'legwear'}},
             '301102_malilian_p':{'bottomwear':{'handwear','legwear'}}}
    for garment,sources in rules.get(name,{}).items():
        p = next((p for p in layers if p.name == garment),None)
        masks = [q.mask for q in layers if q.source in sources and q.role.startswith(('arm_','leg_'))]
        if p is not None and masks:
            support = np.logical_or.reduce(masks)
            overlap = p.mask & support
            p.rgba[overlap] = 0
            records.append({'trimmed':garment,'duplicate_pixels':int(overlap.sum()),'owners':sorted(sources)})
    eligible = {'topwear','bottomwear','handwear','legwear','footwear','objects'}
    rank = {'topwear':0,'bottomwear':1,'objects':2,'legwear':3,'handwear':4,'footwear':5}
    candidates = sorted([p for p in layers if p.name in eligible and p.role == 'original'],key=lambda p:rank[p.name])
    removed = set()
    for i,keep in enumerate(candidates):
        if keep in removed:
            continue
        for other in candidates[i+1:]:
            if other in removed:
                continue
            a,b = keep.mask,other.mask
            both,union = a&b,a|b
            if both.sum()/max(1,union.sum()) < 0.90:
                continue
            colour = np.abs(keep.rgba[both,:3].astype(float)-other.rgba[both,:3].astype(float)).mean(axis=-1)
            if np.median(colour) > 25:
                continue
            extra = b&~a
            keep.rgba[extra] = other.rgba[extra]
            keep.depth[extra] = other.depth[extra]
            removed.add(other)
            records.append({'merged_near_duplicate':other.name,'kept':keep.name})
    if name == '006喷火龙Ymega':
        labels = {'earwear':'尾部火焰','topwear':'躯干与尾巴','objects':'原图阴影','face':'脸部底层'}
        for p in layers:
            if p.name in labels:p.meta['label']=labels[p.name]
    return [p for p in layers if p not in removed and p.mask.any()],records


def export_compact(layers: list[Layer], output: Path, info: dict) -> None:
    """Export real RGBA parts once; depth and completion masks stay internal."""
    from psd_tools import PSDImage
    output.mkdir(parents=True,exist_ok=True)
    layers = [p for p in layers if p.mask.any()]
    names = {p.name for p in layers}
    if len(names) != len(layers):
        raise ValueError('Duplicate layer names')
    for p in layers:
        if p.parent and p.parent not in names:
            raise ValueError(f'Missing parent {p.parent} for {p.name}')
    by_name = {p.name: p for p in layers}
    for key in ('arms', 'legs'):
        for record in info.get(key, []):
            parts = [by_name[n] for n in record.get('parts', []) if n in by_name]
            if len(parts) == 2 and record.get('points') and all(p.meta.get('joint_finish') for p in parts):
                record['points'][0] = parts[0].pivot
                record['points'][1] = parts[1].pivot
                record['joint_finish'] = parts[1].meta['joint_finish']
    # Repeated exports of this new delivery remove only files in its old manifest.
    manifest = output/'关节与图层.json'
    if manifest.is_file():
        old = json.loads(manifest.read_text(encoding='utf-8'))
        for item in old.get('layers',[]):
            p = output/item['file']
            if item['name'] not in names and p.parent.resolve() == output.resolve() and p.is_file():
                p.unlink()
    h,w = layers[0].mask.shape
    psd = PSDImage.new(mode='RGBA',size=(w,h))
    entries = []
    for p in layers:
        Image.fromarray(p.rgba).save(output/f'{p.name}.png')
        entries.append({'name':p.name,'file':f'{p.name}.png','source':p.source,'role':p.role,
                        'pivot_xy':p.pivot,'parent':p.parent,'bbox_xyxy':list(_bbox(p.rgba[...,3]>0)),**p.meta})
    for p in sorted(layers,key=lambda p:p.z,reverse=True):
        x1,y1,x2,y2 = _bbox(p.rgba[...,3]>0)
        pixel=psd.create_pixel_layer(Image.fromarray(p.rgba[y1:y2,x1:x2]),name=p.name,top=y1,left=x1)
        # This psd-tools version imports RGBA transparency both as a channel
        # and an automatic mask. Retain the channel once, avoiding alpha^2.
        if pixel.has_mask():
            pixel.remove_mask()
        if p.visibility_mask is not None:
            pixel.create_mask(Image.fromarray(p.visibility_mask[y1:y2, x1:x2]), top=y1, left=x1)
        pixel.visible = p.meta.get('default_visible', True)
    psd.save(output/'动画拆层.psd')
    info.update({'canvas_wh':[w,h],'layers':entries,'coordinates':'top_left_xy_full_canvas_pixels',
                 'format':'compact_original_layers_plus_two_bone_limb_cuts',
                 'limitations':['Original hidden-part predictions are preserved; no arbitrary-motion guarantee.',
                                'Joint cuts do not establish correct anatomy for every estimated pose.',
                                'PSD uses median depth ordering; static preview uses per-pixel depth.']})
    manifest.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    save_previews(layers,output)
    # Read every exported file back, including every PSD pixel layer.
    stored = PSDImage.open(output/'动画拆层.psd')
    by_name = {p.name:p for p in stored}
    if len(stored) != len(layers):
        raise ValueError('PSD layer count mismatch')
    hashes = set()
    for p in layers:
        png = np.array(Image.open(output/f'{p.name}.png').convert('RGBA'))
        if not np.array_equal(png,p.rgba):
            raise ValueError(f'PNG round-trip mismatch: {p.name}')
        item = by_name[p.name]
        if not np.array_equal(np.array(item.topil().convert('RGBA')),np.array(Image.fromarray(png).crop(item.bbox))):
            raise ValueError(f'PSD layer mismatch: {p.name}')
        digest = hashlib.sha256(png.tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f'Identical duplicate PNG: {p.name}')
        hashes.add(digest)
    unwanted = [p.name for p in output.iterdir() if p.suffix=='.gif' or p.stem.endswith(('_depth','_补全区域')) or 'remainder' in p.name]
    if unwanted:
        raise ValueError(f'Unexpected delivery files: {unwanted}')

"""Reviewed character layouts and bounded attachment completion for delivery."""

from __future__ import annotations

import copy
from pathlib import Path
import json

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import splprep, splev

from utils.animation_parts import Layer, _complete_rgb, composite
from utils.animation_integrity import polygon


LAYOUTS = Path(__file__).resolve().parents[1]/'assets/animation_layouts.json'
REVISION = 'character-anatomy-2026-11-02-v1'


def ellipse(shape: tuple[int, int], values: list[float]) -> np.ndarray:
    x, y, rx, ry = values
    yy, xx = np.indices(shape)
    return ((xx-x)/max(rx, 1))**2+((yy-y)/max(ry, 1))**2 <= 1


def curved_region(shape: tuple[int, int], points: list[list[float]]) -> np.ndarray:
    """A closed, reviewed silhouette; not an ellipse pasted on the artwork."""
    xy = np.asarray(points, float)
    curve, _ = splprep(xy.T, s=0, per=True, k=min(3, len(xy)-1))
    vertices = np.round(np.array(splev(np.linspace(0, 1, 600), curve)).T).astype(np.int32)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [vertices], 1)
    return mask > 0


def _keep_connected(part: Layer) -> None:
    if part.role == 'animal_body':
        opened = cv2.morphologyEx(part.mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        envelope = cv2.dilate(opened, np.ones((3, 3), np.uint8)) > 0
        part.rgba[~envelope] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(part.mask.astype(np.uint8), 8)
    if count <= 2:
        return
    main = 1+int(stats[1:, cv2.CC_STAT_AREA].argmax())
    keep = labels == main
    # Retain adjacent antialias pixels but remove detached cut-boundary debris.
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    part.rgba[~keep] = 0


def _torso_texture(src: np.ndarray, known: np.ndarray, support: np.ndarray,
                   light_material: bool = False) -> np.ndarray:
    """Inpaint from real torso context without radial nearest-pixel streaks."""
    from annotators.lama_inpainter import apply_inpaint
    trusted = cv2.erode(known.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if light_material:
        trusted &= src[..., :3].min(axis=-1) > 105
    if trusted.sum() < 25:
        return src[..., :3].copy()
    yy, xx = np.where(support)
    x1, x2 = max(0, int(xx.min())-20), min(src.shape[1], int(xx.max())+21)
    y1, y2 = max(0, int(yy.min())-20), min(src.shape[0], int(yy.max())+21)
    sl = np.s_[y1:y2, x1:x2]
    rgb = np.zeros((y2-y1, x2-x1, 3), np.uint8)
    rgb[:] = np.median(src[trusted, :3], axis=0).astype(np.uint8)
    rgb[trusted[sl]] = src[sl][trusted[sl], :3]
    missing = support[sl] & ~trusted[sl]
    filled = apply_inpaint(rgb, missing.astype(np.uint8)*255, precision=__import__('torch').float32)
    result = src[..., :3].copy()
    result[sl] = filled
    return result


def _make_part(src: np.ndarray, known: np.ndarray, support: np.ndarray,
               name: str, role: str, depth: int, parent: str | None,
               pivot: list[float], label: str, use_lama: bool) -> Layer:
    if known.sum() < 4:
        raise ValueError(f'No observed pixels for {name}')
    rgba, added = _complete_rgb(src, known, support | known, use_lama)
    # Source pixels keep their original antialiasing and pixel-art outlines.
    # Only a newly inferred attachment receives a soft subpixel edge.
    if added.any():
        alpha = cv2.GaussianBlur(support.astype(np.float32)*255, (0, 0), .45)
        rgba[added, 3] = np.maximum(1, np.round(alpha[added])).astype(np.uint8)
    rgba[rgba[..., 3] == 0] = 0
    return Layer(name, rgba, np.full(known.shape, depth, np.uint8), 'reviewed:source_character',
                 parent=parent, pivot=pivot, role=role, completed=added,
                 meta={'label': label, 'standard_revision': REVISION,
                       'provenance': 'reviewed anatomical regions with bounded hidden-root completion',
                       'completion': 'author_LaMa' if use_lama else 'source_material_continuation',
                       'inferred_pixels': int(added.sum())})


def split_character(src: np.ndarray, layout: dict, use_lama: bool = True) -> tuple[list[Layer], dict]:
    """Keep every source pixel; complete roots only inside the character."""
    shape = src.shape[:2]
    full = src[..., 3] > 10
    x1, y1, x2, y2 = layout['box']
    scope = np.zeros(shape, bool); scope[y1:y2, x1:x2] = True
    full &= scope
    count, labels, stats, _ = cv2.connectedComponentsWithStats(full.astype(np.uint8), 8)
    keep = np.zeros(count, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 12
    full = keep[labels]
    if not full.any():
        raise ValueError(f"Empty character {layout['name']}")
    prefix = layout['name']
    definitions = layout['parts']
    owner = np.zeros(shape, np.int16)
    for i, definition in enumerate(definitions, 1):
        region = polygon(shape, definition['polygon'])
        if definition.get('expand'):
            k = int(definition['expand'])*2+1
            region = cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        owner[region & full] = i
    # Thin cut-boundary fragments belong to the adjacent head/limb, rather
    # than floating around the body after those parts are moved away.
    initial_body = full & (owner == 0)
    n, connected, sizes, _ = cv2.connectedComponentsWithStats(initial_body.astype(np.uint8), 8)
    if n > 2:
        main = 1+int(sizes[1:, cv2.CC_STAT_AREA].argmax())
        fields = [cv2.distanceTransform((~(full & (owner == i))).astype(np.uint8), cv2.DIST_L2, 5)
                  for i in range(1, len(definitions)+1)]
        for component in range(1, n):
            if component == main:
                continue
            fragment = connected == component
            distances = [float(f[fragment].mean()) for f in fields]
            owner[fragment] = 1+int(np.argmin(distances))
    for region in layout.get('body_regions', []):
        owner[polygon(shape, region) & full] = 0
    protected = np.zeros(shape, bool)
    for region in layout.get('body_regions', []):
        protected |= polygon(shape, region)
    fields = np.stack([cv2.distanceTransform((~(full & (owner == i))).astype(np.uint8), cv2.DIST_L2, 5)
                       for i in range(1, len(definitions)+1)])
    fringe = full & (owner == 0) & ~ellipse(shape, layout['body']) & ~protected
    fringe &= fields.min(axis=0) < 8
    nearest_part = fields.argmin(axis=0)+1
    owner[fringe] = nearest_part[fringe]
    body_known = full & (owner == 0)
    body_support = body_known | curved_region(shape, layout['body_outline'])
    overlying = np.zeros(shape, bool)
    for i, definition in enumerate(definitions, 1):
        if definition['role'] in {'animal_head', 'animal_wing'}:
            overlying |= owner == i
    body_known &= ~(cv2.dilate(overlying.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    for region in layout.get('body_regions', []):
        body_support |= polygon(shape, region)
    body_name = prefix+'_body'
    body = _make_part(src, body_known, body_support, body_name, 'animal_body', 120,
                      None, layout['body'][:2], layout.get('label', prefix)+'身体',
                      use_lama and layout['kind'] != 'bird')
    if layout['kind'] != 'bird' and body.completed.any() and use_lama:
        texture = _torso_texture(src, body_known, body_support, prefix.startswith('sheep'))
        body.rgba[body.completed, :3] = texture[body.completed]
    if layout['kind'] == 'bird' and body.completed.any():
        # Do not continue a removed wing's feather outline across the torso.
        sample = cv2.erode(body_known.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        sample &= (src[..., 3] >= 220) & (src[..., :3].max(axis=-1) > 130)
        rows = np.flatnonzero(sample.any(axis=1))
        if len(rows) >= 2:
            colours = np.array([np.median(src[y, sample[y], :3], axis=0) for y in rows])
            trend = np.stack([np.interp(np.arange(shape[0]), rows, colours[:, k]) for k in range(3)], axis=-1)
            trend = cv2.GaussianBlur(trend.astype(np.float32), (1, 0), sigmaX=0, sigmaY=5)
            yy = np.indices(shape)[0]
            # A wing edge is an internal line, not torso texture. Reconstruct
            # the torso interior, retaining the exposed source outline.
            interior = cv2.erode(body_support.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            body.rgba[interior, :3] = np.round(trend[yy[interior]]).astype(np.uint8)
    # Match the small source sprite's outline on the newly uncovered torso.
    # Original exposed pixels and their RGB are left intact.
    edge = body_support & ~(cv2.erode(body_support.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    new_edge = edge & body.completed
    body.rgba[new_edge, :3] = layout['outline_rgb']
    # The full PNG contains hidden torso material; only the source silhouette
    # is shown in the default PSD pose. The mask remains reversible.
    body.visibility_mask = full.astype(np.uint8)*255
    body.meta['psd_visibility'] = 'source silhouette; completed full PNG retained'
    _keep_connected(body)
    result = [body]
    record = {'name': prefix, 'kind': layout['kind'], 'body': body_name, 'parts': [body_name],
              'limbs': {}, 'inferred_hidden_parts': []}
    for i, definition in enumerate(definitions, 1):
        known = full & (owner == i)
        support = known.copy()
        if definition.get('cap'):
            support |= ellipse(shape, definition['cap']) & full
        role = definition['role']
        name = prefix+'_'+definition['name']
        p = _make_part(src, known, support, name, role, definition.get('depth', 80), body_name,
                       definition['pivot'], definition['label'], use_lama)
        _keep_connected(p)
        result.append(p); record['parts'].append(name)
        if role in {'leg_whole', 'animal_wing', 'arm_whole'}:
            record['limbs'][definition['name']] = name
        if role == 'animal_head':
            record['head'] = name
        if role == 'animal_tail':
            record['tail'] = name
    return result, record


def split_farm_sheet(src: np.ndarray, info: dict, use_lama: bool = True) -> tuple[list[Layer], dict]:
    layouts = json.loads(LAYOUTS.read_text(encoding='utf-8'))['45-meowa-watermark']
    parts, instances = [], []
    for layout in layouts:
        current, record = split_character(src, layout, use_lama)
        parts.extend(current); instances.append(record)
        print('ANATOMY_DONE', layout['name'], len(current), flush=True)
    info = copy.deepcopy(info)
    info.update({'arms': [], 'legs': [], 'rounded_limb_pairs': [], 'whole_segment_alternatives': [],
                 'whole_limb_layers': [], 'anatomy_instances': instances, 'standard_revision': REVISION,
                 'method': 'Ours Full source reused; reviewed per-character limbs and author LaMa hidden-root completion'})
    return parts, info


def reset_records(info: dict, instances: list[dict]) -> dict:
    info = copy.deepcopy(info)
    for key in ('arms', 'legs', 'rounded_limb_pairs', 'whole_segment_alternatives', 'whole_limb_layers'):
        info[key] = []
    info.update({'anatomy_instances': instances, 'standard_revision': REVISION,
                 'visibility_policy': 'omit completely unseen parts; complete observed attachments only',
                 'method': 'Ours Full source/predictions reused with reviewed visible anatomy and bounded root completion'})
    return info


def finish_tangka_halo(parts: list[Layer]) -> None:
    """Continue the observed orange border and blue radial backing, not a figure."""
    halo=next(p for p in parts if p.name=='halo')
    if halo.completed is None or not halo.completed.any():return
    yy,xx=np.indices(halo.mask.shape)
    nx,ny=(xx-518)/190,(yy-477)/199
    radius=np.hypot(nx,ny);angle=np.arctan2(ny,nx)
    rgb=np.zeros_like(halo.rgba[...,:3]);rgb[:]=[37,44,75]
    rays=(np.cos(angle*212)>0.91)&(radius>.12)&(radius<.80)
    rgb[rays]=[137,126,75]
    outer=radius>=.80
    t=np.clip((radius-.80)/.20,0,1)
    glow=np.sin(t*np.pi)
    rgb[outer,0]=255
    rgb[outer,1]=(125+70*glow[outer]).astype(np.uint8)
    rgb[outer,2]=(35+65*glow[outer]).astype(np.uint8)
    trim=(abs(radius-.80)<.007)|(abs(radius-.988)<.007)
    rgb[trim]=[254,204,109]
    red_radius=np.sqrt(((xx-516)/98)**2+((yy-323)/96)**2)
    rgb[red_radius<=1]=[195,70,43]
    rgb[(red_radius>.974)&(red_radius<=1)]=[242,174,77]
    halo.rgba[halo.completed,:3]=rgb[halo.completed]
    halo.meta['completion']='visible orange border, red disc and blue/gold ray material continuation'


def partition_regions(src: np.ndarray, target: np.ndarray, definitions: list[dict],
                      use_lama: bool = True) -> list[Layer]:
    """Partition real observed pixels; an empty definition never creates a limb."""
    shape = target.shape
    owner = np.full(shape, -1, np.int16)
    for i, definition in enumerate(definitions):
        region = polygon(shape, definition['polygon']) if definition.get('polygon') else target.copy()
        if definition.get('mask') is not None:
            region &= definition['mask']
        if definition.get('material'):
            rgb=src[...,:3].astype(int)
            material=definition['material']
            if material=='green_skin':
                selected=(rgb[...,1]>rgb[...,0]+12)&(rgb[...,1]>rgb[...,2]+5)&(rgb[...,1]>65)
                selected=cv2.dilate((selected&region).astype(np.uint8),np.ones((3,3),np.uint8))>0
            elif material=='orange_ribbon':
                selected=(rgb[...,0]>170)&(rgb[...,1]>65)&(rgb[...,1]<200)&(rgb[...,2]<135)
                selected=cv2.dilate((selected&region).astype(np.uint8),np.ones((3,3),np.uint8))>0
            elif material=='head_without_red_halo':
                selected=~((rgb[...,0]>140)&(rgb[...,1]<112)&(rgb[...,2]<106))
                selected=ndi.binary_fill_holes(cv2.morphologyEx((selected&region).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0)
            elif material=='dragon_body':
                pale=(rgb[...,0]>65)&(rgb[...,1]>160)&(rgb[...,2]>170)
                lightning=(rgb[...,0]>170)&(rgb[...,1]>150)&(rgb[...,2]<130)
                selected=cv2.morphologyEx((region&~pale&~lightning&target).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
                selected=cv2.dilate(selected.astype(np.uint8),np.ones((3,3),np.uint8))>0
            else:
                raise ValueError(f'Unknown reviewed material {material}')
            region &= selected
        owner[target & region] = i
    if (target & (owner < 0)).any():
        raise ValueError('A reviewed named body/effect must own remaining source pixels')
    parts = []
    for i, definition in enumerate(definitions):
        known = target & (owner == i)
        if known.sum() < 12:
            continue
        support = known.copy()
        if definition.get('cap'):
            support |= ellipse(shape, definition['cap'])
        if definition.get('support_path'):
            path=np.asarray(definition['support_path'],np.int32)
            radius=int(definition.get('path_radius',8))
            tube=np.zeros(shape,np.uint8)
            cv2.polylines(tube,[path],False,1,radius*2,cv2.LINE_AA)
            for x,y in path:cv2.circle(tube,(int(x),int(y)),radius,1,-1)
            support |= tube>0
        for item in definition.get('support_ellipses',[]):
            support |= ellipse(shape,item)
        if definition.get('support_polygon'):
            support |= curved_region(shape, definition['support_polygon'])
        p = _make_part(src, known, support, definition['name'], definition['role'],
                       definition.get('depth', 100), definition.get('parent'),
                       definition.get('pivot'), definition.get('label', definition['name']), use_lama)
        if p.completed.any():
            # Full PNGs retain the smooth completed interface. Rest-pose masks
            # expose only observed pixels, avoiding duplicate seams in the PSD.
            p.visibility_mask = known.astype(np.uint8)*255
            p.meta['psd_visibility'] = 'observed rest pose; bounded completed PNG retained'
        parts.append(p)
    names = {p.name for p in parts}
    for p in parts:
        if p.parent and p.parent not in names:
            p.parent = None
    return parts


def split_configured(src: np.ndarray, info: dict, name: str,
                     target: np.ndarray | None = None, use_lama: bool = True) -> tuple[list[Layer], dict]:
    config = json.loads(LAYOUTS.read_text(encoding='utf-8'))[name]
    target = src[..., 3] > 10 if target is None else target
    if config.get('max_y'):
        target=target.copy();target[int(config['max_y']):]=False
    parts = partition_regions(src, target, config['parts'], use_lama)
    if name=='tangka':finish_tangka_halo(parts)
    if name=='JRPG-meowa-watermark':
        effects=next(p for p in parts if p.name=='dragon_effects')
        for p in parts:
            if p.role=='animal_body':
                before=p.mask.copy();_keep_connected(p)
                debris=before&~p.mask&target
                effects.rgba[debris]=src[debris]
    instances = []
    for entry in config.get('instances', []):
        prefix = entry['name']+'_'
        names = [p.name for p in parts if p.name.startswith(prefix)]
        instances.append({**entry, 'parts': names, 'inferred_hidden_parts': []})
    info=reset_records(info,instances)
    if config.get('arm_pairs'):
        info['arms']=copy.deepcopy(config['arm_pairs'])
    return parts, info


def split_seated_cat(layers: list[Layer], info: dict, use_lama: bool = True) -> tuple[list[Layer], dict]:
    body = next(p for p in layers if p.name == 'cat_body')
    shape = body.mask.shape
    defs = [
        {'name': 'cat_body', 'role': 'animal_body', 'label': '身体', 'depth': 140},
        {'name': 'cat_hind_left', 'role': 'leg_whole', 'label': '画面左后腿与臀部', 'depth': 170,
         'parent': 'cat_body', 'pivot': [334, 664], 'cap': [334, 658, 35, 48],
         'polygon': [[242,561],[323,552],[374,622],[370,756],[362,873],[386,947],[321,952],[267,897],[238,782]]},
        {'name': 'cat_hind_right', 'role': 'leg_whole', 'label': '画面右后腿可见部分', 'depth': 170,
         'parent': 'cat_body', 'pivot': [737,688], 'cap': [735,681,28,45],
         'polygon': [[718,597],[780,575],[808,706],[790,848],[754,927],[710,948],[707,867],[722,773]]}]
    pieces = partition_regions(body.rgba, body.mask, defs, use_lama)
    # Complete a torso outline, not a copy of both detached haunches.
    torso = pieces[0]
    known=torso.mask.copy()
    support=known|curved_region(shape,[[371,556],[457,522],[583,516],[701,535],[748,615],
        [755,743],[717,873],[680,935],[578,952],[464,944],[373,905],[330,819],[323,691],[340,601],[371,556]])
    torso.rgba,torso.completed=_complete_rgb(body.rgba,known,support,use_lama)
    torso.visibility_mask=known.astype(np.uint8)*255
    if body.visibility_mask is not None:torso.visibility_mask &= body.visibility_mask
    torso.meta['label'] = '身体'
    layers = [p for p in layers if p is not body]+pieces
    for p in layers:
        if p.name.startswith('cat_front'):
            p.role = 'leg_whole'
            p.meta['label'] = '画面左前腿' if p.name.endswith('left') else '画面右前腿'
            p.parent = 'cat_body'
    names = [p.name for p in layers]
    return layers, reset_records(info, [{'name': 'cat', 'kind': 'seated_quadruped', 'parts': names,
                                        'inferred_hidden_parts': []}])


def split_doraemon_arms(src: np.ndarray, layers: list[Layer], info: dict, use_lama: bool = True) -> tuple[list[Layer], dict]:
    body = next(p for p in layers if p.name == 'doraemon_body')
    left = next(p for p in layers if p.name == 'doraemon_folded_arm')
    shape = body.mask.shape
    # The visible white hand above the left sleeve belongs to the farther arm.
    hand = polygon(shape, [[308,557],[318,528],[352,516],[381,522],[406,541],[417,565]]) & left.mask
    far = polygon(shape, [[416,571],[665,568],[667,680],[631,701],[539,708],[470,687],[409,663]]) & body.mask & ~left.mask
    far |= hand
    combined = src.copy()
    defs = [
        {'name':'doraemon_arm_far', 'role':'arm_whole', 'label':'远侧完整手臂', 'depth':85,
         'parent':'doraemon_body','pivot':[645,590], 'cap':[645,590,25,30],
         'support_polygon':[[647,575],[660,597],[658,662],[633,692],[555,695],[475,671],[398,631],[335,564],[339,541],[375,549],[453,592],[552,608],[622,608],[647,575]]},
    ]
    p = partition_regions(combined, far, defs, use_lama)[0]
    palm=ellipse(shape,[365,565,51,46])
    inferred=p.completed.copy()
    p.rgba[inferred,:3]=[16,159,214]
    p.rgba[inferred&palm,:3]=[249,249,246]
    palm_rim=palm&~(cv2.erode(palm.astype(np.uint8),np.ones((5,5),np.uint8))>0)
    p.rgba[inferred&palm_rim,:3]=[23,26,27]
    p.parent = 'doraemon_body'
    # Complete the blue upper torso without copying a sleeve contour into it.
    known = body.mask & ~far
    body.rgba, body.completed = _complete_rgb(body.rgba, known, body.mask, use_lama)
    rgb = body.rgba[..., :3].astype(int)
    blue = known & (rgb[...,2]>150) & (rgb[...,0]<40) & (rgb[...,1]>90)
    if blue.any():
        body.rgba[body.completed,:3] = np.median(body.rgba[blue,:3],axis=0).astype(np.uint8)
    left.rgba[hand] = 0
    left.name = 'doraemon_arm_near'; left.parent='doraemon_body'
    left.meta['label']='近侧完整手臂'; left.role='arm_whole'; left.pivot=[286,556]
    p.depth[:] = int(left.z)+1
    # Far hand can be in front of the torso while the near arm crosses it.
    body.visibility_mask = np.where(far,0,255).astype(np.uint8)
    # Rebuild the clean torso/bib behind the crossing sleeves. The old body
    # prediction baked an entire blue sleeve across its chest.
    yy,xx=np.indices(shape)
    support=curved_region(shape,[[303,533],[475,531],[648,548],[666,600],[665,735],
        [645,814],[586,843],[479,850],[365,833],[291,800],[273,729],[275,621],[303,533]])
    visible=body.mask&~far&(yy<840)
    source_blue=(src[...,2]>150)&(src[...,0]<45)&(src[...,1]>90)&visible
    blue_colour=np.median(src[source_blue,:3],axis=0).astype(np.uint8)
    clean=np.zeros_like(body.rgba);clean[support,:3]=blue_colour;clean[support,3]=255
    bib=ellipse(shape,[435,662,160,154])&support
    clean[bib,:3]=[249,249,246]
    rim=bib&~(cv2.erode(bib.astype(np.uint8),np.ones((5,5),np.uint8))>0)
    clean[rim,:3]=[23,26,27]
    # Keep actual lower bib/pocket details; its previously hidden upper half
    # uses the same white material under the independently moving arms.
    observed=visible&(yy>=660)&support
    clean[observed]=src[observed]
    body.rgba=clean
    body.completed=support&~observed
    body.visibility_mask=(visible&support).astype(np.uint8)*255
    body.meta['completion']='clean torso and white bib continuation behind observed crossing arms'
    layers.append(p)
    return layers, reset_records(info, [{'name':'doraemon','kind':'biped',
                                        'parts':[q.name for q in layers], 'inferred_hidden_parts':[]}])


def finish_charizard(src: np.ndarray, layers: list[Layer], info: dict,
                     use_lama: bool = True) -> tuple[list[Layer], dict]:
    """Elbows follow the arms, not the tips of Mega Charizard's forearm fins."""
    from utils.animation_finish import _round_seam_end
    from utils.animation_parts import _cut
    by={p.name:p for p in layers};shape=src.shape[:2]
    outlines={
        'left':[[532,484],[585,476],[619,493],[621,533],[585,559],[561,584],[541,616],[516,618],[505,601],[514,575],[527,545]],
        'right':[[669,471],[704,472],[734,497],[748,529],[760,563],[774,586],[766,609],[741,604],[728,577],[704,555],[683,534]],
    }
    for side,joint,radius in [('left',[524.,603.],15.),('right',[758.,590.],18.)]:
        upper=by[f'arm_image_{side}_upper'];fore=by[f'arm_image_{side}_forearm']
        allowed=polygon(shape,outlines[side]);moved=upper.mask&~allowed
        # Preserve the real orange forearm fin and its source outline. Drop the
        # old cap that the automatic seam had placed on the fin instead of elbow.
        rgb=src[...,:3].astype(int)
        fin=(rgb[...,0]>125)&(rgb[...,0]>rgb[...,2]+30)&(src[...,3]>10)
        fin=cv2.dilate(fin.astype(np.uint8),np.ones((5,5),np.uint8))>0
        selected=moved&fin
        fore.rgba[selected]=src[selected]
        upper.rgba[moved]=0
        if side=='left':
            yy,xx=np.indices(shape)
            obsolete=np.hypot(xx-468.75,yy-574.9)<31
            fore.rgba[obsolete&~fin]=0
        tangent=np.array([1.,.12 if side=='left' else -.35]);tangent/=np.linalg.norm(tangent)
        for p in (upper,fore):
            _round_seam_end(p,np.asarray(joint),tangent,radius,radius*.85)
            _keep_connected(p)
            p.meta['joint_finish']='reviewed_elbow_with_forearm_fin'
        fore.pivot=joint
        record=next(r for r in info['arms'] if r['name']==f'arm_image_{side}')
        record['points'][1]=joint;record['provenance']='reviewed_anatomical_elbow_not_fin_tip'
        for r in info.get('rounded_limb_pairs',[]):
            if r['name']==record['name']:r['radii'][1]=radius*.85
    # The original source clearly shows this tail, which generic human classes
    # had largely omitted. It is one observed part, with only its covered base filled.
    tail_outline=[[134,337],[168,350],[166,459],[188,566],[224,650],[271,690],[360,704],
        [438,690],[473,650],[516,704],[493,790],[431,834],[340,814],[259,776],[203,722],[170,662],[146,585],[134,502]]
    wing_outline=[[522,129],[462,245],[465,347],[476,414],[511,450],[594,484],
        [536,571],[502,613],[456,541],[422,445],[446,596],[434,646],[416,696],
        [403,741],[400,790],[395,849],[370,832],[335,791],[293,724],[268,654],
        [242,572],[236,513],[254,449],[280,386],[327,335],[400,278],[469,179]]
    target=polygon(shape,tail_outline)&(src[...,3]>10)&~polygon(shape,wing_outline)&~by['earwear'].mask
    covered=np.logical_or.reduce([p.mask for p in layers if p.name.startswith('leg_image_') or p.name=='topwear'])
    target &= ~covered
    tail=partition_regions(src,target,[{'name':'tail','role':'animal_tail','label':'尾巴',
        'support_polygon':tail_outline+[tail_outline[0]],'cap':[520,773,48,38],
        'parent':'topwear','pivot':[520,773],'depth':200}],use_lama)[0]
    tail.parent='topwear';layers.append(tail)
    if tail.completed.any():
        nearest=ndi.distance_transform_edt(~target,return_distances=False,return_indices=True)
        material=cv2.GaussianBlur(src[...,:3][tuple(nearest)],(0,0),3).astype(float)
        distance=ndi.distance_transform_edt(~target)
        blend=np.exp(-distance/16)[...,None]
        continued=material*blend+np.array([249,163,89])*(1-blend)
        tail.rgba[tail.completed,:3]=np.round(continued[tail.completed]).astype(np.uint8)
    by['earwear'].parent='tail'
    # These legs have visible knees; the claws remain with the lower legs.
    info['legs']=[]
    for side,joint,end,radius in [('left',[488.,858.],[498.,949.],29.),('right',[783.,855.],[794.,916.],24.)]:
        old=next(p for p in layers if p.name==f'leg_image_{side}')
        yy,xx=np.indices(shape);axis=np.asarray(end)-np.asarray(joint);axis/=np.linalg.norm(axis)
        signed=(xx-joint[0])*axis[0]+(yy-joint[1])*axis[1]
        upper=_cut(old,old.mask&(signed<=0),f'leg_image_{side}_thigh',role='leg_thigh',
                   parent='topwear',pivot=old.pivot,meta={'label':'大腿','joint_finish':'reviewed_knee'})
        lower=_cut(old,old.mask&(signed>0),f'leg_image_{side}_shin',role='leg_shin',
                   parent=upper.name,pivot=joint,meta={'label':'小腿与脚爪','joint_finish':'reviewed_knee'})
        tangent=np.array([axis[1],-axis[0]])
        for p in (upper,lower):
            _round_seam_end(p,np.asarray(joint),tangent,radius,radius*.75)
            p.visibility_mask=old.visibility_mask
        layers=[p for p in layers if p is not old]+[upper,lower]
        name=f'leg_image_{side}'
        info['legs'].append({'name':name,'parts':[upper.name,lower.name],
                             'points':[upper.pivot,joint,end],'provenance':'reviewed_visible_knee'})
        info['rounded_limb_pairs'].append({'name':name,'parts':[upper.name,lower.name],'radii':[radius,radius*.75]})
    # The source and the author's completed anatomy differ around fin/neck
    # occlusions. Assign the actual observed pose to these named full PNGs,
    # keeping their completed hidden material behind reversible PSD masks.
    from utils.animation_character_repairs import rebuilt_charizard_tail
    layers,preferred=rebuilt_charizard_tail(src,layers)
    yy,xx=np.indices(shape)
    opaque=src[...,3]>10
    rgb=src[...,:3].astype(int)
    bright=(src[...,3]>=160)&(((rgb[...,0]>150)&(rgb[...,1]>80))|(rgb.min(axis=-1)>120))
    bright=cv2.dilate(bright.astype(np.uint8),np.ones((7,7),np.uint8))>0
    foreground=opaque&((yy<820)|(bright&(yy<982)))
    clean_tail=next(p for p in layers if p.name=='tail')
    tail_rgba=clean_tail.rgba.copy()
    restore_observed_pose(src,layers,foreground,exclude={'objects'},preferred=preferred)
    # Ownership determines the rest-pose mask, not the completed tail material.
    # Reapplying source RGB here used to paste the wing's dark contact edge back.
    clean_tail.rgba=tail_rgba
    by['objects'].depth[:]=250
    return layers,info


def restore_observed_pose(src: np.ndarray, layers: list[Layer], target: np.ndarray,
                          exclude: set[str] | None = None,
                          preferred: dict[str,np.ndarray] | None = None) -> None:
    """Keep complete PNG material but make the rest pose use real source pixels."""
    selected=[p for p in layers if p.name not in (exclude or set())]
    ys,xs=np.where(target);pixels=src[ys,xs,:3].astype(np.float32)
    best=np.full(len(xs),np.inf,np.float32);owner=np.zeros(len(xs),np.int16)
    for i,p in enumerate(selected):
        distance,near=ndi.distance_transform_edt(~p.mask,return_indices=True)
        colour=p.rgba[near[0,ys,xs],near[1,ys,xs],:3].astype(np.float32)
        cost=distance[ys,xs]+np.linalg.norm(colour-pixels,axis=-1)/30+p.z*.0001
        update=cost<best;owner[update]=i;best[update]=cost[update]
    for i,p in enumerate(selected):
        if preferred and p.name in preferred:
            owner[preferred[p.name][ys,xs]]=i
    for i,p in enumerate(selected):
        owned=np.zeros(target.shape,bool);owned[ys[owner==i],xs[owner==i]]=True
        p.rgba[owned]=src[owned]
        p.visibility_mask=owned.astype(np.uint8)*255
        p.meta['psd_visibility']='observed source pose; full completed PNG retained'


def split_pixel_sheet(src: np.ndarray, layers: list[Layer], info: dict,
                      use_lama: bool = True) -> tuple[list[Layer], dict]:
    """Short sprite limbs stay whole; each of fifteen characters is independent."""
    by = {p.name:p for p in layers}
    shape = src.shape[:2]; yy, xx = np.indices(shape)
    columns = [(35,238),(209,430),(414,613),(609,801),(792,984)]
    rows = [(76,369),(380,660),(674,943)]
    jaws = [[241,241,234,239,242],[519,522,529,526,524],[816,814,823,823,821]]
    parts, records = [], []
    for row, (y1,y2) in enumerate(rows):
        for col, (x1,x2) in enumerate(columns):
            prefix = f'character_{row+1}{col+1}'
            scope = (xx>=x1)&(xx<x2)&(yy>=y1)&(yy<y2)&(src[...,3]>10)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(scope.astype(np.uint8),8)
            scope=labels==(1+int(stats[1:,4].argmax()))
            cx = float(np.median(np.where(scope)[1]))
            arm_mask = cv2.dilate(by['handwear'].mask.astype(np.uint8),np.ones((5,5),np.uint8))>0
            leg_mask = cv2.dilate(by['legwear'].mask.astype(np.uint8),np.ones((5,5),np.uint8))>0
            if row==0 and col==0:
                arm_mask=polygon(shape,[[80,247],[103,254],[96,283],[90,311],[66,315],[62,291]])
                arm_mask|=polygon(shape,[[159,247],[180,250],[190,280],[195,315],[174,314],[163,287]])
            if row == 0 and col == 2:
                arm_mask = by['robot_arm_left'].mask | by['robot_arm_right'].mask
                leg_mask = by['robot_leg_left'].mask | by['robot_leg_right'].mask
            defs = [{'name':prefix+'_body','role':'clothing','label':f'{prefix} 身体与完整衣服','depth':140},
                    {'name':prefix+'_head','role':'animal_head','label':f'{prefix} 头发、帽子与头',
                     'mask':(yy<=jaws[row][col])|(by['hair'].mask&(np.abs(xx-cx)>34)),
                     'parent':prefix+'_body','depth':65}]
            if row==0 and col==3:
                defs.append({'name':prefix+'_tail','role':'animal_tail','label':'可见尾巴',
                             'polygon':[[624,307],[673,307],[681,339],[632,348],[620,337]],
                             'parent':prefix+'_body','depth':165})
            if 'tail' in by and (by['tail'].mask & scope).sum()>=12:
                if row==1 and col==2:
                    tailmask=polygon(shape,[[543,521],[593,521],[598,544],[577,551],[567,574],[554,582],[545,565]])
                else:tailmask=by['tail'].mask
                defs.append({'name':prefix+'_tail','role':'animal_tail','label':'可见尾巴',
                             'mask':tailmask,'parent':prefix+'_body','depth':165})
            for kind,mask,role in [('arm',arm_mask,'arm_whole'),('leg',leg_mask,'leg_whole')]:
                for side,half in [('left',xx<cx),('right',xx>=cx)]:
                    observed = scope & mask & half
                    if kind=='arm':observed &= yy>jaws[row][col]-8
                    if observed.sum()<12:
                        continue
                    ys,xs=np.where(observed); band=ys<=ys.min()+6
                    pivot=[float(np.median(xs[band])),float(ys.min()+3)]
                    radius=max(3,min(10,float(np.ptp(xs[band]))*.45))
                    defs.append({'name':prefix+'_'+kind+'_'+side,'role':role,
                                 'label':f'画面{"左" if side=="left" else "右"}{"臂" if kind=="arm" else "腿"}',
                                 'mask':observed,'parent':prefix+'_body','pivot':pivot,
                                 'cap':[*pivot,radius,radius*.85],'depth':75 if kind=='arm' else 150})
            current=partition_regions(src,scope,defs,use_lama)
            # Source/prediction alpha edges differ by one or two pixels.
            # Fill small cut-edge notches in the torso, discarding detached
            # outline scraps instead of leaving a ghost of removed limbs.
            torso=next(p for p in current if p.name==prefix+'_body')
            opened=cv2.morphologyEx(torso.mask.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
            envelope=cv2.dilate(opened,np.ones((3,3),np.uint8))>0
            torso.rgba[~envelope]=0
            _keep_connected(torso)
            parts.extend(current)
            records.append({'name':prefix,'kind':'short_limb_sprite','parts':[p.name for p in current],
                            'inferred_hidden_parts':[]})
    return parts,reset_records(info,records)


def split_paired_existing(layers: list[Layer], source_name: str, names: list[str],
                          split_x: int, parent: str, labels: list[str],
                          use_lama: bool = True) -> list[Layer]:
    original=next(p for p in layers if p.name==source_name)
    yy,xx=np.indices(original.mask.shape)
    defs=[]
    for i,name in enumerate(names):
        mask=(xx<split_x) if i==0 else (xx>=split_x)
        observed=original.mask&mask
        if not observed.any():continue
        y,x=np.where(observed); band=y<=y.min()+max(4,float(np.ptp(y))*.12)
        pivot=[float(np.median(x[band])),float(np.median(y[band]))]
        radius=max(4,min(24,float(np.ptp(x[band]))*.45))
        defs.append({'name':name,'role':original.role,'label':labels[i],'mask':mask,
                     'parent':parent,'pivot':pivot,'cap':[*pivot,radius,radius*.7],
                     'depth':int(original.z)})
    pieces=partition_regions(original.rgba,original.mask,defs,use_lama)
    for p in pieces:
        p.parent=parent
        if original.visibility_mask is not None:
            p.visibility_mask=original.visibility_mask.copy()
    # Any child prop (the sword) follows the appropriate visible arm.
    for p in layers:
        if p.parent==source_name:p.parent=names[0]
    return [p for p in layers if p is not original]+pieces


def separate_warrior_ribbon(src: np.ndarray, layers: list[Layer]) -> list[Layer]:
    """Restore the source's two gold arcs instead of leaving a detached top dash."""
    shape=src.shape[:2]
    outlines=[
        [[415,39],[416,59],[346,60],[306,75],[273,101],[250,133],[239,164],[240,193],
         [255,220],[285,242],[306,244],[306,264],[276,259],[248,242],[227,216],[219,182],
         [220,147],[235,113],[260,83],[292,61],[333,46]],
        [[514,56],[566,67],[616,87],[666,115],[711,148],[755,187],[787,229],[809,265],
         [818,292],[811,317],[789,336],[759,351],[720,361],[714,345],[751,334],[779,318],
         [795,296],[796,274],[777,238],[743,196],[701,159],[654,125],[608,99],[558,79],[515,72]],
        [[447,0],[476,0],[471,17],[489,31],[514,42],[516,62],[499,78],[474,86],
         [448,73],[435,55],[438,35],[461,27],[449,17]],
        [[635,220],[657,232],[684,232],[705,220],[709,235],[682,248],[652,246],[634,235]],
    ]
    region=np.logical_or.reduce([polygon(shape,p) for p in outlines])
    visible=region&(src[...,3]>10)
    if visible.sum()<200:raise ValueError('Warrior ribbon source region is empty')
    # Remove source-aligned duplicates in every predicted layer, keeping their
    # other hidden anatomy/material. This includes a small isolated top dash.
    for p in layers:
        if p.name=='clothing':
            count,labels,stats,_=cv2.connectedComponentsWithStats(p.mask.astype(np.uint8),8)
            for index,(x,y,w,h,area) in enumerate(stats[1:],1):
                if int(y+h)<275 and int(area)<6000:
                    p.rgba[labels==index]=0
        if p.name=='hair':
            p.rgba[~ellipse(shape,[575,181,58,69])]=0
        p.rgba[region]=0
    rgba=src.copy();rgba[~visible]=0
    ribbon=Layer('golden_ribbon',rgba,np.full(shape,230,np.uint8),'reviewed:source_decoration',
                 role='prop',parent='clothing',pivot=[513,262],
                 meta={'label':'金色弧形装饰与顶部饰件','provenance':'complete observed source decoration; no invented bridge'})
    return [p for p in layers if p.mask.any()]+[ribbon]


def finish_humanoid_roots(layers: list[Layer], info: dict) -> None:
    """Join a rounded proximal cap to its shaft without stacking another ball."""
    by={p.name:p for p in layers}
    for kind in ('arms','legs'):
        for record in info.get(kind,[]):
            names=record.get('parts',[])
            if len(names)!=2 or names[0] not in by:continue
            p=by[names[0]]
            root=np.asarray(p.pivot or record['points'][0],float)
            joint=np.asarray(record['points'][1],float)
            direction=joint-root;length=float(np.linalg.norm(direction))
            if length<18:continue
            direction/=length
            yy,xx=np.indices(p.mask.shape)
            along=(xx-root[0])*direction[0]+(yy-root[1])*direction[1]
            local=p.mask&(along<length*.38)
            y,x=np.where(local)
            if len(x)<30:continue
            hull=cv2.convexHull(np.column_stack([x,y]).astype(np.int32))
            support=np.zeros(p.mask.shape,np.uint8);cv2.fillConvexPoly(support,hull,1)
            distance=ndi.distance_transform_edt(~p.mask)
            added=(support>0)&~p.mask&(along<length*.24)&(distance<=min(10.,length*.10))
            if not added.any():continue
            known=p.mask.copy()
            old_visibility=p.visibility_mask.copy() if p.visibility_mask is not None else np.full(p.mask.shape,255,np.uint8)
            p.rgba,inferred=_complete_rgb(p.rgba,known,known|added,False)
            # Source rest pose remains stable; the complete PNG carries the
            # continuous attachment for animation.
            old_visibility[inferred]=0;p.visibility_mask=old_visibility
            p.completed=inferred
            p.meta['proximal_finish']='continuous cap-to-shaft contour; same-part material continuation'
            p.meta['proximal_added_pixels']=int(inferred.sum())

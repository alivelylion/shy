"""Finish existing animation pieces without diffusion or duplicate visible limbs."""

from __future__ import annotations

import copy
from io import BytesIO
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from utils.animation_parts import Layer, composite, load_layers
from utils.animation_review import _clean_engineer_face, _complete_round_joint, _curved_joint_region, largest
from utils.animation_integrity import polygon


ENGINEER = 'sc_gen_498564-meowa-watermark'
ARMOUR = 'sc_gen_45206-meowa-watermark'


def read_snapshot(archive: zipfile.ZipFile, name: str) -> tuple[list[Layer], dict]:
    prefix = name+'/'
    info = json.loads(archive.read(prefix+'关节与图层.json'))
    order = json.loads(archive.read(prefix+'_psd_stack.json'))
    step = max(1, 240//len(order))
    depths = {n: (len(order)-1-i)*step for i, n in enumerate(order)}
    core = {'name', 'file', 'source', 'role', 'pivot_xy', 'parent', 'bbox_xyxy'}
    layers = []
    for e in info['layers']:
        rgba = np.array(Image.open(BytesIO(archive.read(prefix+e['file']))).convert('RGBA'))
        layers.append(Layer(e['name'], rgba, np.full(rgba.shape[:2], depths[e['name']], np.uint8),
                            e['source'], pivot=e.get('pivot_xy'), parent=e.get('parent'), role=e['role'],
                            meta={k: v for k, v in e.items() if k not in core}))
    return layers, info


def read_delivery(folder: Path) -> tuple[list[Layer], dict]:
    """Read full PNG artwork, default visibility and reversible PSD masks."""
    from psd_tools import PSDImage
    info = json.loads((folder/'关节与图层.json').read_text(encoding='utf-8'))
    psd = PSDImage.open(folder/'动画拆层.psd')
    order = {p.name: i for i, p in enumerate(psd)}
    by_psd = {p.name: p for p in psd}
    step = max(1, 240//max(1, len(psd)))
    core = {'name', 'file', 'source', 'role', 'pivot_xy', 'parent', 'bbox_xyxy'}
    layers = []
    for e in info['layers']:
        rgba = np.array(Image.open(folder/e['file']).convert('RGBA'))
        visibility = None
        stored = by_psd[e['name']]
        if stored.has_mask():
            patch = np.array(stored.mask.topil().convert('L'))
            visibility = np.zeros(rgba.shape[:2], np.uint8)
            y, x = stored.mask.top, stored.mask.left
            visibility[y:y+patch.shape[0], x:x+patch.shape[1]] = patch
        layers.append(Layer(e['name'], rgba, np.full(rgba.shape[:2], (len(psd)-1-order[e['name']])*step, np.uint8),
            e['source'], pivot=e.get('pivot_xy'), parent=e.get('parent'), role=e['role'],
            meta={k: v for k, v in e.items() if k not in core}, visibility_mask=visibility))
    return layers, info


def smooth_alpha(layer: Layer, amount: float = .55) -> None:
    """Antialias the contour without blurring the illustrated RGB texture."""
    mask = layer.rgba[..., 3] > 0
    if not mask.any():
        return
    alpha = cv2.GaussianBlur(layer.rgba[..., 3].astype(np.float32), (0, 0), amount)
    alpha[alpha < 4] = 0
    alpha[alpha > 251] = 255
    added = (alpha > 0) & ~mask
    if added.any():
        near = ndi.distance_transform_edt(~mask, return_distances=False, return_indices=True)
        layer.rgba[added, :3] = layer.rgba[..., :3][tuple(near)][added]
    layer.rgba[..., 3] = np.round(alpha).astype(np.uint8)
    layer.rgba[alpha == 0] = 0


def _move_to_clothing(part: Layer, removed: np.ndarray, clothes: list[Layer]) -> None:
    if not removed.any() or not clothes:
        return
    owner = next((p for p in clothes if p.name == 'bottomwear'), clothes[0])
    uncovered = removed & ~np.logical_or.reduce([p.mask for p in clothes])
    owner.rgba[uncovered] = part.rgba[uncovered]
    part.rgba[removed] = 0


def clean_engineer_legs(layers: list[Layer]) -> None:
    by = {p.name: p for p in layers}
    clothes = [p for p in layers if p.name in {'topwear', 'bottomwear'}]
    thigh = by['leg_image_right_thigh']
    allowed = polygon(thigh.mask.shape, [[540, 640], [621, 640], [652, 707], [652, 745],
        [704, 759], [711, 811], [692, 863], [641, 873], [610, 839], [552, 820], [532, 773]])
    _move_to_clothing(thigh, thigh.mask & ~allowed, clothes)
    for side in ('left', 'right'):
        p = by[f'leg_image_{side}_thigh']
        _move_to_clothing(p, p.mask & ~largest(p.mask), clothes)


def repair_engineer_right_boot(layers: list[Layer], info: dict, original: Layer) -> None:
    """Keep the complete boot tongue/cuff on the shin, below the trouser leg."""
    by = {p.name: p for p in layers}
    thigh, shin = (by[f'leg_image_right_{part}'] for part in ('thigh', 'shin'))
    if thigh.meta.get('boot_boundary_review') == 'complete_boot_on_shin':
        return
    shape = thigh.mask.shape
    yy, xx = np.indices(shape)
    # Follow the top of the leather cuff and the raised gold boot tongue.
    # A bone bisector through the gold plate splits the same boot in two.
    boot = _curved_joint_region(shape, [574, 824], [
        [[586, 813], [611, 802], [637, 796]],
        [[626, 796], [618, 793], [623, 783]],
        [[638, 775], [660, 761], [681, 749]],
    ], [[720, 740], [840, 740], [840, 1024], [530, 1024], [530, 834]])
    upper_boot = boot & (yy < 855)
    # Recover the author's continuous texture, avoiding the old inferred seam.
    shin.rgba[yy < 855] = 0
    shin.rgba[upper_boot] = original.rgba[upper_boot]
    # Include the thin antialiased outline of the tongue in the removal.
    remove = cv2.dilate(boot.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    thigh.rgba[remove] = 0
    before = thigh.rgba.copy()
    known = thigh.mask.copy()
    rgb = thigh.rgba[..., :3].astype(np.int16)
    material = known & (yy > 715) & (yy < 790) & (xx > 565) & (xx < 620)
    material &= rgb.max(axis=-1) < 90
    # Complete only the hidden trouser end with trouser material; the detailed
    # boot outline stays on the front layer and moves with the entire boot.
    cap = _curved_joint_region(shape, [553, 755], [
        [[554, 780], [558, 803], [574, 813]],
        [[592, 829], [615, 830], [630, 814]],
        [[647, 796], [653, 778], [643, 755]],
    ], [[553, 755]])
    target = (known & (yy < 755)) | cap
    trusted = material & (before[..., 3] >= 220)
    if trusted.sum() < 12:
        raise ValueError('The engineer boot repair requires clean trouser material')
    nearest = ndi.distance_transform_edt(~trusted, return_distances=False, return_indices=True)
    texture = cv2.GaussianBlur(before[..., :3][tuple(nearest)], (0, 0), 2.5)
    alpha = cv2.GaussianBlur(target.astype(np.float32)*255, (0, 0), .55)
    alpha[alpha < 4] = 0
    alpha[alpha > 251] = 255
    old_edge = ndi.distance_transform_edt(known) <= 4
    replace = (yy >= 755) & ((~known) | old_edge | (yy >= 790)) & (alpha > 0)
    thigh.rgba[replace, :3] = texture[replace]
    thigh.rgba[yy >= 755, 3] = np.round(alpha[yy >= 755]).astype(np.uint8)
    edge = target & (ndi.distance_transform_edt(target) <= 2) & (yy >= 755)
    thigh.rgba[edge, :3] = [23, 23, 25]
    thigh.rgba[thigh.rgba[..., 3] == 0] = 0
    added = int((thigh.mask & ~known).sum())
    if thigh.visibility_mask is not None:
        thigh.visibility_mask[thigh.mask & ~known] = 255
    shin.pivot = [615., 802.]
    for p in (thigh, shin):
        p.meta['boot_boundary_review'] = 'complete_boot_on_shin'
    thigh.meta['boot_joint_inferred_pixels'] = added
    for record in info.get('legs', []):
        if record['name'] == 'leg_image_right':
            record['points'][1] = shin.pivot.copy()


def _finish_reference_contour(layer: Layer, target: np.ndarray, scope: np.ndarray,
                              material: Layer, trusted: np.ndarray, repaint: np.ndarray) -> None:
    """Finish one reviewed attachment with its own material and one outer rim."""
    trusted = trusted & (material.rgba[..., 3] >= 220)
    if trusted.sum() < 12:
        raise ValueError(f'Clean material is required to finish {layer.name}')
    original = layer.mask.copy()
    nearest = ndi.distance_transform_edt(~trusted, return_distances=False, return_indices=True)
    texture = cv2.GaussianBlur(material.rgba[..., :3][tuple(nearest)], (0, 0), 2)
    alpha = cv2.GaussianBlur(target.astype(np.float32)*255, (0, 0), .55)
    alpha[alpha < 4] = 0
    alpha[alpha > 251] = 255
    fill = scope & (alpha > 0) & (repaint | ~original)
    layer.rgba[fill, :3] = texture[fill]
    layer.rgba[scope, 3] = np.round(alpha[scope]).astype(np.uint8)
    rim = scope & target & (ndi.distance_transform_edt(target) <= 2)
    layer.rgba[rim, :3] = [23, 23, 25]
    layer.rgba[layer.rgba[..., 3] == 0] = 0
    added = layer.mask & ~original
    layer.completed = added if layer.completed is None else layer.completed | added
    if layer.visibility_mask is not None:
        layer.visibility_mask[added] = 255


def repair_engineer_mechanical_root(layers: list[Layer], original: Layer) -> None:
    """Complete the proximal shell hidden by the foreground fist."""
    upper = next(p for p in layers if p.name == 'arm_image_right_upper')
    if upper.meta.get('mechanical_root_finish') == 'continuous_proximal_shell_v3':
        return
    yy, xx = np.indices(upper.mask.shape)
    old = upper.mask.copy()
    # The whole left edge was an occlusion contour of the other fist. Continue
    # the shoulder shell below that edge, rather than rounding one small notch.
    root = _curved_joint_region(old.shape, [548, 433], [
        [[549, 460], [566, 485], [595, 501]],
        [[605, 508], [620, 515], [634, 511]],
        [[637, 494], [627, 475], [614, 459]],
        [[592, 444], [568, 434], [548, 433]],
    ], [[548, 433]])
    target = old | root
    smooth = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    smooth = cv2.GaussianBlur(smooth.astype(np.float32), (0, 0), 1.4) >= .5
    soften = (xx < 600) | ((xx > 590) & (xx < 650) & (yy > 500))
    target[soften] = smooth[soften]
    rgb = original.rgba[..., :3].astype(int)
    trusted = original.mask & (xx > 557) & (xx < 589) & (yy > 414) & (yy < 437)
    trusted &= (rgb.max(axis=-1) < 125) & (rgb.min(axis=-1) > 35)
    trusted &= ndi.distance_transform_edt(original.mask) > 5
    trusted &= original.rgba[..., 3] >= 220
    internal_edge = old & (ndi.distance_transform_edt(old) <= 8) & (ndi.distance_transform_edt(target) > 4)
    scope = cv2.dilate(root.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    scope |= (xx > 535) & (xx < 605) & (yy > 395) & (yy < 505)
    scope |= (xx > 590) & (xx < 650) & (yy > 500) & (yy < 527)
    _finish_reference_contour(upper, target, scope, original, trusted, internal_edge)
    inferred = scope & target & (~old | internal_edge)
    radius = np.sqrt(((xx-585)/38)**2+((yy-444)/47)**2)
    shade = np.clip((radius-.55)/.45, 0, 1)
    dark = np.percentile(original.rgba[trusted, :3], 12, axis=0)
    colour = upper.rgba[..., :3]*(1-shade[..., None])+dark*shade[..., None]
    upper.rgba[inferred, :3] = np.round(colour[inferred]).astype(np.uint8)
    # Colour the antialiased outside of the new rim as well, avoiding a pale
    # prediction fringe around an otherwise continuous black outline.
    rim = scope & (upper.rgba[..., 3] > 0) & (ndi.distance_transform_edt(target) <= 2)
    upper.rgba[rim, :3] = [23, 23, 25]
    upper.meta['mechanical_root_finish'] = 'continuous_proximal_shell_v3'


def finish_engineer_reference(layers: list[Layer], info: dict, original: Layer) -> None:
    """Apply the bear reference's smooth roots and continuous material joins."""
    by = {p.name: p for p in layers}
    left, shin = (by[f'leg_image_left_{part}'] for part in ('thigh', 'shin'))
    if left.meta.get('reference_finish') == 'bear_round_underlaps_v1':
        return
    yy, xx = np.indices(left.mask.shape)
    boot = _curved_joint_region(left.mask.shape, [256, 793], [
        [[267, 768], [279, 746], [298, 740]],
        [[306, 743], [309, 751], [309, 763]],
        [[309, 749], [313, 738], [323, 736]],
        [[331, 737], [333, 730], [340, 734]],
        [[352, 748], [373, 759], [391, 753]],
    ], [[420, 750], [420, 1000], [200, 1000], [200, 800]])
    shin.rgba[yy < 840] = 0
    upper_boot = boot & (yy < 840)
    shin.rgba[upper_boot] = original.rgba[upper_boot]
    left.rgba[cv2.dilate(boot.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0] = 0
    palette = Layer('trouser_material', left.rgba.copy(), left.depth, left.source)
    rgb = palette.rgba[..., :3].astype(int)
    trusted = palette.mask & (yy > 708) & (yy < 735) & (xx > 330) & (xx < 375)
    trusted &= (rgb.max(axis=-1) < 90) & (ndi.distance_transform_edt(palette.mask) > 4)
    trusted &= (rgb.min(axis=-1) > 24) & (rgb.max(axis=-1)-rgb.min(axis=-1) < 15)
    if trusted.sum() < 12:
        raise ValueError('The left trouser attachment needs a clean dark cloth sample')
    palette.rgba[trusted, :3] = np.median(palette.rgba[trusted, :3], axis=0).astype(np.uint8)
    knee = ((xx-317)/28)**2+((yy-750)/28)**2 <= 1
    seed = (left.mask | knee).astype(np.uint8)
    points = cv2.findNonZero(seed)
    target = np.zeros_like(seed)
    cv2.fillConvexPoly(target, cv2.convexHull(points), 1)
    _finish_reference_contour(left, target > 0, np.ones_like(seed, bool), palette, trusted, target > 0)
    shin.pivot = [310., 750.]
    left.meta['reference_finish'] = 'bear_round_underlaps_v1'
    shin.meta['boot_boundary_review'] = 'complete_boot_on_shin'
    for record in info.get('legs', []):
        if record['name'] == 'leg_image_left':
            record['points'][1] = shin.pivot.copy()

    # Join the old hip disk into the existing trouser contour, removing the
    # internal arc instead of stacking another outlined circle on the root.
    right = by['leg_image_right_thigh']
    palette = Layer('trouser_material', right.rgba.copy(), right.depth, right.source)
    points = cv2.findNonZero((right.mask & (yy < 735)).astype(np.uint8))
    hull = np.zeros_like(seed)
    cv2.fillConvexPoly(hull, cv2.convexHull(points), 1)
    target = right.mask | ((hull > 0) & (yy < 735))
    trusted = palette.mask & (yy > 710) & (yy < 765) & (xx > 565) & (xx < 620)
    trusted &= ndi.distance_transform_edt(palette.mask) > 5
    _finish_reference_contour(right, target, yy < 735, palette, trusted, yy < 710)
    right.meta['reference_finish'] = 'bear_round_underlaps_v1'

    # The sleeve has a flesh-coloured upper-arm stub under the moving
    # forearm. The old inferred white disk is not an actual second cuff.
    upper, fore = (by[f'arm_image_left_{part}'] for part in ('upper', 'forearm'))
    plug = _curved_joint_region(left.mask.shape, [444, 471], [
        [[420, 469], [393, 486], [388, 507]],
        [[382, 523], [390, 532], [402, 532]],
        [[425, 535], [446, 527], [447, 507]],
        [[450, 490], [449, 479], [444, 471]],
    ], [[444, 471]])
    rgb = fore.rgba[..., :3].astype(int)
    trusted = fore.mask & (xx < 424) & (yy > 480) & (yy < 523)
    trusted &= (rgb[..., 0] > 180) & (rgb[..., 0] > rgb[..., 1]+12) & (rgb[..., 1] > 110)
    plug |= upper.mask & trusted
    _finish_reference_contour(upper, upper.mask | plug, plug, fore, trusted, plug)
    upper.meta['reference_finish'] = 'bear_round_underlaps_v1'


def repair_reviewed_fragments(name: str, layers: list[Layer]) -> None:
    """Return four visually checked stray components to their real owners."""
    routes = {
        '301702_xiali_p': ('person1_leg_right_shin', 'person1_leg_left_shin'),
        '302001_bolinyidong_p': ('arm_right_forearm', 'neckwear'),
        'f8bdb4d2d17cc6a56bbeb911b2864245': ('leg_image_right_thigh', 'bottomwear'),
        ARMOUR: ('red_arm_image_right_upper', 'blue_arm_image_left_upper'),
    }
    if name not in routes:
        return
    by = {p.name: p for p in layers}
    part, owner = (by[n] for n in routes[name])
    removed = part.mask & ~largest(part.mask)
    copy_pixels = removed & (part.rgba[..., 3] > owner.rgba[..., 3])
    owner.rgba[copy_pixels] = part.rgba[copy_pixels]
    part.rgba[removed] = 0
    part.meta['stray_component_returned_to'] = owner.name


def _redraw_outline(layer: Layer, scope: np.ndarray, width: float = 2) -> None:
    distance = ndi.distance_transform_edt(layer.mask)
    edge = scope & layer.mask & (distance <= width)
    layer.rgba[edge, :3] = [22, 23, 27]


def clean_watermarks(name: str, layers: list[Layer]) -> list[Layer]:
    layers = [p for p in layers if p.name not in {'source_text', 'poster_text'}]
    by = {p.name: p for p in layers}
    shape = layers[0].mask.shape
    yy, xx = np.indices(shape)
    if name == 'f8bdb4d2d17cc6a56bbeb911b2864245':
        lettering = ((xx >= 582) & (xx <= 822) & (yy >= 25) & (yy <= 95)) | ((xx < 385) & (yy > 855))
        for p in layers:
            p.rgba[lettering] = 0
    elif name == ENGINEER:
        foot = by['leg_image_right_shin']
        # Only the real boot remains. Logo pixels extend well beyond its sole.
        sole = polygon(shape, [[574, 928], [791, 928], [804, 954], [805, 973],
            [793, 982], [752, 987], [685, 988], [628, 984], [588, 977], [579, 960]])
        foot.rgba[(yy >= 936) & ~sole] = 0
        text = (yy >= 941) & (yy < 988) & foot.mask & (foot.rgba[..., :3].max(axis=-1) < 95)
        text = cv2.dilate(text.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        text &= foot.mask & (yy >= 940)
        rgb = cv2.inpaint(foot.rgba[..., :3], text.astype(np.uint8)*255, 5, cv2.INPAINT_TELEA)
        foot.rgba[text, :3] = rgb[text]
        band = _curved_joint_region(shape, [589, 950], [
            [[645, 966], [737, 980], [798, 954]],
            [[801, 959], [803, 967], [802, 973]],
            [[756, 989], [651, 991], [589, 975]],
        ], [[589, 950]]) & foot.mask
        shade = np.clip((yy-950)/38, 0, 1)
        gold = np.round(np.array([202, 159, 84])*(1-shade[..., None])+
                        np.array([166, 116, 56])*shade[..., None]).astype(np.uint8)
        foot.rgba[band, :3] = gold[band]
        edge = band & (ndi.distance_transform_edt(band) < 2.5)
        foot.rgba[edge, :3] = [24, 24, 26]
        _redraw_outline(foot, yy >= 937, 2.5)
        foot.meta['watermark_removed'] = 'non-character lettering over the boot and outside its true sole'
    elif name == ARMOUR:
        for n in ('blue_leg_image_left_shin', 'blue_leg_image_right_shin'):
            p = by[n]
            clean = cv2.morphologyEx(p.mask.astype(np.uint8), cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
            clean = largest(clean > 0)
            # Restrict the logo area; keep the knee/ankle armour unmodified.
            p.rgba[(yy > 925) & ~clean] = 0
            logo = p.mask & (yy > 932) & (p.rgba[..., :3].max(axis=-1) < 100)
            logo = cv2.dilate(logo.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            logo &= p.mask & (yy > 930)
            rgb = cv2.inpaint(p.rgba[..., :3], logo.astype(np.uint8)*255, 4, cv2.INPAINT_TELEA)
            p.rgba[logo, :3] = rgb[logo]
            _redraw_outline(p, yy > 927, 1.5)
            p.meta['watermark_removed'] = 'logo below and across the blue boot'
    elif name.startswith('Keep-the-car-'):
        left, right = by['tyre_left'], by['tyre_right']
        # The original rear-view car is symmetric. Reuse the clean real tyre.
        right.rgba = np.ascontiguousarray(left.rgba[:, ::-1])
        right.meta['watermark_removed'] = 'right tyre restored from the clean symmetric left tyre'
        body = by['car_body']
        body.rgba[(xx > 735) & (yy >= 800)] = 0
    return layers


def original_head_visibility(layers: list[Layer], originals: list[Layer]) -> None:
    """Keep complete PNGs; reversible PSD masks resolve local head occlusion."""
    raw = {p.name: p for p in originals}
    heads = [p for p in layers if p.name in {'hair', 'face', 'eyes', 'ears', 'nose', 'mouth', 'headwear', 'neck'}
             and p.name in raw]
    for p in heads:
        hide = np.zeros(p.mask.shape, bool)
        for q in heads:
            if p is q or p.z >= q.z:
                continue
            hide |= p.mask & q.mask & (q.rgba[..., 3] >= 245) & (raw[p.name].depth > raw[q.name].depth)
        if hide.any():
            visibility = np.full(p.mask.shape, 255, np.uint8)
            visibility[hide] = 0
            p.visibility_mask = visibility
            p.meta['psd_visibility'] = 'reversible original-depth occlusion mask; complete PNG retained'


def merge_clothes(name: str, layers: list[Layer], info: dict, *, use_example_rules: bool = False) -> list[Layer]:
    if not use_example_rules and info.get('detected_people', 0) != 1:
        info.setdefault('warnings', []).append('Clothing retained separately: a single human pose was not confirmed.')
        return layers
    animal = use_example_rules and (name.startswith(('006', '025')) or name in {'JRPG-meowa-watermark', 'tangka', '303401_xingzhiyan_p'})
    if animal:
        return layers
    groups = []
    if use_example_rules and name == ARMOUR:
        for colour in ('red', 'blue'):
            groups.append((colour+'_clothing', [p for p in layers if p.name == colour+'_torso' or
                p.name.startswith(colour+'_skirt_') and p.role != 'cloth_inner']))
    else:
        selected = [p for p in layers if p.name in {'topwear', 'bottomwear'} or p.role == 'garment']
        if any(p.role == 'cloth_outer' for p in layers):
            selected += [p for p in layers if p.role == 'cloth_outer']
        if selected:
            groups.append(('clothing', selected))
    replacements = {}
    for new_name, pieces in groups:
        if not pieces:
            continue
        rgba = composite(pieces)
        depth = np.full(rgba.shape[:2], 255, np.uint8)
        for p in pieces:
            depth[p.mask] = np.minimum(depth[p.mask], p.depth[p.mask])
        merged = Layer(new_name, rgba, depth, 'merged:clothing',
                       role='clothing', meta={'label': '完整衣服', 'merged_from': [p.name for p in pieces]})
        smooth_alpha(merged, .45)
        layers = [p for p in layers if p not in pieces]+[merged]
        for p in pieces:
            replacements[p.name] = new_name
    for p in layers:
        if p.parent in replacements:
            p.parent = replacements[p.parent]
    info['clothing_merge'] = replacements
    return layers


def _joint_radius(whole: np.ndarray, centre: np.ndarray, length: float) -> float:
    yy, xx = np.indices(whole.shape)
    distance = ndi.distance_transform_edt(whole)
    local = (np.hypot(xx-centre[0], yy-centre[1]) < max(8, length*.18)) & whole
    if not local.any():
        return 4.
    radius = float(np.percentile(distance[local], 85))
    return max(4., min(radius, length*.3, 42.))


def _cap(layer: Layer, centre: np.ndarray, radius: float, outline: bool = False) -> int:
    yy, xx = np.indices(layer.mask.shape)
    original = layer.mask
    # Some old hip/shoulder pivots are outside the visible cut. Snap the cap
    # into its own material rather than exporting a detached circular speck.
    core = original & (ndi.distance_transform_edt(original) >= min(3, radius*.4))
    if core.any():
        ys, xs = np.where(core)
        index = np.argmin((xs-centre[0])**2+(ys-centre[1])**2)
        nearest = np.array([xs[index], ys[index]], float)
        if np.linalg.norm(nearest-centre) > radius*.4:
            centre = nearest
    # Use the material from this end of this piece, away from its black rim.
    near = np.hypot(xx-centre[0], yy-centre[1]) < max(12, radius*3)
    material = near & (ndi.distance_transform_edt(layer.mask) > max(1, radius*.12))
    if (material & (layer.rgba[..., 3] >= 220)).sum() < 12:
        material = layer.mask
    layer.rgba, added = _complete_round_joint(layer.rgba, layer.mask, tuple(centre), radius, material)
    inferred = layer.mask & ~original
    if not outline and inferred.any() and material.any():
        near_material = ndi.distance_transform_edt(~material, return_distances=False, return_indices=True)
        fill = cv2.GaussianBlur(layer.rgba[..., :3][tuple(near_material)], (0, 0), 2)
        layer.rgba[inferred, :3] = fill[inferred]
    layer.completed = inferred if layer.completed is None else layer.completed | inferred
    return added


def _round_seam_end(layer: Layer, centre: np.ndarray, tangent: np.ndarray,
                    half_width: float, extension: float) -> int:
    """Complete an ellipse fitted to the actual cut, keeping original texture."""
    original = layer.mask.copy()
    yy, xx = np.indices(original.shape)
    normal = np.array([-tangent[1], tangent[0]])
    dx, dy = xx-centre[0], yy-centre[1]
    across = dx*tangent[0]+dy*tangent[1]
    along = dx*normal[0]+dy*normal[1]
    radius = np.sqrt((across/half_width)**2+(along/extension)**2)
    coverage = np.clip((1-radius)*min(half_width, extension)+.5, 0, 1)
    added = (coverage > 0) & ~original
    if not added.any():
        return 0
    trusted = original & (layer.rgba[..., 3] >= 220) & (ndi.distance_transform_edt(original) > 2)
    trusted &= np.hypot(dx, dy) < max(16, half_width*2.5)
    if trusted.sum() < 12:
        trusted = original & (layer.rgba[..., 3] >= 220)
    if not trusted.any():
        return 0
    nearest = ndi.distance_transform_edt(~trusted, return_distances=False, return_indices=True)
    material = cv2.GaussianBlur(layer.rgba[..., :3][tuple(nearest)], (0, 0), 2)
    # Fill under the old antialiased cut as well: otherwise its faint alpha
    # leaves a pale straight seam between the original piece and its new cap.
    repair = (coverage > 0) & (layer.rgba[..., 3] < 250)
    a = layer.rgba[..., 3].astype(float)/255
    cap = coverage
    out_alpha = a+cap*(1-a)
    premult = layer.rgba[..., :3]*a[..., None]+material*cap[..., None]*(1-a[..., None])
    colour = np.round(premult/np.maximum(out_alpha[..., None], 1e-8)).clip(0, 255).astype(np.uint8)
    weak_rim = repair & (a < .7) & (coverage > .9)
    colour[weak_rim] = material[weak_rim]
    layer.rgba[repair, :3] = colour[repair]
    layer.rgba[repair, 3] = np.round(out_alpha[repair]*255).astype(np.uint8)
    layer.completed = repair if layer.completed is None else layer.completed | repair
    return int(added.sum())


def _fit_shared_seam(upper: Layer, distal: Layer, joint: np.ndarray,
                      reach: float) -> tuple[np.ndarray, np.ndarray, float] | None:
    yy, xx = np.indices(upper.mask.shape)
    overlap = upper.mask & distal.mask
    if overlap.sum() < 6:
        overlap = upper.mask & (cv2.dilate(distal.mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        overlap &= np.hypot(xx-joint[0], yy-joint[1]) < max(20, reach*.5)
    if overlap.sum() < 6:
        return None
    _, labels, stats, _ = cv2.connectedComponentsWithStats(overlap.astype(np.uint8), 8)
    near = overlap & (np.hypot(xx-joint[0], yy-joint[1]) < max(20, reach*.8))
    hits = np.bincount(labels[near], minlength=len(stats));hits[0] = 0
    selected = int(hits.argmax()) if hits.any() else int(1+stats[1:, 4].argmax())
    overlap = labels == selected
    ys, xs = np.where(overlap)
    points = np.column_stack([xs, ys]).astype(float)
    centre = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points-centre).T))
    tangent = vectors[:, -1]
    spread = (points-centre) @ tangent
    lo, hi = np.percentile(spread, [1, 99])
    if hi-lo < 4:
        return None
    centre += tangent*(lo+hi)/2
    return centre, tangent, float((hi-lo)/2+1.5)


def _round_flat_root(layer: Layer, direction: np.ndarray, length: float) -> None:
    """Cap a true flat root; keep naturally rounded shoulders and design tips."""
    direction = direction/max(float(np.linalg.norm(direction)), 1)
    tangent = np.array([-direction[1], direction[0]])
    yy, xx = np.where(layer.mask)
    points = np.column_stack([xx, yy]).astype(float)
    along = points @ direction
    across = points @ tangent
    start = float(along.min())
    local = along <= start+max(12, length*.22)
    if local.sum() < 20:
        return
    local_lo, local_hi = np.percentile(across[local], [2, 98])
    if layer.role == 'leg_thigh':
        # The hip-side crop is often jagged rather than one straight segment.
        # Fit its full cross section and replace only that proximal cut region.
        plane = start+min(8., length*.12)
        section = (along >= plane) & (along <= plane+max(6, length*.15))
        if section.sum() < 12:
            section = local
        lo, hi = np.percentile(across[section], [1, 99])
        centre = direction*plane+tangent*(lo+hi)/2
        half = max(4., float((hi-lo)/2+1.5))
        extension = max(4., min(half*.85, length*.35))
        grid_y, grid_x = np.indices(layer.mask.shape)
        dx, dy = grid_x-centre[0], grid_y-centre[1]
        q = dx*tangent[0]+dy*tangent[1]
        s = dx*direction[0]+dy*direction[1]
        ellipse = (q/half)**2+(s/extension)**2 <= 1.03
        layer.rgba[(s < 0) & ~ellipse] = 0
        _round_seam_end(layer, centre, tangent, half, extension)
        layer.pivot = centre.tolist()
        return
    contours, _ = cv2.findContours(layer.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    outline = cv2.approxPolyDP(contour, 1., True)[:, 0].astype(float)
    candidates = []
    for a, b in zip(outline, np.roll(outline, -1, axis=0)):
        edge = b-a; span = float(np.linalg.norm(edge))
        if span < max(9, .65*(local_hi-local_lo)):
            continue
        t = edge/span; centre = (a+b)/2
        if abs(float(t @ direction)) > .65 or min(float(a @ direction), float(b @ direction)) > start+max(3, length*.08):
            continue
        candidates.append((span, centre, t))
    if not candidates:
        return
    span, centre, tangent = max(candidates, key=lambda item: item[0])
    inward = np.array([-tangent[1], tangent[0]])
    if inward @ direction < 0:
        inward = -inward
    centre = centre+inward
    half = span/2+1.5
    _round_seam_end(layer, centre, tangent, half, max(4, min(half*.75, length*.28)))


def round_limb_pairs(layers: list[Layer], info: dict) -> list[dict]:
    by = {p.name: p for p in layers}
    finished = []
    for kind in ('arms', 'legs'):
        for record in info.get(kind, []):
            parts = [by[n] for n in record.get('parts', []) if n in by]
            if len(parts) != 2 or len(record.get('points', [])) < 3:
                continue
            upper, distal = parts
            points = np.asarray(record['points'], float)
            root, joint, end = points[:3]
            whole = upper.mask | distal.mask
            length = float(np.linalg.norm(joint-root))
            if length < 10:
                continue
            r0 = _joint_radius(whole, root, length)
            r1 = _joint_radius(whole, joint, min(length, np.linalg.norm(end-joint)))
            if not upper.meta.get('joint_finish'):
                seam = _fit_shared_seam(upper, distal, joint, min(length, np.linalg.norm(end-joint)))
                _round_flat_root(upper, joint-root, length)
                if seam is not None:
                    centre, tangent, half = seam
                    extension = max(4, min(half*.8, length*.3, np.linalg.norm(end-joint)*.3))
                    count = _round_seam_end(upper, centre, tangent, half, extension)
                    upper.meta['rounded_inferred_pixels'] = count
                    count = _round_seam_end(distal, centre, tangent, half, extension)
                    distal.meta['rounded_inferred_pixels'] = count
                    distal.pivot = centre.tolist()
                    record['points'][1] = centre.tolist()
                    r1 = min(half, extension)
                else:
                    _cap(upper, joint, r1)
                    _cap(distal, joint, r1)
                # Keep the original detailed outline; soften only pixel steps.
                smooth_alpha(upper, .55)
                smooth_alpha(distal, .55)
            for p in parts:
                p.meta.update({'joint_finish': 'rounded_overlapping_joint_ends',
                               'joint_round_radius_px': round(r1, 2)})
            finished.append({'name': record['name'], 'parts': record['parts'], 'radii': [r0, r1]})
    return finished


def round_whole_attachments(layers: list[Layer]) -> None:
    by = {p.name: p for p in layers}
    for p in layers:
        if p.role not in {'arm_whole', 'leg_whole', 'arm_visible'}:
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(p.mask.astype(np.uint8), 8)
        if sum(stats[1:, cv2.CC_STAT_AREA] > 40) != 1:
            continue  # Pixel-art sheets and combined short limbs stay intact.
        yy, xx = np.where(p.mask)
        if not len(xx) or len(xx) < 120:
            continue
        root = p.pivot
        if root is None:
            if not p.parent or p.parent not in by:
                continue
            band = yy <= np.percentile(yy, 12)
            root = [float(np.median(xx[band])), float(np.percentile(yy, 7))]
        centre = np.asarray(root, float)
        radius = _joint_radius(p.mask, centre, max(float(np.ptp(yy)), float(np.ptp(xx))))
        original = p.mask.copy()
        _cap(p, centre, radius, outline=False)
        smooth_alpha(p, .5)
        p.completed = p.mask & ~original
        p.meta['attachment_finish'] = 'rounded_root_without_forced_new_joint'


def add_whole_alternatives(layers: list[Layer], info: dict) -> list[Layer]:
    """Keep complete arm backups; split legs are delivered without duplicates."""
    layers[:] = [p for p in layers if p.role not in {'arm_whole_alternative', 'leg_whole_alternative'}]
    by = {p.name: p for p in layers}
    alternatives = []
    for r in info.get('arms', []):
        pieces = [by[n] for n in r.get('parts', []) if n in by]
        if len(pieces) != 2:
            continue
        name = r['name']+'_whole'
        # External rest-pose masks must not punch holes into the whole
        # backup artwork; only the two limb segments occlude one another.
        complete = [Layer(p.name, p.rgba, p.depth, p.source) for p in pieces]
        rgba = composite(complete)
        whole = Layer(name, rgba, np.full(rgba.shape[:2], int(min(p.z for p in pieces)), np.uint8),
                      'alternative:'+pieces[0].source, pivot=pieces[0].pivot,
                      parent=pieces[0].parent, role='arm_whole_alternative',
                      meta={'label': '完整手臂（备用）',
                            'default_visible': False, 'alternative_to': r['parts']})
        layers.append(whole)
        alternatives.append({'whole': name, 'segmented': r['parts']})
    info['whole_segment_alternatives'] = alternatives
    return layers


def finish(name: str, layers: list[Layer], info: dict, originals: list[Layer], *,
           use_example_rules: bool = False) -> tuple[list[Layer], dict]:
    info = copy.deepcopy(info)
    reviewed_cow = use_example_rules and name == '303401_xingzhiyan_p'
    if reviewed_cow:
        # The source is a saucer/cow scene. The author's generic human-face
        # prediction is not an actual scene object and must not become visible.
        layers = [p for p in layers if p.name not in {'face', 'eyes'}]
    raw = {p.name: p for p in originals}
    for p in layers:
        if p.source in raw and not p.meta.get('joint_finish') and not reviewed_cow:
            p.depth = raw[p.source].depth.copy()
    if use_example_rules and name == ENGINEER:
        layers = _clean_engineer_face(layers)
        clean_engineer_legs(layers)
    if use_example_rules:
        layers = clean_watermarks(name, layers)
        repair_reviewed_fragments(name, layers)
    layers = merge_clothes(name, layers, info, use_example_rules=use_example_rules)
    info['rounded_limb_pairs'] = round_limb_pairs(layers, info)
    round_whole_attachments(layers)
    if use_example_rules and name == ENGINEER:
        repair_engineer_right_boot(layers, info, raw['legwear'])
        finish_engineer_reference(layers, info, raw['legwear'])
        repair_engineer_mechanical_root(layers, raw['handwear'])
    # A normal stack needs reversible local occlusion masks. Keep the PNG
    # artwork complete instead of physically cutting face/hand/cloth holes.
    for p in layers:
        hide = np.zeros(p.mask.shape, bool)
        for q in layers:
            if p is q:
                continue
            opaque = q.mask & (q.rgba[..., 3] >= 245)
            if p.z < q.z:
                hide |= p.mask & opaque & (p.depth > q.depth)
            if p.completed is not None and q.role in {'clothing', 'animal_body', 'animal_head'}:
                hide |= p.completed & q.mask
        if hide.any():
            p.visibility_mask = np.where(hide, 0, 255).astype(np.uint8)
            p.meta['psd_visibility'] = 'reversible source-depth occlusion; full PNG retained'
    stack_depths = {p.name: min(254, max(1, round(p.z))) for p in layers}
    for p in layers:
        p.depth[:] = stack_depths[p.name]
    layers = add_whole_alternatives(layers, info)
    names = {p.name for p in layers}
    for key in ('arms', 'legs'):
        info[key] = [r for r in info.get(key, []) if all(n in names for n in r.get('parts', []))]
    info['whole_limb_layers'] = [p.name for p in layers if p.role in {'arm_whole', 'leg_whole'}]
    info['rule_scope'] = 'reviewed_examples' if use_example_rules else 'generic'
    info['method'] = ('Ours Full reused predictions, reviewed rounded interfaces, unified clothing, whole and segmented limbs'
                      if use_example_rules else 'Ours Full with automatic pose and contour refinement; visual review required')
    info['delivery_finish'] = '2026-10-30-reference-standard'
    return layers, info

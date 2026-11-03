"""Reopen all finished PNG/PSD deliveries and check active/alternate artwork."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'common')]
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '8'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+')
    args = parser.parse_args()
    import cv2
    import numpy as np
    from PIL import Image
    from psd_tools import PSDImage
    from scipy import ndimage as ndi
    from utils.animation_parts import Layer, composite, _on_checker, load_layers
    from utils.animation_finish import read_delivery, ENGINEER
    total = alternatives = joint_checks = folders_checked = 0
    fragments = []
    for folder in sorted((ROOT/'立绘拆分结果').iterdir()):
        if not folder.is_dir():
            continue
        if args.only and not folder.name.startswith(tuple(args.only)):
            continue
        layers, info = read_delivery(folder)
        psd = PSDImage.open(folder/'动画拆层.psd')
        stored = {p.name: p for p in psd}
        by = {p.name: p for p in layers}
        assert len(stored) == len(layers) == len(info['layers']), folder.name
        expected = {e['file'] for e in info['layers']} | {'关节与图层.json', '动画拆层.psd', '静态合成.png', '部件总览.png'}
        assert {p.name for p in folder.iterdir()} == expected, folder.name
        for p in layers:
            assert p.parent is None or p.parent in by, (folder.name, p.name, p.parent)
            pixel = stored[p.name]
            np.testing.assert_array_equal(np.array(pixel.topil().convert('RGBA')),
                                          np.array(Image.fromarray(p.rgba).crop(pixel.bbox)))
            assert pixel.visible == p.meta.get('default_visible', True), (folder.name, p.name)
            if p.role in {'arm_upper', 'arm_forearm', 'leg_thigh', 'leg_shin', 'arm_visible'}:
                _, _, stats, _ = cv2.connectedComponentsWithStats(p.mask.astype(np.uint8), 8)
                areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True)
                if any(a >= 15 for a in areas[1:]):
                    fragments.append((folder.name, p.name, list(map(int, areas))))
        for record in info.get('whole_segment_alternatives', []):
            whole = by[record['whole']]
            assert not whole.meta['default_visible'] and not stored[whole.name].visible
            pieces = [by[n] for n in record['segmented']]
            unmasked = [Layer(p.name, p.rgba, p.depth, p.source) for p in pieces]
            np.testing.assert_allclose(whole.rgba.astype(int), composite(unmasked).astype(int), atol=1)
            alternatives += 1
        for key in ('arms', 'legs'):
            for record in info.get(key, []):
                assert all(n in by for n in record.get('parts', [])), (folder.name, key)
        for finished in info.get('rounded_limb_pairs', []):
            record = next(r for key in ('arms', 'legs') for r in info.get(key, []) if r['name'] == finished['name'])
            upper, distal = [by[n].rgba[..., 3] for n in record['parts']]
            x, y = record['points'][1]
            half = 10 if folder.name == ENGINEER else 3
            dy, dx = np.indices((half*2+1, half*2+1));xx, yy = dx-half+x, dy-half+y
            radius = 9. if folder.name == ENGINEER else min(2., finished['radii'][1]*.25)
            core = np.hypot(xx-x, yy-y) < radius
            fixed = cv2.remap(upper, xx.astype(np.float32), yy.astype(np.float32), cv2.INTER_LINEAR)
            for angle in (-60, -30, 0, 30, 60):
                matrix = cv2.invertAffineTransform(cv2.getRotationMatrix2D((x, y), angle, 1))
                sx = matrix[0, 0]*xx+matrix[0, 1]*yy+matrix[0, 2]
                sy = matrix[1, 0]*xx+matrix[1, 1]*yy+matrix[1, 2]
                moved = cv2.remap(distal, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR)
                assert (((fixed > 10) | (moved > 10))[core]).all(), (folder.name, record['name'], angle)
            joint_checks += 1
        if 'clothing' in by:
            assert 'topwear' not in by and 'bottomwear' not in by
        native = np.array(_on_checker(np.array(psd.composite(force=True).convert('RGBA')), 1024)).astype(int)
        preview = np.array(Image.open(folder/'静态合成.png').convert('RGB')).astype(int)
        expected_preview = np.array(_on_checker(composite(layers), 1024)).astype(int)
        np.testing.assert_array_equal(preview, expected_preview)
        difference = np.abs(native-preview)
        assert difference.max() <= 2, (folder.name, int(difference.max()), float(difference.mean()))
        if folder.name == ENGINEER:
            _, original = load_layers(ROOT/'立绘测试结果_OursFull_20261027'/folder.name)
            hair = next(p for p in original if p.name == 'hair')
            np.testing.assert_array_equal(by['hair'].rgba, hair.rgba)
            assert 'hair_back' not in by
            face = by['face'].rgba
            inner = ndi.distance_transform_edt(by['face'].mask) > 6
            brown = inner & (face[..., 0] > 120) & (face[..., 0] < 240) & (face[..., 1] > 65) & (face[..., 1] < 195)
            brown &= face[..., 0].astype(int) > face[..., 1].astype(int)+14
            assert not brown.any(), 'Baked facial shadow remains'
            assert not by['leg_image_right_shin'].rgba[994:, :, 3].any(), 'Boot lettering remains below sole'
            for side in ('left', 'right'):
                thigh = by[f'leg_image_{side}_thigh']
                rgb = thigh.rgba[..., :3].astype(int)
                yy = np.indices(thigh.mask.shape)[0]
                boot_colour = thigh.mask & (yy > 730) & (rgb[..., 0] > 95) & (rgb[..., 0]-rgb[..., 2] > 25)
                assert not boot_colour.any(), f'Leather cuff or gold boot tongue remains on the {side} thigh'
            assert not any(p.role == 'leg_whole_alternative' for p in layers), 'Unwanted full-leg backup'
        if folder.name == '45-meowa-watermark':
            assert 'source_text' not in by
            if info.get('standard_revision'):
                instances = info['anatomy_instances']
                assert len(instances) == 8
                for instance in instances:
                    roles = [by[n].role for n in instance['parts']]
                    expected_legs = 2 if instance['kind'] == 'bird' else (3 if instance['name'] == 'pig' else 4)
                    assert roles.count('leg_whole') == expected_legs, instance['name']
                    assert not instance['inferred_hidden_parts'], instance['name']
                    assert all(n in by for n in instance['parts'])
        if folder.name == 'f8bdb4d2d17cc6a56bbeb911b2864245':
            assert 'poster_text' not in by
            assert not any(p.rgba[25:96, 582:823, 3].any() for p in layers)
        if folder.name == '303401_xingzhiyan_p':
            assert 'face' not in by and 'eyes' not in by
        total += len(layers)
        folders_checked += 1
        print('DELIVERY_OK', folder.name, 'layers', len(layers), 'PSD_delta', int(difference.max()), flush=True)
    assert not fragments, fragments
    print('VERIFIED', folders_checked, 'groups', 'layers', total, 'whole_alternatives', alternatives, 'joint_centres_5_angles', joint_checks,
          'unintended_limb_fragments', len(fragments), flush=True)


if __name__ == '__main__':
    main()

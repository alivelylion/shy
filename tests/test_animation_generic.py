"""New-image behavior must not depend on the reviewed example filenames."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'common')]

from utils.animation_finish import ARMOUR, ENGINEER, finish
from utils.animation_parts import Layer, composite, export_layers, load_refined, refine_for_animation


def fixture() -> tuple[np.ndarray, list[Layer]]:
    src = np.zeros((64, 64, 4), np.uint8)
    regions = [('face', (8, 23), (24, 40)), ('topwear', (23, 42), (20, 44)),
               ('bottomwear', (42, 58), (23, 41))]
    layers = []
    for i, (name, (y0, y1), (x0, x1)) in enumerate(regions):
        rgba = np.zeros_like(src)
        rgba[y0:y1, x0:x1] = [75+i*40, 120, 160, 255]
        src[rgba[..., 3] > 0] = rgba[rgba[..., 3] > 0]
        layers.append(Layer(name, rgba, np.full(src.shape[:2], 80+i*30, np.uint8), name))
    return src, layers


class GenericCharacterTests(unittest.TestCase):
    def setUp(self) -> None:
        cache = ROOT / 'workspace/.cache/tmp'
        cache.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache, prefix='generic_test_')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.pose = self.directory / 'pose.pt'
        self.pose.write_bytes(b'mocked detector; no model is loaded')

    def source(self, name: str) -> tuple[Path, np.ndarray, list[Layer]]:
        folder = self.directory / name
        folder.mkdir()
        src, layers = fixture()
        Image.fromarray(src).save(folder / 'src_img.png')
        for p in layers:
            Image.fromarray(p.rgba).save(folder / f'{p.name}.png')
            Image.fromarray(p.depth).save(folder / f'{p.name}_depth.png')
        (folder / 'info.json').write_text(json.dumps({'parts': {p.name: {} for p in layers}}), encoding='utf-8')
        return folder, src, layers

    def test_finish_is_invariant_to_names_that_match_old_examples(self) -> None:
        _, originals = fixture()
        expected = None
        for name in ('fresh_character', ENGINEER, ARMOUR, '303401_xingzhiyan_p', '025_new_person'):
            with self.subTest(name=name):
                layers, info = finish(name, copy.deepcopy(originals), {'detected_people': 1}, originals)
                self.assertEqual(info['rule_scope'], 'generic')
                self.assertNotIn('reviewed', info['method'])
                names = [p.name for p in layers]
                self.assertEqual(set(names), {'face', 'clothing'})
                rendered = composite(layers)
                if expected is None:
                    expected = rendered
                np.testing.assert_array_equal(rendered, expected)

    def test_no_confirmed_human_pose_keeps_uncertain_clothing_intact(self) -> None:
        _, originals = fixture()
        layers, info = finish('fresh_character', copy.deepcopy(originals), {'detected_people': 0}, originals)
        self.assertEqual([p.name for p in layers], [p.name for p in originals])
        self.assertTrue(info['warnings'])
        self.assertEqual(info['rounded_limb_pairs'], [])

    def test_refinement_does_not_apply_old_polygons_to_a_renamed_new_image(self) -> None:
        for name in ('fresh_character', ENGINEER, '303401_xingzhiyan_p'):
            with self.subTest(name=name):
                source, src, _ = self.source(name)
                output = self.directory / (name+'_output')
                with patch('utils.animation_parts.detect_poses', return_value=[]):
                    info = refine_for_animation(source, output, self.pose, use_lama=False, previews=False)
                layers, _ = load_refined(output)
                self.assertEqual(info['rule_scope'], 'generic')
                self.assertFalse(info['reviewed_joint_hints'])
                self.assertEqual(info['integrity']['after']['missing_pixels'], 0)
                np.testing.assert_array_equal(composite(layers), src)

    def test_reviewed_cache_is_rebuilt_before_generic_refinement(self) -> None:
        source, src, originals = self.source('new_character')
        output = self.directory / 'output'
        foreign = copy.deepcopy(originals[0])
        foreign.name = 'old_reviewed_cut'
        export_layers([foreign], output, {'rule_scope': 'reviewed_examples', 'refinement_version': 5})
        with patch('utils.animation_parts.detect_poses', return_value=[]) as detector:
            info = refine_for_animation(source, output, self.pose, use_lama=False, previews=False)
        detector.assert_called_once()
        layers, _ = load_refined(output)
        self.assertNotIn('old_reviewed_cut', [p.name for p in layers])
        self.assertEqual(info['rule_scope'], 'generic')
        np.testing.assert_array_equal(composite(layers), src)

    def test_changed_source_rebuilds_pose_instead_of_reusing_old_cuts(self) -> None:
        source, src, _ = self.source('new_character')
        output = self.directory / 'output'
        with patch('utils.animation_parts.detect_poses', return_value=[]):
            refine_for_animation(source, output, self.pose, use_lama=False, previews=False)
        src[25:35, 25:35, :3] = [210, 80, 30]
        Image.fromarray(src).save(source / 'src_img.png')
        with patch('utils.animation_parts.detect_poses', return_value=[]) as detector:
            refine_for_animation(source, output, self.pose, use_lama=False, previews=False)
        detector.assert_called_once()
        layers, _ = load_refined(output)
        np.testing.assert_array_equal(composite(layers), src)

    def test_raw_inference_cache_rejects_different_bytes_under_the_same_filename(self) -> None:
        from scripts.run_ours_full_batch import input_fingerprint, matches_input, store_input_fingerprint
        source, src, _ = self.source('new_character')
        path = source / 'src_img.png'
        before = input_fingerprint(path)
        self.assertFalse(matches_input(source, before))
        store_input_fingerprint(source, before)
        self.assertTrue(matches_input(source, input_fingerprint(path)))
        info = json.loads((source / 'info.json').read_text(encoding='utf-8'))
        self.assertIn('parts', info)
        store_input_fingerprint(source, {})
        self.assertFalse(matches_input(source, before))
        store_input_fingerprint(source, before)
        src[25:35, 25:35, :3] = [210, 80, 30]
        Image.fromarray(src).save(path)
        self.assertFalse(matches_input(source, input_fingerprint(path)))


if __name__ == '__main__':
    unittest.main()

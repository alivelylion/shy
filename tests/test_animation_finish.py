"""Regression checks for fitted cuts and complete backup artwork."""

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'common')]
from utils.animation_parts import Layer, composite, _composite_dense
from utils.animation_finish import round_limb_pairs, add_whole_alternatives, _round_seam_end


class ReferenceFinishTests(unittest.TestCase):
    def test_partition_preserves_both_sides_and_skips_an_unseen_limb(self) -> None:
        from utils.animation_standard import partition_regions
        src=np.full((16,24,4),255,np.uint8)
        target=np.ones((16,24),bool); original=target.copy()
        x=np.indices(target.shape)[1]
        definitions=[{'name':'left','role':'leg_whole','mask':x<12},
                     {'name':'right','role':'leg_whole','mask':x>=12},
                     {'name':'unseen','role':'leg_whole','mask':np.zeros_like(target)}]
        parts=partition_regions(src,target,definitions,False)
        self.assertEqual([p.name for p in parts],['left','right'])
        np.testing.assert_array_equal(target,original)
        np.testing.assert_array_equal(composite(parts),src)

    def test_large_character_sheet_honours_rest_pose_masks_across_tiles(self) -> None:
        layers = []
        for i in range(42):
            rgba = np.zeros((144, 144, 4), np.uint8)
            rgba[112:140, 112:140] = [i*5, 160, 90, 255]
            mask = np.full((144, 144), 255, np.uint8)
            mask[120:135, 120:135] = 0 if i % 2 else 128
            layers.append(Layer(str(i), rgba, np.full((144, 144), i+1, np.uint8),
                                'fixture', visibility_mask=mask))
        np.testing.assert_array_equal(composite(layers), _composite_dense(layers))

    def fixture(self) -> tuple[list[Layer], dict]:
        a = np.zeros((72, 72, 4), np.uint8)
        b = np.zeros_like(a)
        a[12:37, 24:48] = [175, 110, 95, 255]
        b[33:64, 24:48] = [175, 110, 95, 255]
        depth = np.full(a.shape[:2], 100, np.uint8)
        layers = [Layer('upper', a, depth.copy(), 'legwear', pivot=[35, 12], role='leg_thigh'),
                  Layer('lower', b, depth.copy(), 'legwear', pivot=[47, 35], parent='upper', role='leg_shin')]
        info = {'legs': [{'name': 'leg', 'points': [[35, 12], [47, 35], [35, 62]], 'parts': ['upper', 'lower']}]}
        return layers, info

    def test_joint_end_follows_the_cut_width_instead_of_an_offset_pose_point(self) -> None:
        layers, info = self.fixture()
        round_limb_pairs(layers, info)
        centre = info['legs'][0]['points'][1]
        self.assertAlmostEqual(centre[0], 35.5, delta=1)
        self.assertAlmostEqual(centre[1], 34.5, delta=1)
        self.assertEqual(layers[1].pivot, centre)
        self.assertLess(int(np.where(layers[0].mask)[0].min()), 12)
        self.assertGreater(layers[0].rgba[40, 35, 3], 200)
        self.assertGreater(layers[1].rgba[29, 35, 3], 200)
        for p in layers:
            _, _, stats, _ = cv2.connectedComponentsWithStats(p.mask.astype(np.uint8), 8)
            self.assertEqual(sum(stats[1:, 4] >= 15), 1)

    def test_full_arm_backup_retains_material_hidden_by_a_rest_pose_mask(self) -> None:
        layers, info = self.fixture()
        info['arms'] = info.pop('legs')
        layers[0].role, layers[1].role = 'arm_upper', 'arm_forearm'
        layers[0].visibility_mask = np.full((72, 72), 255, np.uint8)
        layers[0].visibility_mask[12:20] = 0
        before = composite(layers)
        add_whole_alternatives(layers, info)
        whole = layers[-1]
        self.assertFalse(whole.meta['default_visible'])
        self.assertEqual(whole.rgba[15, 35, 3], 255)
        np.testing.assert_array_equal(composite(layers), before)

    def test_new_cap_fills_under_the_old_partial_alpha_cut(self) -> None:
        rgba = np.zeros((48, 48, 4), np.uint8)
        rgba[22:42, 12:36] = [200, 140, 110, 255]
        rgba[21, 12:36] = [245, 240, 235, 30]
        p = Layer('piece', rgba, np.full((48, 48), 100, np.uint8), 'test')
        _round_seam_end(p, np.array([23.5, 22]), np.array([1., 0.]), 12, 9)
        self.assertEqual(p.rgba[21, 23, 3], 255)
        np.testing.assert_allclose(p.rgba[21, 23, :3], [200, 140, 110], atol=2)


if __name__ == '__main__':
    unittest.main()

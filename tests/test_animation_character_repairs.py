"""Regression checks for completion and transparent cartoon contour assembly."""
from pathlib import Path
import sys
import unittest
import warnings

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'common')]
from utils.animation_character_repairs import diffuse_material, _over
from utils.animation_detail_repairs import _textured_part


class CartoonRepairTests(unittest.TestCase):
    def test_unobserved_disconnected_island_does_not_make_the_solver_singular(self) -> None:
        source=np.zeros((12,14,4),np.uint8)
        support=np.zeros((12,14),bool);support[1:6,1:6]=True;support[8:11,10:13]=True
        trusted=np.zeros_like(support);trusted[2,2]=True
        source[2,2,:3]=[249,174,80]
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result=diffuse_material(source,support,trusted)
        np.testing.assert_array_equal(result[support],np.tile([249,174,80],(support.sum(),1)))

    def test_transparent_canvas_does_not_darken_an_antialiased_contour(self) -> None:
        bottom=np.zeros((1,1,4),np.uint8)
        top=np.array([[[8,157,226,128]]],np.uint8)
        np.testing.assert_array_equal(_over(bottom,top),top)

    def test_overlapping_antialias_edges_accumulate_alpha(self) -> None:
        bottom=np.array([[[100,100,100,128]]],np.uint8)
        top=np.array([[[200,200,200,128]]],np.uint8)
        result=_over(bottom,top)[0,0]
        np.testing.assert_array_equal(result,[167,167,167,192])

    def test_feathered_fur_edge_keeps_material_instead_of_black_rgb(self) -> None:
        source=np.zeros((32,32,4),np.uint8);source[:]=[168,118,74,255]
        yy,xx=np.indices((32,32));support=(xx-16)**2+(yy-16)**2<100
        part=_textured_part(source,support,support,'fur','animal_body',100,None,
                            [16,16],'fur',False)
        edge=(part.rgba[...,3]>0)&~support
        self.assertTrue(edge.any())
        np.testing.assert_array_equal(part.rgba[edge,:3],np.tile([168,118,74],(edge.sum(),1)))


if __name__=='__main__':
    unittest.main()

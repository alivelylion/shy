"""The compact export must preserve source textures and avoid auxiliary copies."""

from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'common')]
from utils.animation_parts import Layer, composite
from utils.animation_clean import split_existing_limb, split_reviewed_crossing_limbs, split_handheld_sword, export_compact, remove_reviewed_duplicates, restore_named_regions, split_umbrella_from_clothing
from utils.animation_integrity import polygon
from utils.animation_special import preserve_visible_source, split_rear_car
from utils.animation_review import ground_shadow, apply_reviewed_repairs, _complete_round_joint, _clean_engineer_face


class CompactLayerTests(unittest.TestCase):
    def test_face_cleanup_restores_whole_hair_without_recutting_it(self) -> None:
        rgba = np.zeros((32, 32, 4), np.uint8)
        rgba[2:30, 2:30] = [30, 30, 35, 255]
        back_rgba = rgba.copy();back_rgba[:16] = 0
        front_rgba = rgba.copy();front_rgba[16:] = 0
        depth = np.full((32, 32), 30, np.uint8)
        hair = Layer('hair', front_rgba, depth.copy(), 'hair')
        back = Layer('hair_back', back_rgba, depth.copy(), 'hair')
        face_rgba = np.zeros_like(rgba);face_rgba[2:30, 2:30] = [255, 226, 205, 255]
        face_rgba[10:20, 10:20, :3] = [209, 156, 143]
        face = Layer('face', face_rgba, depth.copy(), 'face')
        result = _clean_engineer_face([hair, back, face])
        self.assertNotIn('hair_back', {p.name for p in result})
        np.testing.assert_array_equal(hair.rgba, rgba)
        np.testing.assert_array_equal(face.rgba[14, 14], [255, 226, 205, 255])

    def test_complete_png_survives_reversible_psd_mask_and_alternate_stays_hidden(self) -> None:
        rgba = np.zeros((24, 24, 4), np.uint8);rgba[2:22, 2:22] = [30, 40, 50, 255]
        mask = np.full((24, 24), 255, np.uint8);mask[8:18, 8:18] = 0
        hair = Layer('hair', rgba.copy(), np.full((24, 24), 10, np.uint8), 'hair', visibility_mask=mask)
        face_rgba = np.zeros_like(rgba);face_rgba[8:18, 8:18] = [240, 200, 170, 255]
        face = Layer('face', face_rgba, np.full((24, 24), 20, np.uint8), 'face')
        alternate = Layer('limb_whole', rgba.copy(), np.full((24, 24), 30, np.uint8), 'alternative',
                          meta={'default_visible': False})
        # Different raw PNG data is needed to rule out duplicate-output errors.
        alternate.rgba[2:22, 2:22, :3] = [120, 130, 140]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            export_compact([hair, face, alternate], path, {})
            psd = PSDImage.open(path/'动画拆层.psd')
            stored = {p.name: p for p in psd}
            self.assertTrue(stored['hair'].has_mask())
            self.assertFalse(stored['limb_whole'].visible)
            np.testing.assert_array_equal(np.array(Image.open(path/'hair.png')), rgba)
            rendered = np.array(psd.composite(force=True).convert('RGBA'))
            expected = composite([hair, face, alternate])
            np.testing.assert_allclose(rendered[..., 3].astype(int), expected[..., 3].astype(int), atol=1)
            visible = expected[..., 3] > 0
            np.testing.assert_allclose(rendered[visible].astype(int), expected[visible].astype(int), atol=1)

    def test_round_joint_preserves_known_material_and_keeps_a_covered_rotation_core(self) -> None:
        rgba = np.zeros((72, 72, 4), np.uint8)
        rgba[:, :] = [30, 90, 170, 255]
        known = np.zeros((72, 72), bool)
        known[20:44, 34:60] = True
        rgba[known] = [240, 190, 160, 255]
        result, count = _complete_round_joint(rgba, known, (31, 31), 12, known)
        self.assertGreater(count, 0)
        np.testing.assert_array_equal(result[known], rgba[known])
        np.testing.assert_array_equal(result[31, 31], [240, 190, 160, 255])
        yy, xx = np.indices(known.shape)
        self.assertFalse(result[(np.hypot(xx-31, yy-31) > 12.5) & ~known].any())
        core = np.hypot(xx-31, yy-31) < 8
        for angle in (-60, -30, 0, 30, 60):
            rotated = cv2.warpAffine(result[..., 3], cv2.getRotationMatrix2D((31, 31), angle, 1), (72, 72))
            self.assertTrue((rotated[core] == 255).all())

    def fixture(self) -> Layer:
        rgba=np.zeros((64,64,4),np.uint8)
        yy,xx=np.indices(rgba.shape[:2])
        mask=((xx>=30)&(xx<=42)&(yy>=4)&(yy<=39))|((yy>=29)&(yy<=41)&(xx>=7)&(xx<=42))
        rgba[mask]=[80,160,220,255]
        rgba[mask,0]=xx[mask]*3
        rgba[mask & (yy==29),3]=150
        return Layer('handwear',rgba,np.full(mask.shape,100,np.uint8),'handwear')

    def test_bent_limb_preserves_full_rgba_and_has_no_external_caps(self) -> None:
        limb=self.fixture()
        chains=[{'name':'arm','points':[[36,4],[36,35],[12,35]],'provenance':'reviewed_image_coordinates'}]
        pieces,events=split_existing_limb(limb,chains,'arm')
        self.assertEqual(len(events),1)
        self.assertEqual({p.role for p in pieces},{'arm_upper','arm_forearm'})
        np.testing.assert_array_equal(composite(pieces),composite([limb]))
        for piece in pieces:
            self.assertFalse((piece.mask & ~limb.mask).any())
            np.testing.assert_array_equal(piece.rgba[piece.mask],limb.rgba[piece.mask])
        self.assertTrue(pieces[0].mask[8,36])
        self.assertFalse(pieces[1].mask[8,36])
        self.assertTrue(pieces[1].mask[35,10])
        self.assertFalse(pieces[0].mask[35,10])

    def test_disconnected_trouser_legs_do_not_swap_a_flared_edge(self) -> None:
        rgba=np.zeros((68,104,4),np.uint8)
        left=polygon(rgba.shape[:2],[[13,3],[32,3],[32,23],[57,20],[57,28],[32,29],
                                    [32,40],[12,61],[2,51],[18,32]])
        right=polygon(rgba.shape[:2],[[69,3],[83,3],[83,28],[98,53],[87,62],[70,39]])
        rgba[left]=[80,160,220,255];rgba[right]=[180,70,40,255]
        layer=Layer('legwear',rgba,np.full(rgba.shape[:2],100,np.uint8),'legwear')
        chains=[{'name':'left','points':[[25,5],[23,35],[8,55]]},
                {'name':'right','points':[[75,5],[77,35],[92,55]]}]
        parts,_=split_existing_limb(layer,chains,'leg')
        self.assertEqual(len(parts),4)
        for p in parts:
            expected=left if p.name.startswith('left') else right
            self.assertFalse((p.mask & ~expected).any())
        np.testing.assert_array_equal(composite(parts),rgba)

    def test_crossed_legs_remain_connected_with_completed_hidden_back_thigh(self) -> None:
        shape=(128,128)
        front_outline=[[90,8],[103,22],[48,72],[65,116],[49,124],[27,73]]
        back_outline=[[30,12],[46,10],[74,65],[114,113],[100,124],[57,78]]
        front=polygon(shape,front_outline);back=polygon(shape,back_outline)
        rgba=np.zeros((*shape,4),np.uint8)
        rgba[back]=[70,50,40,255];rgba[front]=[180,130,95,255]
        layer=Layer('legwear',rgba,np.full(shape,100,np.uint8),'legwear')
        chains=[{'name':'near','points':[[97,18],[36,70],[56,116]]},
                {'name':'far','points':[[38,18],[64,70],[106,116]]}]
        parts,_=split_reviewed_crossing_limbs(layer,chains,'leg',front_outline,back_outline,False)
        self.assertEqual(len(parts),4)
        for p in parts:
            count,*_=cv2.connectedComponentsWithStats(p.mask.astype(np.uint8),8)
            self.assertEqual(count,2,p.name)
        np.testing.assert_array_equal(composite(parts),rgba)
        far=composite([p for p in parts if p.name.startswith('far')])
        self.assertTrue((far[front&back,3]==255).all())
        self.assertTrue((far[front&back,:3]==[70,50,40]).all())

    def test_sword_moves_without_taking_fingers_or_leaving_a_duplicate(self) -> None:
        rgba=np.zeros((40,40,4),np.uint8)
        rgba[1:8,20:30]=[20,70,80,255]
        rgba[8:20,20:30]=[200,130,90,255]
        rgba[20:36,22:26]=[45,48,56,255]
        hand=Layer('handwear',rgba.copy(),np.full((40,40),100,np.uint8),'handwear')
        parts=split_handheld_sword([hand],[[20,8],[30,8],[30,36],[20,36]],
                                   [[22,10],[25,10],[25,25],[22,25]],[24,18])
        sword=next(p for p in parts if p.name=='sword')
        self.assertEqual(hand.rgba[25,24,3],0)
        np.testing.assert_array_equal(hand.rgba[12,24],rgba[12,24])
        np.testing.assert_array_equal(sword.rgba[12,24],[45,48,56,255])
        np.testing.assert_array_equal(composite(parts),rgba)

    def test_ground_shadow_has_a_complete_soft_centre_without_sprite_fragments(self) -> None:
        p=ground_shadow((64,96),(48,44),(32,9),'ground')
        self.assertGreater(p.rgba[44,48,3],p.rgba[44,70,3])
        self.assertGreater(p.rgba[44,48,3],0)
        self.assertEqual(p.rgba[10,48,3],0)
        self.assertTrue((p.rgba[p.mask,:3]==[35,30,32]).all())
        count,*_=cv2.connectedComponentsWithStats(p.mask.astype(np.uint8),8)
        self.assertEqual(count,2)
        cache=ROOT/'workspace/.cache/tmp';cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache,prefix='soft_shadow_') as temp:
            output=Path(temp);export_compact([p],output,{})
            rendered=np.array(PSDImage.open(output/'动画拆层.psd').composite(force=True).convert('RGBA'))
            np.testing.assert_allclose(rendered[...,3],p.rgba[...,3],atol=1)

    def test_animal_sheet_preserves_eight_whole_sprites_without_wear_copies(self) -> None:
        rgba=np.zeros((1024,1024,4),np.uint8)
        for x,y in [(220,230),(520,230),(790,230),(220,520),(790,520),
                    (220,800),(510,800),(790,800)]:
            cv2.circle(rgba,(x,y),45,(x%200+30,y%200+30,90,255),-1)
        parts=apply_reviewed_repairs('45-meowa-watermark',rgba,[],False)
        self.assertEqual(len(parts),8)
        self.assertEqual({p.role for p in parts},{'animal_whole'})
        self.assertFalse(any('wear' in p.name for p in parts))
        np.testing.assert_array_equal(composite(parts),rgba)

    def test_sprite_tail_is_removed_from_hair_without_removing_a_hairstyle(self) -> None:
        rgba=np.zeros((64,64,4),np.uint8)
        rgba[5:20,5:20]=[50,40,30,255]
        rgba[35:50,40:45]=[30,45,70,255]
        hair=Layer('hair',rgba.copy(),np.full((64,64),100,np.uint8),'hair')
        tail=Layer('tail',rgba.copy(),np.full((64,64),90,np.uint8),'tail')
        tail.rgba[:30]=0
        parts=apply_reviewed_repairs('sc_gen_7245-meowa-watermark',rgba,[hair,tail],False)
        self.assertTrue(hair.mask[10,10])
        self.assertFalse(hair.mask[40,42])
        self.assertTrue(tail.mask[40,42])
        np.testing.assert_array_equal(composite(parts),rgba)

    def test_compact_delivery_has_one_png_per_part_and_no_masks(self) -> None:
        limb=self.fixture()
        pieces,_=split_existing_limb(limb,[{'name':'arm','points':[[36,4],[36,35],[12,35]]}],'arm')
        cache=ROOT/'workspace/.cache/tmp'
        cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache,prefix='clean_layers_') as temp:
            output=Path(temp)
            export_compact(pieces,output,{})
            self.assertEqual({p.name for p in output.glob('*.png')},
                             {'arm_upper.png','arm_forearm.png','静态合成.png','部件总览.png'})
            self.assertTrue((output/'动画拆层.psd').is_file())
            rendered=np.array(PSDImage.open(output/'动画拆层.psd').composite(force=True).convert('RGBA'))
            reference=composite(pieces)
            np.testing.assert_allclose(rendered[...,3],reference[...,3],atol=1)
            covered=reference[...,3]>0
            np.testing.assert_allclose(rendered[covered,:3],reference[covered,:3],atol=1)
            whole,_=split_existing_limb(limb,[{'name':'arm','points':[[36,4],[36,35],[12,35]],
                                              'subdivide':False}],'arm')
            export_compact(whole,output,{})
            self.assertEqual({p.name for p in output.glob('*.png')},
                             {'handwear.png','静态合成.png','部件总览.png'})

    def test_original_duplicate_prediction_is_merged_without_losing_outline(self) -> None:
        original=self.fixture()
        original.name=original.source='topwear'
        other=Layer('legwear',original.rgba.copy(),original.depth.copy(),'legwear')
        other.rgba[45,35]=[80,160,220,255]
        parts,records=remove_reviewed_duplicates('fixture',[original,other])
        self.assertEqual(len(parts),1)
        self.assertTrue(parts[0].mask[45,35])
        self.assertEqual(records[0]['merged_near_duplicate'],'legwear')

    def test_reviewed_unclear_limb_stays_whole(self) -> None:
        limb=self.fixture()
        pieces,events=split_existing_limb(limb,[{'name':'cartoon_arm','points':[[36,4],[36,35],[12,35]],
                                               'subdivide':False}],'arm')
        self.assertEqual(len(pieces),1)
        self.assertEqual(pieces[0].role,'arm_whole')
        self.assertEqual(events,[])
        np.testing.assert_array_equal(pieces[0].rgba,limb.rgba)

    def test_skeleton_estimate_alone_does_not_force_an_elbow(self) -> None:
        limb=self.fixture()
        pieces,events=split_existing_limb(limb,[{'name':'sleeve','points':[[36,4],[36,35],[12,35]],
                                               'provenance':'completed_mask_skeleton_estimate'}],'arm')
        self.assertEqual([p.role for p in pieces],['arm_whole'])
        self.assertEqual(events[0]['mode'],'whole')
        np.testing.assert_array_equal(composite(pieces),composite([limb]))

    def test_occluded_knee_outside_boot_does_not_create_a_thigh(self) -> None:
        limb=self.fixture()
        pieces,events=split_existing_limb(limb,[{'name':'boot','points':[[10,2],[12,15],[12,35]]}],'leg')
        self.assertEqual([p.role for p in pieces],['leg_whole'])
        self.assertEqual(events[0]['reason'],'joint_outside_existing_limb')

    def test_missing_body_pixels_are_assigned_to_named_body_without_residual_image(self) -> None:
        original=self.fixture()
        partial=original.rgba.copy()
        partial[25:]=0
        partial[partial[...,3]>0,:3]=[230,40,20]
        hints={'source_instances':[{'name':'sprite','replace_all':True,'regions':[
            {'name':'sprite_body','role':'animal_body','polygon':[[0,0],[63,0],[63,63],[0,63]]}]}]}
        cache=ROOT/'workspace/.cache/tmp'
        cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache,prefix='clean_body_') as temp:
            previous=Path(temp)
            Image.fromarray(partial).save(previous/'sprite_body.png')
            parts,_=restore_named_regions(original.rgba,[original],[original],hints,previous)
            self.assertEqual([p.name for p in parts],['sprite_body'])
            np.testing.assert_array_equal(composite(parts),composite([original]))

    def test_umbrella_is_removed_from_coat_and_shaft_has_one_owner(self) -> None:
        rgba=np.zeros((64,64,4),np.uint8)
        rgba[4:42,4:50]=[20,50,100,255]
        rgba[24:55,30:55]=[180,80,40,255]
        rgba[24:55,40:44]=[5,5,5,255]
        top=Layer('topwear',rgba,np.full((64,64),100,np.uint8),'topwear')
        expected=composite([top])
        parts=split_umbrella_from_clothing([top],[[30,24],[54,24],[54,54],[30,54]],
                                           [[4,4],[49,4],[49,41],[4,41]],
                                           [[40,24],[43,24],[43,54],[40,54]],use_lama=False)
        by_name={p.name:p for p in parts}
        self.assertFalse(by_name['topwear'].mask[10,10])
        self.assertTrue(by_name['umbrella'].mask[10,10])
        np.testing.assert_array_equal(by_name['topwear'].rgba[40,41],[180,80,40,255])
        np.testing.assert_array_equal(by_name['umbrella_handle'].rgba[40,41],[5,5,5,255])
        np.testing.assert_array_equal(composite(parts),expected)

    def test_source_edges_have_one_owner_without_destroying_hidden_parts(self) -> None:
        src=np.zeros((24,24,4),np.uint8)
        src[4:20,4:20]=[210,40,30,255]
        src[4,4:20]=[220,50,35,150]
        hidden=src.copy();hidden[4:20,4:20]=[20,30,55,255]
        front=Layer('front',src.copy(),np.full((24,24),50,np.uint8),'fixture')
        back=Layer('back',hidden,np.full((24,24),150,np.uint8),'fixture')
        parts=preserve_visible_source(src,[front,back])
        np.testing.assert_array_equal(composite(parts),src)
        np.testing.assert_array_equal(back.rgba[12,12],[20,30,55,255])
        self.assertEqual(back.rgba[4,12,3],0)
        self.assertEqual([p.name for p in parts],['front','back'])

    def test_hidden_tyres_use_rubber_even_with_red_bumper_and_black_watermark(self) -> None:
        src=np.zeros((1024,1024,4),np.uint8)
        src[450:805,50:975]=[190,15,20,255]
        src[450:640,50:975]=[230,240,240,255]
        src[383:459,5:121]=[190,15,20,255]
        src[383:459,903:1019]=[190,15,20,255]
        src[805:851,54:213]=[25,28,50,255]
        src[805:851,812:975]=[25,28,50,255]
        src[800:835,850:858]=[0,0,0,255]
        for x in [149,873]:
            cv2.circle(src,(x,554),58,(20,5,10,255),-1)
            cv2.circle(src,(x,554),50,(230,12,15,255),-1)
        parts=split_rear_car(src,use_lama=False)
        by_name={p.name:p for p in parts}
        self.assertEqual(len(parts),7)
        np.testing.assert_array_equal(composite(parts),src)
        for name,x in [('tyre_left',132),('tyre_right',893)]:
            rubber=by_name[name].rgba[750,x]
            self.assertEqual(rubber[3],255)
            self.assertLess(rubber[0],90)
            self.assertGreater(rubber[2],rubber[0])
        self.assertGreater(by_name['car_body'].rgba[554,149,1],200)
        np.testing.assert_array_equal(by_name['tyre_right'].rgba[820,853],src[820,853])


if __name__=='__main__':
    unittest.main()

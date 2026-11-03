import os
import random
import os.path as osp
import numpy as np
from pathlib import Path
import shutil
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from PIL import Image
from  tqdm import tqdm
import click
import cv2

from utils.cv import mask2rle, rle2mask, mask_xyxy
from utils.io_utils import load_exec_list, find_all_files_recursive, find_all_files_with_name, pil_ensure_rgb, imglist2imgrid, imread, imwrite, json2dict, save_tmp_img, dict2json
from utils.sampler import NameSampler
from utils.visualize import visualize_segs, visualize_segs_with_labels, visualize_pos_keypoints
from live2d.scrap_model import Live2DScrapModel, VALID_BODY_PARTS_V1, VALID_BODY_PARTS_V2, compose_mask_from_drawables, init_drawable_visible_map, load_detected_character, load_pos_estimation


@click.group()
def cli():
    """live2d data processing related scripts.
    """


def get_unique_render_lst(exec_list):
    unique_lst = []
    processed_models = set()
    unique_src_to_models = dict()
    for p in tqdm(exec_list):
        modeld = osp.dirname(p)
        if modeld not in processed_models:
            processed_models.add(modeld)
        else:
            continue
        plist = sub_render_parts([p])
        mlist = [Live2DScrapModel(p, skip_load=True) for p in plist]
        for m in mlist:
            m.init_drawables()
        unique_mlist = [mlist[4]]
        for m in mlist:
            is_unique = True
            mklist = list(m.did2drawable.keys())
            mklist.sort()
            for um in unique_mlist:
                umklist = list(um.did2drawable.keys())
                umklist.sort()
                if mklist == umklist:
                    srcp = um.directory
                    is_unique = False
                    break

            tgtp = m.directory
            if is_unique:
                unique_mlist.append(m)
                srcp = m.directory
            if srcp not in unique_src_to_models:
                unique_src_to_models[srcp] = []
            unique_src_to_models[srcp].append(tgtp)
                
        unique_mlist = [m.directory for m in unique_mlist]
        unique_lst += unique_mlist

    return unique_lst, unique_src_to_models


@cli.command('get_tgt_list')
@click.option('--src_dir')
@click.option('--savep', default=None)
def get_tgt_list(src_dir, savep):
    if savep is None:
        savep = osp.join('workspace/datasets', osp.basename(src_dir) + '.txt')

    valid_list = []
    for f in find_all_files_recursive(src_dir, ext={'.json'}):
        tgtf = f.rstrip('.json') + '.png'
        if osp.exists(tgtf):
            valid_list.append(tgtf)
    print(len(valid_list))

    with open(savep, 'w', encoding='utf8') as f:
        f.write('\n'.join(valid_list))


@cli.command('get_png_list')
@click.option('--src_dir')
@click.option('--savep', default=None)
def get_png_list(src_dir, savep):
    if savep is None:
        savep = osp.join('workspace/datasets', osp.basename(src_dir) + '.txt')

    valid_list = []
    for f in find_all_files_recursive(src_dir, ext={'.png'}):
        valid_list.append(f)
    print(len(valid_list))

    with open(savep, 'w', encoding='utf8') as f:
        f.write('\n'.join(valid_list))


@cli.command('check_unique_rst')
@click.option('--exec_list')
@click.option('--savep', default=None)
def check_unique_rst(exec_list, savep):
    if savep is None:
        savep = exec_list
    exec_list = load_exec_list(exec_list)
    exec_list, unique_src_to_models = get_unique_render_lst(exec_list)
    print(len(exec_list))

    with open(savep, 'w', encoding='utf8') as f:
        f.write('\n'.join(exec_list))
    dict2json(unique_src_to_models, savep + '.json')


@cli.command('compress_live2d')
@click.option('--src_dir')
@click.option('--save_dir')
@click.option('--ext', default='.jxl')
@click.option('--disable_crop', is_flag=True, default=False)
def compress_live2d(src_dir, save_dir, ext, disable_crop):

    src_dir = osp.normpath(src_dir)
    model_final_list = find_all_files_with_name(src_dir, 'final')

    crop = not disable_crop
    if save_dir is None:
        save_dir = src_dir +  f'_{ext}'
        if crop:
            save_dir += '_crop'
    save_dir = osp.normpath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    ndir_leading = len(src_dir.split(os.path.sep))
    for model_f in tqdm(model_final_list, desc=f'saving to {save_dir}'):
        model_dir = osp.dirname(model_f)
        model_save_dir = model_dir.split(os.path.sep)[ndir_leading:]
        model = Live2DScrapModel(model_dir, crop_to_final=crop, pad_to_square=False)
        model.save_model_to(osp.join(save_dir, *model_save_dir), 
                            crop_to_final=crop, img_ext=ext)


@cli.command('build_live2d_exec_list')
@click.option('--srcd')
@click.option('--save_dir', default=None)
@click.option('--filter_p', default=None)
@click.option('--target_fno', default=-1)
@click.option('--num_chunk', default=-1)
@click.option('--save_name', default='exec_list')
def build_live2d_exec_list(srcd, save_dir, filter_p, target_fno, num_chunk, save_name):

    exec_list = find_all_files_with_name(srcd, name='final', exclude_suffix=True)

    tgt_list = []
    filter_set = set()
    if filter_p is not None:
        filter_set = set(load_exec_list(filter_p))
    for d in exec_list:
        if d in filter_set or osp.dirname(d) in filter_set:
            continue
        dname = osp.basename(osp.dirname(d))
        if target_fno > 0:
            fno = dname.split('-')[-1]
            if not fno.isdigit():
                print(f'{d} is not a valid path')
                continue
            fno = int(fno)
            if fno == target_fno:
                tgt_list.append(d)
        else:
            tgt_list.append(d)

    random.shuffle(tgt_list)
    print(f'num samples: {len(tgt_list)}')

    if save_dir is None:
        save_dir = srcd

    with open(osp.join(save_dir, f'{save_name}.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(tgt_list))

    if num_chunk > 0:
        world_size = num_chunk
        for ii in range(world_size):
            t = load_exec_list(tgt_list, ii, world_size=world_size)
            with open(osp.join(save_dir, f'{save_name}{ii}.txt'), 'w', encoding='utf8') as f:
                f.write('\n'.join(t))
            print(f'chunk {ii} num samples: {len(t)}')


@cli.command('build_exec_list')
@click.option('--srcd')
@click.option('--exts', default=None)
@click.option('--save_dir', default=None)
@click.option('--num_chunk', default=-1)
@click.option('--save_name', default='exec_list')
def build_exec_list(srcd, exts, save_dir, num_chunk, save_name):
    exec_list = []
    exts = exts.split(',')
    for s in srcd.split(','):
        exec_list += find_all_files_recursive(s, ext=exts)

    print(f'found {len(exec_list)} samples for {exts}')
    tgt_list = exec_list
    random.shuffle(tgt_list)

    if save_dir is None:
        save_dir = srcd

    with open(osp.join(save_dir, f'{save_name}.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(tgt_list))

    if num_chunk > 0:
        world_size = num_chunk
        for ii in range(world_size):
            t = load_exec_list(tgt_list, ii, world_size=world_size)
            with open(osp.join(save_dir, f'{save_name}{ii}.txt'), 'w', encoding='utf8') as f:
                f.write('\n'.join(t))
            print(f'chunk {ii} num samples: {len(t)}')


def sub_render_parts(exec_list):
    new_exec_list = []
    for d in exec_list:
        d = '-'.join(d.split('-')[:-1])
        for ii in range(9):
            md = d + f'-{ii}'
            if osp.exists(md):
                new_exec_list.append(md)
    return new_exec_list


def assign_masks_to_points(points, mask_list, distance_thr=1.0):

    def _coord_stats(cx):
        x1, x2 = np.min(cx), np.max(cx)
        cent_x = np.round(np.mean(cx))
        return x1, x2, cent_x

    mask_asignment = {}
    n_points = len(points)
    n_mask = len(mask_list)

    if n_points < 1 or n_mask < 1:
        return mask_asignment
    
    for ii in range(n_points):
        mask_asignment[ii] = {'mask': None, 'distance': [], 'mask_idx': None}  

    h, w = mask_list[0].shape[:2]
    dist_max = h * w
    for m in mask_list:
        coords = np.where(m)

        if len(coords[0]) == 0:
            for pi in range(n_points):
                mask_asignment[pi]['distance'].append(dist_max)
            continue

        x1, x2, cx = _coord_stats(coords[1])
        y1, y2, cy = _coord_stats(coords[0])
        diag = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        for pi, pnt in enumerate(points):
            dist = np.linalg.norm(pnt - np.array([cx, cy]))
            if dist > distance_thr * diag:
                dist = dist_max
            mask_asignment[pi]['distance'].append(dist)

    for pi in range(n_points):
        mask_asignment[pi]['distance'] = np.array(mask_asignment[pi]['distance'])
    not_assigned = list(range(n_points))
    while len(not_assigned) > 0:
        pi_dist_pair = [(pi, np.min(mask_asignment[pi]['distance']), np.argmin(mask_asignment[pi]['distance'])) for pi in not_assigned]
        pi_dist_pair.sort(key=lambda x: x[1])
        m_pi, m_dist, m_mid = pi_dist_pair.pop(0)
        if m_dist < dist_max:
            mask_asignment[m_pi]['mask'] = mask_list[m_mid]
            mask_asignment[m_pi]['mask_idx'] = m_mid
        else:
            break
        not_assigned = [p[0] for p in pi_dist_pair if p[1] < dist_max]
        for pi in not_assigned:
            mask_asignment[pi]['distance'][m_mid] = dist_max
        pass
    return mask_asignment


def try_assign_sam_mask(lmodel: Live2DScrapModel, mask_list, body_part_tag, check_points=None, 
                        mask_ioa_thr=0.4, valid_face_ids=None, skip_existed_bodypart=True,
                        exclude_mask=None, bbox_constraint=None, y2_max=None):

    if exclude_mask is not None:
        mask_list = [m for m in mask_list if not np.any(m & exclude_mask)]

    if bbox_constraint is not None or y2_max is not None:
        _mask_list = []
        for m in mask_list:
            coords = np.where(m)
            mx1, mx2 = np.min(coords[1]), np.max(coords[1])
            my1, my2 = np.min(coords[0]), np.max(coords[0])
            if bbox_constraint is not None:
                if my1 < bbox_constraint[1] or my2 > bbox_constraint[3] or mx1 < bbox_constraint[0] or mx2 > bbox_constraint[2]:
                    continue
            if y2_max is not None and my2 > y2_max:
                continue
            _mask_list.append(m)
        mask_list = _mask_list

    if check_points is not None:
        msk_assignment = assign_masks_to_points(check_points, mask_list)
        msk = [m['mask'] for m in msk_assignment.values() if m['mask'] is not None]
    else:
        msk = mask_list
    if len(msk) < 1:
        return None, None
    if len(msk) > 1:
        msk = np.logical_or.reduce(np.stack(msk), axis=0)
    else:
        msk = msk[0]
    msk_assigned = False

    if valid_face_ids is None:
        valid_face_ids = set()

    msk_area = msk.sum() + 1e-6
    ioa_lst = {}
    for ii, d in enumerate(lmodel.drawables):
        if d.final_visible_area < 1:
            continue
        if d.face_part_id is not None and d.face_part_id not in valid_face_ids:
            continue
        if skip_existed_bodypart and d.body_part_tag is not None:
            continue
        x1, y1, x2, y2 = d.xyxy
        mask_sum = np.sum(msk[y1: y2, x1: x2] & d.final_visible_mask)
        ioa = mask_sum / d.final_visible_area
        ios = mask_sum / msk_area
        ioa_lst[ii] = {'ioa': ioa, 'ios': ios}
        if ioa > mask_ioa_thr or ios > 1.:
            d.body_part_tag = body_part_tag
            msk_assigned = True

    # if not msk_assigned:
    #     max_ioa = -1
    #     assigned_drawable = None
    #     for ii, d in enumerate(lmodel.drawables):
    #         if d.final_visible_area < 1:
    #             continue
    #         if d.face_part_id is not None and d.face_part_id not in valid_face_ids:
    #             continue
    #         if skip_existed_bodypart and d.body_part_tag is not None:
    #             continue
    #         ioa, ios = ioa_lst[ii]['ioa'], ioa_lst[ii]['ios']
    #         if ios > 0.5:
    #             assigned_drawable = d
    #             max_ioa = ioa
    #     if assigned_drawable is not None:

    return msk, msk_assigned


def mask_cover_pos(mask, pos_list, pos_ids, mode='any', xshift=0, yshift=0):
    if isinstance(pos_ids, (int, np.ScalarType)):
        pos_ids = [pos_ids]

    h, w = mask.shape[:2]
    # if mode == 'any':
    for pi in pos_ids:
        pi = pos_list[pi]
        py, px  = pi[1] + yshift, pi[0] + xshift
        covered = py < h and py >= 0 and px < w and px >= 0
        if covered and mask[py, px] > 0:
            if mode == 'any':
                return True
        else:
            return False

    if mode == 'any':
        return True
    else:
        return False


def mask_line_sample(mask, line_start, line_end, divide_long_side=False):
    x1, y1 = line_start
    x2, y2 = line_end
    lh, lw = abs(y2 - y1), abs(x2 - x1)
    long_side = max(lh, lw)
    h, w = mask.shape[:2]
    if long_side < 1:
        return 0
    x_lst = np.round(np.linspace(x1, x2, long_side)).astype(np.int32)
    y_lst = np.round(np.linspace(y1, y2, long_side)).astype(np.int32)
    valid_pnts = (x_lst < w) & (y_lst < h)
    if valid_pnts.sum() == 0:
        return 0
    score = mask[(y_lst[valid_pnts], x_lst[valid_pnts])].sum()
    if divide_long_side:
        score /= long_side
    return score
    # pass


def assign_mask_to_armature(mask, pos_list, score_list, score_thr=0.12, selected_armatures=None):
    _valid_armatures = {
        "handwear_0": (10, 8),
        "handwear_1": (8, 6),
        "topwear": (6, 5),
        "handwear_2": (5, 7),
        "handwear_3": (7, 9),

        "legwear_1": (16, 14),
        "legwear_2": (14, 12),
        "bottomwear": (12, 11),
        "legwear_3": (11, 13),
        "legwear_4": (13, 15)
    }
    valid_armatures = {}
    for k, vs in _valid_armatures.items():
        is_valid = True
        for v in vs:
            if score_list[v] < score_thr:
                is_valid = False
                break
        if is_valid:
            valid_armatures[k] = vs
    # armature_scores = {}
    max_mask_score = 0
    armature_assigned = None
    assign_scores = {}

    for k, vs in valid_armatures.items():
        mask_score = mask_line_sample(mask, pos_list[vs[0]], pos_list[vs[1]], divide_long_side=True)
        if mask_score > max_mask_score:
            max_mask_score = mask_score
            armature_assigned = k
        assign_scores[k] = mask_score
    
    armature_ids = None
    if armature_assigned is not None:
        armature_ids = valid_armatures[armature_assigned]
        armature_assigned = armature_assigned.split('_')[0]

    if selected_armatures is not None and armature_assigned not in selected_armatures:
        return None, None, None

    return armature_assigned, armature_ids, assign_scores


def armature_cc(mask_input, pos, armature_pairs, min_num_cc=-1):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_input.astype(np.uint8), connectivity=8)
    mask = None
    if retval > min_num_cc:
        # leg_pairs = [(16, 14), (14, 12), (11, 13), (13, 15)]
        mask = np.zeros_like(labels, dtype=bool)
        for l in range(1, retval):
            mask_assigned = False
            lmsk = l == labels
            for p in armature_pairs:
                if mask_line_sample(lmsk, pos[p[0]], pos[p[1]]) > 0:
                    mask_assigned = True
                    break
            if mask_assigned:
                mask = mask | lmsk
    return mask


def taglist_has_keywords(tag_list, keywords):
    if isinstance(keywords, str):
        keywords = [keywords]
    for tag in tag_list:
        for k in keywords:
            if k in tag:
                return True
    return False

TARGET_FRAME_SIZE = 1024

@cli.command('parse_render_body_samples')
@click.option('--exec_list')
@click.option('--bg_list')
@click.option('--save_dir')
@click.option('--rank_to_worldsize', default='', type=str)
def parse_render_body_samples(exec_list, bg_list, save_dir, rank_to_worldsize):

    from live2d.scrap_model import animal_ear_detected, Drawable
    from utils.cv import fgbg_hist_matching, quantize_image, random_crop, rle2mask, mask2rle, img_alpha_blending, resize_short_side_to, batch_save_masks, batch_load_masks
    from utils.torch_utils import seed_everything

    def _compose_body_samples(lmodel: Live2DScrapModel):
        '''
        some augmentation can be done here
        '''

        part_mask_list = []

        body_final = lmodel.compose_bodypart_drawables(VALID_BODY_PARTS_V1)

        for tag in VALID_BODY_PARTS_V1:
            m  = lmodel.compose_bodypart_drawables(tag, mask_only=True, final_visible_mask=True).astype(np.uint8)
            # save_tmp_img(m, mask2img=True)
            part_mask_list.append(m)

        return True, part_mask_list, body_final
    
    os.makedirs(save_dir, exist_ok=True)

    seed_everything(42)

    hist_match_prob = 0.2
    quantize_prob = 0.25
    color_correction_sampler = NameSampler({'hist_match': hist_match_prob, 'quantize': quantize_prob})
    
    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)
    bg_list = load_exec_list(bg_list)

    tagcluster_bodypart = json2dict('workspace/datasets/tagcluster_bodypart.json')
    tag2generaltag = {}
    for general_tag, tlist in tagcluster_bodypart.items():
        for t in tlist:
            if t in tag2generaltag and tag2generaltag[t] != general_tag:
                print(f'conflict tag def: {t} - {general_tag}, ' + tag2generaltag[t])
            tag2generaltag[t] = general_tag


    for ii, p in enumerate(tqdm(exec_list[0:])):
        try:

            instance_mask, crop_xyxy, score = load_detected_character(p)
            if instance_mask is None:
                print(f'skip {p}, no character instance detected')
                continue
            
            lmodel = Live2DScrapModel(p, crop_xyxy=crop_xyxy, pad_to_square=False)
            model_dir = lmodel.directory

            if len(lmodel.facedet) == 0:
                print(f'skip {model_dir}, no face detected')

            pos_estim = load_pos_estimation(model_dir)
            pos = pos_estim['pos']
            pos_scores = pos_estim['scores']

            if pos is None:
                print(f'skip {model_dir}, no pos detected')
                continue
            
            pos = np.round(pos[:, ::-1]).astype(np.int32)
            feet_valid = pos_scores[15] > 0.12 or pos_scores[16] > 0.12

            tag_loaded = lmodel.load_tag_stats()
            if not tag_loaded:
                print(f'skip {model_dir}, no valid tag stats')
                continue
            tags = json2dict(osp.join(model_dir, 'general_tags.json'))
            general_tags = set([tag2generaltag[k] for k in tags if k in tag2generaltag])
            has_animal_ear = animal_ear_detected(tags)

            face_parsing_loaded = lmodel.load_face_parsing()
            
            if not face_parsing_loaded:
                print(f'skip {face_parsing_loaded}, no valid faceparsing result')
                continue

            fx1, fy1, fx2, fy2 = lmodel.facedet[0]['bbox'][:4]
            lmodel.init_drawable_visible_map()

            frame_h, frame_w = lmodel.final.shape[:2]
            
            wrist_left, wrist_right = pos[9], pos[10]

            lang_sam_masks = None
            lang_samp = osp.join(model_dir, 'langsam_masks.json')
            if osp.exists(lang_samp):
                lang_sam_masks = json2dict(lang_samp)['instances']

            top_wear_assigned = hand_msk_assigned = mouth_msk_assigned = nose_msk_assigned = neck_msk_assigned = False
            if lang_sam_masks is not None:
                if has_animal_ear:
                    ear_msk = [m for m in lang_sam_masks['ears']['masks']]
                    if len(ear_msk) > 0:
                        _, ear_msk_assigned = try_assign_sam_mask(
                            lmodel,
                            [rle2mask(m) for m in ear_msk], body_part_tag='ears',
                            exclude_mask=lmodel.compose_face_drawables(face_part_ids=[7, 8], mask_only=True, final_visible_mask=True), mask_ioa_thr=0.3,
                            valid_face_ids={17, 18}, y2_max=fy2
                        )

                _, hand_msk_assigned = try_assign_sam_mask(
                    lmodel,
                    [rle2mask(m) for m in lang_sam_masks['hand']['masks']],
                    check_points=[wrist_left, wrist_right], body_part_tag='handwear', mask_ioa_thr=0.1, valid_face_ids={16, 17}
                )

                ankle_kpts = [pos[k] for k in [15, 16] if pos_scores[k] > 0.12]
                if len(ankle_kpts) > 0:
                    _, feet_msk_assigned = try_assign_sam_mask(
                        lmodel,
                        [rle2mask(m) for m in lang_sam_masks['feet']['masks']],
                        check_points=ankle_kpts, body_part_tag='footwear'
                    )
                # _, shoes_msk_assigned = try_assign_sam_mask(
                #     lmodel,
                #     [rle2mask(m) for m in lang_sam_masks['shoes']['masks']],
                #     check_points=[ankle_left, ankle_right], body_part_tag='footwear'
                # )
                # save_tmp_img(visualize_segs([rle2mask(m) for m in lang_sam_masks['shoes']['masks']], lmodel.final[..., :3]))
                # save_tmp_img(visualize_segs([rle2mask(m) for m in lang_sam_masks['feet']['masks']], lmodel.final[..., :3]))
                # feet_msk_assigned = feet_msk_assigned or shoes_msk_assigned

                top_wear_mask = None
                top_wear_masks = lang_sam_masks['jacket']['masks'] + lang_sam_masks['dress']['masks']
                top_wear_scores = lang_sam_masks['jacket']['scores'] + lang_sam_masks['dress']['scores']
                if len(top_wear_scores) > 0:
                    midx = np.argmax(np.array(top_wear_scores))
                    top_wear_mask = rle2mask(top_wear_masks[midx])
                if top_wear_mask is not None:
                    top_wear_masks = [top_wear_mask]
                else:
                    top_wear_masks = []
                # top_wear_masks += [rle2mask(m) for m in lang_sam_masks['shirt']['masks']]

                if top_wear_masks is not None:
                    _, top_wear_assigned = try_assign_sam_mask(
                        lmodel,
                        top_wear_masks,
                        body_part_tag='topwear',
                        valid_face_ids={16}
                    )

                _, leg_msk_assigned = try_assign_sam_mask(
                    lmodel,
                    [rle2mask(m) for m in lang_sam_masks['leg']['masks']],
                    body_part_tag='legwear'
                )
                # save_tmp_img(visualize_segs([rle2mask(m) for m in lang_sam_masks['hand']['masks']], lmodel.final[..., :3]))

                _, hair_msk_assigned = try_assign_sam_mask(
                    lmodel,
                    [rle2mask(m) for m in lang_sam_masks['hair']['masks']],
                    body_part_tag='hair',
                    valid_face_ids={17}
                )

                _, face_msk_assigned = try_assign_sam_mask(
                    lmodel,
                    [rle2mask(m) for m in lang_sam_masks['face']['masks']],
                    body_part_tag='face',
                    valid_face_ids={1}
                )

                _, neck_msk_assigned = try_assign_sam_mask(
                    lmodel,
                    [rle2mask(m) for m in lang_sam_masks['face']['masks']],
                    body_part_tag='neck',
                    valid_face_ids={14, 16,}, mask_ioa_thr=0.3
                )

                _, nose_msk_assigned = try_assign_sam_mask(
                    lmodel,
                    [rle2mask(m) for m in lang_sam_masks['nose']['masks']],
                    body_part_tag='nose',
                    valid_face_ids={10, 11, 1}, mask_ioa_thr=0.3,
                    bbox_constraint=lmodel.facedet[0]['bbox'][:4]
                )

                _, mouth_msk_assigned = try_assign_sam_mask(
                    lmodel,
                    [rle2mask(m) for m in lang_sam_masks['mouth']['masks']],
                    body_part_tag='mouth',
                    valid_face_ids={10, 11, 1}, mask_ioa_thr=0.3,
                    bbox_constraint=lmodel.facedet[0]['bbox'][:4]
                )
            
            left_out_drawables: list[Drawable] = []
            for d in lmodel.drawables:
                if d.final_visible_area < 1:
                    continue
                if d.face_part_id in {2, 3, 4, 5}:  # eyes, eyebrows
                    d.body_part_tag = 'eyes'
                if d.face_part_id in {6}: # glasses
                    d.body_part_tag = 'eyewear'
                if d.face_part_id == 17:    # hair
                    if d.body_part_tag is None:
                        d.body_part_tag = 'hair'
                    elif d.body_part_tag == 'ears':
                        d.face_part_id = 7
                if d.face_part_id == 18:   # hat
                    if d.body_part_tag is None:
                        d.body_part_tag = 'headwear'
                    elif d.body_part_tag == 'ears':
                        d.face_part_id = 7
                if d.face_part_id in {7, 8}:    # ears
                    d.body_part_tag = 'ears'
                if d.face_part_id == 16: # cloth
                    if d.body_part_tag is None:
                        d.body_part_tag = 'topwear'
                    elif d.body_part_tag == 'neck':
                        d.face_part_id = 14
                if d.face_part_id == 14:    # neck
                    if not neck_msk_assigned:
                        d.body_part_tag = 'neck'
                    elif d.body_part_tag == 'topwear':
                        d.face_part_id = 16
                if not mouth_msk_assigned:
                    if d.face_part_id == 11:
                        d.body_part_tag = 'mouth'
                if not nose_msk_assigned:
                    if d.face_part_id == 10:
                        d.body_part_tag = 'nose'
                

                if d.body_part_tag is None and d.tag_stats is not None:
                    left_out_drawables.append(d)

            GENERAL_TAGS_FOR_LEFTOUT = [
                'headwear', 'eyewear', 'earwear', 'beard', 'neckwear',
                'skin', 'topwear', 'handwear',
                'bottomwear', 'legwear', 'footwear', 
                'tail', 'wings'
            ]

            def tagstats_to_general_tagstats(tagstats, valid_gtags=None):
                general_tagstats = {}
                if valid_gtags is not None:
                    valid_gtags = set(valid_gtags)
                for t, stats in tagstats.items():
                    if t not in tag2generaltag:
                        continue
                    gt = tag2generaltag[t]
                    if valid_gtags is not None and gt not in valid_gtags:
                        continue
                    if gt in general_tagstats:
                        if general_tagstats[gt]['avg_score'] < stats['avg_score']:
                            general_tagstats[gt] = stats
                    else:
                        general_tagstats[gt] = stats

                return general_tagstats

            for d in left_out_drawables:
                tagstats = tagstats_to_general_tagstats(d.tag_stats, valid_gtags=GENERAL_TAGS_FOR_LEFTOUT)
                sorted_items = sorted(tagstats.items(), key=lambda item: item[1]['avg_score'], reverse=True)
                tagstats = dict(sorted_items)
                if 'handwear' in tagstats:
                    if not mask_cover_pos(d.final_visible_mask, pos, [5, 6, 7, 8, 9, 10], xshift=-d.x, yshift=-d.y):
                        if mask_cover_pos(d.final_visible_mask, pos, [11, 12], xshift=-d.x, yshift=-d.y):
                            tagstats.pop('handwear')
                tagstats_lst = list(tagstats.keys())
                if len(tagstats) > 1 and tagstats[tagstats_lst[0]]['avg_score'] > 0.1:
                    d.body_part_tag = tagstats_lst[0]
                
                if d.body_part_tag is None:
                    assigned_armature, _, _ = assign_mask_to_armature(d.get_full_mask(final_visible_mask=True), pos, pos_scores)
                    if assigned_armature is not None:
                        d.body_part_tag = assigned_armature
                    
            for d in lmodel.drawables:
                if d.body_part_tag == 'topwear':
                    tagstats = tagstats_to_general_tagstats(d.tag_stats, valid_gtags=['topwear', 'bottomwear', 'legwear', 'handwear', 'hair'])
                    max_score_tag = max(list(tagstats.keys()), key=lambda x: tagstats[x]['avg_score'])
                    # d.body_part_tag = max_score_tag
                    if max_score_tag == 'handwear':
                        coords = np.where(d.final_visible_mask)
                        cx, cy = np.mean(coords[1]), np.mean(coords[0])
                        cy += d.y
                        if cy < np.mean(pos[[11, 12, 13, 14]], axis=0)[1]:
                            d.body_part_tag = max_score_tag
                    elif max_score_tag in {'legwear', 'bottomwear'}:
                        coords = np.where(d.final_visible_mask)
                        x1, y1, x2, y2 = np.min(coords[1]), np.min(coords[0]), np.max(coords[1]), np.max(coords[0])
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        cy += d.y
                        if cy > np.mean(pos[[11, 12]], axis=0)[1]:
                            d.body_part_tag = max_score_tag
                    else:
                        d.body_part_tag = max_score_tag

            head_y = np.mean(pos[[0, 1, 2, 3, 4]], axis=0)[1]
            shoulder_mid = np.mean(pos[[5, 6]], axis=0)[0]
            shoulder_y = np.mean(pos[[5, 6], 1], axis=0)
            shoulder_x1, shoulder_x2 = np.min(pos[[5, 6]], axis=0)[0], np.max(pos[[5, 6]], axis=0)[0]
            head_order = min([d.draw_order for d in lmodel.drawables if d.body_part_tag == 'face'])
            hip_y = np.mean(pos[[11, 12]], axis=0)[1]
            hip_x1, hip_x2 = pos[11][0], pos[11][1]
            if hip_x1 > hip_x2:
                hip_x1, hip_x2 = hip_x2, hip_x1
            hip_mid = (hip_x1 + hip_x2) / 2

            mask = lmodel.compose_bodypart_drawables('handwear', mask_only=True, final_visible_mask=True)
            # save_tmp_img(mask, mask2img=True)

            hair_drawabls = lmodel.get_body_part_drawables('hair')
            topwear_drawabls = lmodel.get_body_part_drawables('topwear')
            # topwear_order = None
            # if len(topwear_drawabls) > 0:
            #     topwear_order = max([d.draw_order for d in topwear_drawabls])

            for d in hair_drawabls + topwear_drawabls:
                x1, y1, x2, y2 = d.visible_xyxy
                dh = y2 - y1
                if (x1 - shoulder_mid) * (x2 - shoulder_mid) < 0:
                    continue
                if min(shoulder_x2, x2) - max(shoulder_x1, x1) > 0.3 * (shoulder_x2 - shoulder_x1):
                    continue
                # if d.draw_order > topwear_order:
                #     continue
                d_midx = (x1 + x2) / 2 
                if d_midx > shoulder_x1 and d_midx < shoulder_x2 and d.body_part_tag == 'topwear':
                    continue

                if d.y < head_y:
                    continue
                assigned_armature, armature_ids, assign_scores = assign_mask_to_armature(
                    d.get_full_mask(final_visible_mask=True), pos, pos_scores, selected_armatures={'handwear'})
                
                
                if assigned_armature is not None:
                    armature_scores = []
                    if 8 in armature_ids:
                        armature_h = np.max(pos[[6, 8, 10], 1]) - np.min(pos[[6, 8, 10], 1])
                        for h in ['handwear_0', 'handwear_1']:
                            if h in assign_scores:
                                armature_scores.append(assign_scores[h])
                    else:
                        armature_h = np.max(pos[[5, 7, 9], 1]) - np.min(pos[[5, 7, 9], 1])
                        for h in ['handwear_2', 'handwear_3']:
                            if h in assign_scores:
                                armature_scores.append(assign_scores[h])
                    armature_length = np.linalg.norm(pos[armature_ids[0]] - pos[armature_ids[1]])
                    if not (len(armature_scores) > 1 and np.mean(armature_scores) > 0.8) and dh > armature_length * 1.5 and d.body_part_tag == 'topwear':
                        continue
                    # print(int(d_midx), pos[5][1])
                    # save_tmp_img(cv2.circle(lmodel.final[..., :3].copy(), (int(d_midx), pos[5][1]), 15, (255, 255, 0), thickness=-1))
                    d.body_part_tag = assigned_armature

            for d in lmodel.drawables:
                if d.body_part_tag == 'handwear':
                    if d.y < head_y and d.draw_order < head_order:
                        d.body_part_tag = 'hair'
                if d.body_part_tag == 'legwear':
                    x1, y1, x2, y2 = d.visible_xyxy
                    if y2 < hip_y:
                        d.body_part_tag = 'topwear'
                if d.body_part_tag in {'neck', 'neckwear', 'headwear'}:
                    x1, y1, x2, y2 = d.visible_xyxy
                    if y1 > hip_y:
                        d.body_part_tag = 'topwear'

            legmask = armature_cc(
                lmodel.compose_bodypart_drawables('legwear', mask_only=True, final_visible_mask=True),
                pos, [(16, 14), (14, 12), (11, 13), (13, 15)], min_num_cc=3
            )
            dr_x1, dr_y1, dr_x2, dr_y2 = mask_xyxy(lmodel.compose_bodypart_drawables(['topwear', 'bottomwear'], mask_only=True, final_visible_mask=True))
            if legmask is not None:
                for d in lmodel.drawables:
                    x1, y1, x2, y2 = d.xyxy
                    if d.body_part_tag != 'legwear':
                        continue
                    if not np.any(legmask[y1: y2, x1: x2] & d.final_visible_mask):
                        d.body_part_tag = 'topwear'

            bottommask = armature_cc(
                lmodel.compose_bodypart_drawables('bottomwear', mask_only=True, final_visible_mask=True),
                pos, [(11, 12), (14, 12), (11, 13)]
            )
            if not np.any(bottommask):
                for d in lmodel.drawables:
                    if d.body_part_tag == 'bottomwear':
                        d.body_part_tag = 'topwear'

            
            if len(lmodel.get_body_part_drawables('bottomwear')) == 0:
                for d in lmodel.drawables:
                    if not d.body_part_tag == 'topwear':
                        continue
                    x1, y1, x2, y2 = d.visible_xyxy
                    ix1, ix2 = max(x1, hip_x1), min(x2, hip_x2)
                    if (ix2 - ix1) / (hip_x2 - hip_x1) < 0.9:
                        continue
                    my = (y1 + y2) / 2
                    if my > hip_y or (hip_y - my) <  (my - shoulder_y) / 5 and y1 > np.min(pos[[5, 6], 1]):
                        d.body_part_tag = 'bottomwear'

            bt_x1, bt_y1, bt_x2, bt_y2 = mask_xyxy(lmodel.compose_bodypart_drawables(['bottomwear'], mask_only=True, final_visible_mask=True))

            neck_drawables = lmodel.get_body_part_drawables('neck')
            if len(neck_drawables) > 0:
                neck_order = min(d.draw_order for d in neck_drawables)
                neck_mask = lmodel.compose_bodypart_drawables('neck', mask_only=True, final_visible_mask=False)

                # save_tmp_img(neck_mask, mask2img=True)
                neck_mask2 = lmodel.compose_bodypart_drawables('neck', mask_only=True, final_visible_mask=True)
                n2x1, n2y1, n2x2, n2y2 = mask_xyxy(neck_mask2)
                
                neck_mask[:n2y1] = 0
                nx1, ny1, nx2, ny2 = mask_xyxy(neck_mask)
                neck_mask = neck_mask[ny1: ny2, nx1: nx2]
                nh = ny2 - ny1

            # dr_x1, dr_y1, dr_x2, dr_y2 = mask_xyxy(lmodel.compose_bodypart_drawables(['topwear', 'bottomwear'], mask_only=True, final_visible_mask=True))
            
            lg_x1, lg_y1, lg_x2, lg_y2 = mask_xyxy(lmodel.compose_bodypart_drawables(['legwear'], mask_only=True, final_visible_mask=True))
            no_leg = lg_x1 == 0 and lg_x2 == 0 and lg_y1 == 0 and lg_y2 == 0
            feet_drawable_exists = False
            for d in lmodel.drawables:
                if d.body_part_tag is None:
                    continue
                x1, y1, x2, y2 = d.visible_xyxy
                my = (y1 + y2) / 2
                dh = y2 - y1
                if d.body_part_tag in {'footwear', 'wings'}:
                    if x1 > dr_x1 and y1 > dr_y1 and x2 < dr_x2 and y2 < dr_y2:
                        d.body_part_tag = 'topwear'
                        continue
                    elif d.body_part_tag == 'footwear':
                        leg_len = (13, 15) if (x1 + x2) / 2 < hip_mid else (16, 14)
                        leg_len = np.linalg.norm(pos[leg_len[0]] - pos[leg_len[1]])
                        if np.abs(y1 - y2) > leg_len * 1.0:
                            d.body_part_tag = 'legwear'
                        elif no_leg:
                            d.body_part_tag = 'legwear'
                        elif np.abs(y1 - y2) > 2 * (lg_y2 - lg_y1):
                            d.body_part_tag = 'legwear'

                elif d.body_part_tag in {'hair'}:
                    if y1 > fy2 and (x1 - shoulder_mid) * (x2 - shoulder_mid) < 0 and x1 > dr_x1 and x2 < dr_x2 and y1 > dr_y1 and y2 < dr_y2:
                        d.body_part_tag = 'topwear'
                    elif len(neck_drawables) > 0 and neck_order < d.draw_order  and y1 > ny1 - nh:
                        mask_i = d.bitwise_and(neck_mask, [nx1, ny1, nx2, ny2], final_vis_mask=True)
                        if np.any(mask_i):
                            d.body_part_tag = 'topwear'
                    # d.body_part_tag =

                elif d.body_part_tag == 'neckwear':
                    if (hip_y - my) <  (my - shoulder_y):
                        d.body_part_tag = 'topwear'
                    
                elif d.body_part_tag == 'topwear':
                    if bt_y1 > 0 and y1 > bt_y1 and y2 < bt_y2 and x1 > bt_x1 and x2 < bt_x2:
                        d.body_part_tag = 'bottomwear'

                if d.body_part_tag == 'footwear':
                    if feet_valid:
                        feet_drawable_exists = True
                    else:
                        d.body_part_tag = 'legwear'

            feet_mask_valid = feet_drawable_exists == feet_valid
            # print(feet_drawable_exists, feet_valid, feet_mask_valid)


            metadata = {'tag_valid': {k: True for k in VALID_BODY_PARTS_V1}}
            metadata['tag_valid']['footwear'] = feet_mask_valid
            lmodel.save_body_parsing(metadata=metadata)
            tagged_drawables = [d for d in lmodel.drawables if d.body_part_tag in VALID_BODY_PARTS_V1]
            init_drawable_visible_map(tagged_drawables)
            is_valid, masks, final_img = _compose_body_samples(lmodel,)

            reference_img = lmodel.final[..., :3]
            # from utils.visualize import visualize_pos_keypoints
            # savep = None
            # savep = osp.join('workspace/segs', osp.basename(model_dir) + '.png')
            # imwrite(savep, reference_img)
            # savep = osp.join('workspace/segs', osp.basename(model_dir) + '_segs.png')
            # imwrite(savep, visualize_segs_with_labels(masks, final_img, tag_list=VALID_BODY_PARTS_V1, image_weight=0.0, draw_legend=False))

            # savep = osp.join('workspace/cases/output', osp.basename(model_dir) + '.png')
            # reference_img = np.array(visualize_pos_keypoints(reference_img, keypoints=pos[..., ::-1]))
            # save_tmp_img(visualize_segs_with_labels(masks, final_img, tag_list=VALID_BODY_PARTS_V1, image_weight=0.0, reference_img=reference_img))

            foot_msk_idx = VALID_BODY_PARTS_V1.index('footwear')
            leg_msk_idx = VALID_BODY_PARTS_V1.index('legwear')
            masks[leg_msk_idx] = masks[leg_msk_idx] | masks[foot_msk_idx]
            
            bgp = random.choice(bg_list)
            fh, fw = final_img.shape[:2]
            bg = imread(bgp)
            fsize = max(fh, fw)
            target_bg_size = max(fsize, TARGET_FRAME_SIZE)
            fsze_max = int(round(fsize * 1.6))
            if fsze_max != target_bg_size:
                target_bg_size = random.randint(min(fsze_max, target_bg_size), max(fsze_max, target_bg_size))
            bg = resize_short_side_to(bg, target_bg_size)
            bg = random_crop(imread(bgp), (target_bg_size, target_bg_size))
            
            if fh != target_bg_size or fw != target_bg_size:
                px = py = 0
                if fh != target_bg_size:
                    py = random.randint(0, target_bg_size - fh)
                if fw != target_bg_size:
                    px = random.randint(0, target_bg_size - fw)
                blank_final = np.zeros((target_bg_size, target_bg_size, 4), np.uint8)
                blank_final[py: py + fh, px: px + fw] = final_img
                final_img = blank_final
                for mi, m in enumerate(masks):
                    blank = np.zeros((target_bg_size, target_bg_size), bool)
                    blank[py: py + fh, px: px + fw] = m
                    masks[mi] = blank
            fh, fw = final_img.shape[:2]

            color_correct = color_correction_sampler.sample()
            if color_correct == 'hist_match':
                fgbg_hist_matching([final_img], bg)

            face_wbg = img_alpha_blending([bg, final_img])
            
            if color_correct == 'quantize':
                mask = final_img[..., -1] > 35
                # cv2.imshow("mask", mask)
                face_wbg[..., :3] = quantize_image(face_wbg[..., :3], random.choice([12, 16, 32]), 'kmeans', mask=mask)[0]
            
            d = osp.abspath(model_dir).replace('\\', '/').rstrip('/').replace('.', '_DOT_')
            d1 = d.split('/')[-1]
            d2 = d.split('/')[-3]
            savename = d2 + '____' + d1
            savep = osp.join(save_dir, savename)
            # save_tmp_img(face_wbg)
            imwrite(savep, face_wbg, quality=97, ext='.jpg')

            mask_meta_list = [{} for _ in range(len(VALID_BODY_PARTS_V1))] # dont use [{}] * len
            mask_meta_list[foot_msk_idx]['is_valid'] = feet_mask_valid
            batch_save_masks(masks, savep + '.json', mask_meta_list=mask_meta_list)

        except Exception as e:
            # raise
            print(f'Failed to process {p}: {e}')



@cli.command('parse_render_body_samples_wsegs')
@click.option('--exec_list')
@click.option('--bg_list')
@click.option('--mask_name')
@click.option('--save_dir', default='')
@click.option('--rank_to_worldsize', default='', type=str)
def parse_render_body_samples_wsegs(exec_list, bg_list, mask_name, save_dir, rank_to_worldsize):

    from live2d.scrap_model import animal_ear_detected, Drawable
    from utils.cv import fgbg_hist_matching, quantize_image, random_crop, rle2mask, mask2rle, img_alpha_blending, resize_short_side_to, batch_save_masks, batch_load_masks
    from utils.torch_utils import seed_everything

    def _compose_body_samples(lmodel: Live2DScrapModel):
        '''
        some augmentation can be done here
        '''

        part_mask_list = []

        body_final = lmodel.compose_bodypart_drawables(VALID_BODY_PARTS_V1)

        for tag in VALID_BODY_PARTS_V1:
            m  = lmodel.compose_bodypart_drawables(tag, mask_only=True, final_visible_mask=True).astype(np.uint8)
            # save_tmp_img(m, mask2img=True)
            part_mask_list.append(m)

        return True, part_mask_list, body_final

    seed_everything(42)

    hist_match_prob = 0.2
    quantize_prob = 0.25
    color_correction_sampler = NameSampler({'hist_match': hist_match_prob, 'quantize': quantize_prob})
    
    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)
    bg_list = load_exec_list(bg_list)

    tagcluster_bodypart = json2dict('workspace/datasets/tagcluster_bodypart.json')
    tag2generaltag = {}
    for general_tag, tlist in tagcluster_bodypart.items():
        for t in tlist:
            if t in tag2generaltag and tag2generaltag[t] != general_tag:
                print(f'conflict tag def: {t} - {general_tag}, ' + tag2generaltag[t])
            tag2generaltag[t] = general_tag

    if save_dir != '':
        os.makedirs(save_dir, exist_ok=True)
    render_sample = save_dir != ''

    for ii, p in enumerate(tqdm(exec_list[0:])):
        try:

            instance_mask, crop_xyxy, score = load_detected_character(p)
            if instance_mask is None:
                print(f'skip {p}, no character instance detected')
                continue
            
            lmodel = Live2DScrapModel(p, crop_xyxy=crop_xyxy, pad_to_square=False)
            model_dir = lmodel.directory

            if len(lmodel.facedet) == 0:
                print(f'skip {model_dir}, no face detected')

            pos_estim = load_pos_estimation(model_dir)
            pos = pos_estim['pos']
            pos_scores = pos_estim['scores']

            if pos is None:
                print(f'skip {model_dir}, no pos detected')
                continue
            
            pos = np.round(pos[:, ::-1]).astype(np.int32)

            tag_loaded = lmodel.load_tag_stats()
            if not tag_loaded:
                print(f'skip {model_dir}, no valid tag stats')
                continue
            tags = json2dict(osp.join(model_dir, 'general_tags.json'))
            general_tags = set([tag2generaltag[k] for k in tags if k in tag2generaltag])
            has_animal_ear = animal_ear_detected(tags)

            face_parsing_loaded = lmodel.load_face_parsing()
            if not face_parsing_loaded:
                print(f'skip {model_dir}, no face_parsing')
                continue

            bodyparsing_loaded = lmodel.load_body_parsing()
            if not bodyparsing_loaded:
                print(f'skip {model_dir}, no bodyparsing_loaded')
                continue

            head_y = np.mean(pos[[0, 1, 2, 3, 4]], axis=0)[1]
            shoulder_mid = np.mean(pos[[5, 6]], axis=0)[0]
            shoulder_y = np.mean(pos[[5, 6], 1], axis=0)
            shoulder_x1, shoulder_x2 = np.min(pos[[5, 6]], axis=0)[0], np.max(pos[[5, 6]], axis=0)[0]
            head_order = min([d.draw_order for d in lmodel.drawables if d.body_part_tag == 'face'])
            hip_y = np.mean(pos[[11, 12]], axis=0)[1]
            hip_x1, hip_x2 = pos[11][0], pos[11][1]
            belly_y = (shoulder_y + hip_y) / 2

            metadata = lmodel._body_parsing['metadata']
            feet_mask_valid = metadata['tag_valid']['footwear']

            masks_ann = json2dict(osp.join(model_dir, mask_name + '.json'))
            sam_masks = [rle2mask(m, to_bool=True) for m in masks_ann]

            tagged_drawables = [d for d in lmodel.drawables if d.body_part_tag in VALID_BODY_PARTS_V1]
            init_drawable_visible_map(tagged_drawables)

            for tg in tagged_drawables:
                if tg.body_part_tag in {'eyewear', 'nose', 'mouth', 'eyes', 'face', 'tail'}:
                    continue
                ori_tag = tg.body_part_tag

                if feet_mask_valid and ori_tag == 'footwear':
                    continue
            
                score_list = []
                for m in sam_masks:
                    area, u_area, i_area = tg.mask_union_intersection(m, final_vis_mask=True)
                    if i_area is None:
                        i_area = -1
                    score = i_area / tg.final_visible_area
                    score_list.append(score)
                best_match = np.argmax(np.array(score_list))
                best_match = VALID_BODY_PARTS_V1[best_match]
                tg.body_part_tag = best_match
                if tg.body_part_tag == 'legwear' and score_list[VALID_BODY_PARTS_V1.index('footwear') > 0.5]:
                    tg.body_part_tag = 'footwear'
                if ori_tag in {'bottomwear', 'legwear', 'footwear'} and tg.body_part_tag in {'handwear', 'hair'}:
                    tg.body_part_tag = ori_tag

            for tg in tagged_drawables:
                x1, y1, x2, y2 = tg.xyxy
                if tg.body_part_tag in {'headwear', 'neck', 'neckwear'} and (y1 + y2) / 2 > belly_y:
                    tg.body_part_tag = 'topwear'


            lmodel.save_body_parsing(metadata=metadata, save_name='parsinglog_' + mask_name)

            if not render_sample:
                continue

            is_valid, masks, final_img = _compose_body_samples(lmodel,)

            reference_img = lmodel.final[..., :3]
            from utils.visualize import visualize_pos_keypoints
            savep = None
            # savep = osp.join('workspace/segs', osp.basename(model_dir) + '.png')
            # imwrite(savep, reference_img)
            # savep = osp.join('workspace/segs', osp.basename(model_dir) + '_segs.png')
            # imwrite(savep, visualize_segs_with_labels(masks, final_img, tag_list=VALID_BODY_PARTS_V1, image_weight=0.0, draw_legend=True, reference_img=reference_img))

            # savep = osp.join('workspace/cases24k', osp.basename(model_dir) + '.png')
            # reference_img = np.array(visualize_pos_keypoints(reference_img, keypoints=pos[..., ::-1]))
            # save_tmp_img(visualize_segs_with_labels(masks, final_img, tag_list=VALID_BODY_PARTS_V1, image_weight=0.0, reference_img=reference_img), savep=savep)
            # continue

            foot_msk_idx = VALID_BODY_PARTS_V1.index('footwear')
            leg_msk_idx = VALID_BODY_PARTS_V1.index('legwear')
            masks[leg_msk_idx] = masks[leg_msk_idx] | masks[foot_msk_idx]
            
            bgp = random.choice(bg_list)
            fh, fw = final_img.shape[:2]
            bg = imread(bgp)
            fsize = max(fh, fw)
            target_bg_size = max(fsize, TARGET_FRAME_SIZE)
            fsze_max = int(round(fsize * 1.6))
            if fsze_max != target_bg_size:
                target_bg_size = random.randint(min(fsze_max, target_bg_size), max(fsze_max, target_bg_size))
            bg = resize_short_side_to(bg, target_bg_size)
            bg = random_crop(imread(bgp), (target_bg_size, target_bg_size))
            
            if fh != target_bg_size or fw != target_bg_size:
                px = py = 0
                if fh != target_bg_size:
                    py = random.randint(0, target_bg_size - fh)
                if fw != target_bg_size:
                    px = random.randint(0, target_bg_size - fw)
                blank_final = np.zeros((target_bg_size, target_bg_size, 4), np.uint8)
                blank_final[py: py + fh, px: px + fw] = final_img
                final_img = blank_final
                for mi, m in enumerate(masks):
                    blank = np.zeros((target_bg_size, target_bg_size), bool)
                    blank[py: py + fh, px: px + fw] = m
                    masks[mi] = blank
            fh, fw = final_img.shape[:2]

            color_correct = color_correction_sampler.sample()
            if color_correct == 'hist_match':
                fgbg_hist_matching([final_img], bg)

            face_wbg = img_alpha_blending([bg, final_img])
            
            if color_correct == 'quantize':
                mask = final_img[..., -1] > 35
                # cv2.imshow("mask", mask)
                face_wbg[..., :3] = quantize_image(face_wbg[..., :3], random.choice([12, 16, 32]), 'kmeans', mask=mask)[0]
            
            d = osp.abspath(model_dir).replace('\\', '/').rstrip('/').replace('.', '_DOT_')
            d1 = d.split('/')[-1]
            d2 = d.split('/')[-3]
            savename = d2 + '____' + d1
            savep = osp.join(save_dir, savename)
            # save_tmp_img(face_wbg)
            imwrite(savep, face_wbg, quality=97, ext='.jpg')

            mask_meta_list = [{} for _ in range(len(VALID_BODY_PARTS_V1))] # dont use [{}] * len
            mask_meta_list[foot_msk_idx]['is_valid'] = feet_mask_valid
            batch_save_masks(masks, savep + '.json', mask_meta_list=mask_meta_list)

        except Exception as e:
            # raise
            print(f'Failed to process {p}: {e}')


@cli.command('render_face_samples')
@click.option('--exec_list')
@click.option('--bg_list')
@click.option('--save_dir')
@click.option('--rank_to_worldsize', default='', type=str)
def render_face_samples(exec_list, bg_list, save_dir, rank_to_worldsize):

    TARGET_FRAME_SIZE = 2048

    from utils.cv import fgbg_hist_matching, quantize_image, random_crop, img_bbox, img_alpha_blending, resize_short_side_to, batch_save_masks, batch_load_masks
    from utils.torch_utils import seed_everything
    from utils.visualize import FACE_LABEL2NAME

    def _compose_face_samples(lmodel: Live2DScrapModel):
        '''
        todo: save complete part
        '''
        face_xyxy = lmodel.face_seg_xyxy
        face_h, face_w = face_xyxy[3] - face_xyxy[1], face_xyxy[2] - face_xyxy[0]
        all_face_labels = list(FACE_LABEL2NAME.keys())
        face_final = lmodel.compose_face_drawables(list(FACE_LABEL2NAME.keys()), xyxy=face_xyxy)
        # save_tmp_img(face_final, 'local_tmp.png')

        part_mask_list = []
        # segmap = np.zeros((face_h, face_w), dtype=np.int32)
        alphas = np.zeros((face_h, face_w), dtype=np.int32)
        for ii in range(1, len(all_face_labels)):
            m  = lmodel.compose_face_drawables(ii, mask_only=True, xyxy=face_xyxy, final_visible_mask=True).astype(np.uint8)
            # save_tmp_img(m, mask2img=True)
            part_mask_list.append(m)

        mask_bg = np.bitwise_not(np.bitwise_or.reduce(np.stack(part_mask_list).astype(bool), axis=0))
        part_mask_list.insert(0, mask_bg.astype(np.uint8))
        
        nose_detected, mouth_detected = lmodel.face_part_detected([10, 11])
        tp = osp.join(lmodel.directory, 'faceseg_nosemouth.json.gz')
        if osp.exists(tp) and (not nose_detected or not mouth_detected):
            nose_mouth = batch_load_masks(tp)
            if not nose_detected:
                part_mask_list[10] = nose_mouth[0]
                part_mask_list[1][np.where(nose_mouth[0] > 0)] = 0
            if not mouth_detected:
                part_mask_list[11] = nose_mouth[1]
                part_mask_list[1][np.where(nose_mouth[1] > 0)] = 0

        bx, by, bw, bh = cv2.boundingRect(cv2.findNonZero(part_mask_list[0].astype(np.uint8)))
        by2 = by + bh
        bx2 = bw + bx

        # DONT DELETE THESE!!!!
        
        # depth_lower = 100000
        # depth_upper = -1
        # for d_id, drawable in enumerate(lmodel.drawables):
        #     if drawable.area < 1 or not drawable.face_part_id == 1:
        #         continue

        #     dx, dy, dw, dh = drawable.get_bbox(xyxy=face_xyxy)
        #     dx2 = dx + dw
        #     dy2 = dy + dh

        #     # check if hair drawable is actually background
        #     if drawable.face_part_id == 17:
        #         if drawable.face_part_stats['ioa'][0] > 0.7 and drawable.face_part_stats['ioa'][17] < 0.3:
        #             drawable.face_part_id = None

        #     if drawable.face_part_id == 1 and dw / bw > 0.7 and dh > bw > 0.7:
        #         if drawable.draw_order < depth_lower:
        #             depth_lower = drawable.draw_order
        #         if drawable.draw_order > depth_upper:
        #             depth_upper = drawable.draw_order

        # depth_buffer = np.zeros((face_h, face_w), dtype=np.uint8)
        # base_depth = 1
        # mask = np.zeros_like(depth_buffer, dtype=bool)
        # valid_face_ids = set(range(1, 19))
        # for d in lmodel.drawables:
        #     if d.area < 1 or d.face_part_id not in valid_face_ids:
        #         continue
        #     if np.any(d.bitwise_and(mask, face_xyxy)):
        #         base_depth += 1
        #     m = d.get_full_mask(xyxy=face_xyxy)
        #     mask |= m 
        #     d.depth = base_depth
        #     depth_buffer[np.where(m)] = base_depth
        
        # depth = (depth_buffer / np.max(depth_buffer) * 255).astype(np.uint8)
        # save_tmp_img(depth)


        # base_face_mask = compose_from_drawables([d for d in lmodel.drawables if \
        #                                          drawable.draw_order >= depth_lower and drawable.draw_order > depth_upper])
        # for drawable in lmodel.drawables:
        #     if drawable.draw_order < depth_lower or drawable.draw_order > depth_upper:
        #         continue

        # segmap = segmap.astype(np.uint8)
        # lmodel.compose_face_drawables([4, 5], output_type='pil').save('local_tst.png')
        # save_tmp_img(face_final)
        # save_tmp_img(segmap == 1, mask2img=True)
        # save_tmp_img(segmap == 4, mask2img=True)

        return True, part_mask_list, face_final
    
    os.makedirs(save_dir, exist_ok=True)

    seed_everything(42)

    hist_match_prob = 0.2
    quantize_prob = 0.25
    color_correction_sampler = NameSampler({'hist_match': hist_match_prob, 'quantize': quantize_prob})
    
    if exec_list.endswith('.json'):
        new_exec_list = []
        exec_list = json2dict(exec_list)
        for k, vs in exec_list.items():
            for v in vs:
                new_exec_list.append({v: k})
        exec_list = new_exec_list
        pass
    
    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)
    bg_list = load_exec_list(bg_list)

    VALID_FACE_SET = set(range(19))
    for ii, p in enumerate(tqdm(exec_list[0:])):
        try:
            face_parsingp = None
            if isinstance(p, dict):
                for k, v in p.items():
                    p = k
                    face_parsingp = osp.join(v, 'face_parsing.json')
            lmodel = Live2DScrapModel(p)
            model_dir = lmodel.directory

            if face_parsingp is None:
                face_parsingp = osp.join(model_dir, 'face_parsing.json')
                if not osp.exists(face_parsingp):
                    face_parsingp = '-'.join(model_dir.split('-')[:-1]) + '-4'
                    face_parsingp = osp.join(face_parsingp, 'face_parsing.json')
            if not osp.exists(face_parsingp):
                print(f"skip {p} due to face parsing result not found")
                continue
            
            lmodel.load_face_parsing(face_parsingp)
            face_drawables = [d for d in lmodel.drawables if d.face_part_id in VALID_FACE_SET]
            init_drawable_visible_map(face_drawables)

            is_valid, labels, face_final = _compose_face_samples(lmodel,)
            mask_list = labels
            if not is_valid:
                continue
            # save_tmp_img(labels[0], mask2img=True)

            bgp = random.choice(bg_list)
            fh, fw = face_final.shape[:2]
            bg = imread(bgp)
            bgh, bgw = bg.shape[:2]
            target_bg_size = min(bgh, bgw, TARGET_FRAME_SIZE)
            fsize = max(fh, fw)
            if fsize * 2 < target_bg_size:
                target_bg_size = random.randint(fsize * 2, target_bg_size)
            bg = resize_short_side_to(bg, target_bg_size)
            bg = random_crop(imread(bgp), (fh, fw))
            # save_tmp_img(bg)
            
            color_correct = color_correction_sampler.sample()
            if color_correct == 'hist_match':
                fgbg_hist_matching([face_final], bg)

            face_wbg = img_alpha_blending([bg, face_final])
            
            if color_correct == 'quantize':
                mask = face_final[..., -1] > 35
                # cv2.imshow("mask", mask)
                face_wbg[..., :3] = quantize_image(face_wbg[..., :3], random.choice([12, 16, 32]), 'kmeans', mask=mask)[0]
            
            d = osp.abspath(model_dir).replace('\\', '/').rstrip('/').replace('.', '_DOT_')
            d1 = d.split('/')[-1]
            d2 = d.split('/')[-3]
            savename = d2 + '____' + d1
            savep = osp.join(save_dir, savename)
            # save_tmp_img(face_wbg)
            imwrite(savep, face_wbg, quality=97, ext='.jpg')
            batch_save_masks(mask_list, savep + '.json', compress='gzip')
            # print(f'finished {savep}')
        except Exception as e:
            # raise
            print(f'Failed to process {p}: {e}')


@cli.command('dump_clean_lst')
@click.option('--exec_list')
@click.option('--savep', default=None)
def dump_clean_lst(exec_list, savep):

    final_list = []

    for exec_list in exec_list.split(','):
        save_dir = osp.dirname(exec_list)
        exec_src = exec_list
        exec_src_name = osp.basename(exec_src)
        srcd = osp.dirname(exec_src)
        execsrc2tgt = osp.join(srcd, exec_src_name.split('chunk')[0].rstrip('_')) + '.txt.json'
        assert osp.exists(execsrc2tgt)
        execsrc2tgt = json2dict(execsrc2tgt)

        exec_list = load_exec_list(exec_list)

        for srcp in tqdm(exec_list):
            tgt_list = execsrc2tgt[srcp]
            if srcp not in tgt_list:
                tgt_list.append(srcp)
            final_list += tgt_list

    if savep is None:
        savep = osp.join(save_dir, 'cleaned_list.txt')
    with open(savep, 'w', encoding='utf8') as f:
        f.write('\n'.join(final_list))


@cli.command('dump_complete_lst')
@click.option('--exec_list')
@click.option('--savep', default=None)
def dump_complete_lst(exec_list, savep):

    final_list = []

    from utils.io_utils import load_exec_list

    execp = osp.splitext(exec_list)[0]
    savep = execp + '_complete.txt'
    for p in load_exec_list(exec_list):
        filep = osp.splitext(p)[0] + '_ann.json'
        ann = json2dict(filep)
        if not ann['cleaned']:
            continue
        if ann['is_incomplete']:
            continue
        final_list.append(p)

    print(f'filtered: {len(final_list)}')
    with open(savep, 'w', encoding='utf8') as f:
        f.write('\n'.join(final_list))


@cli.command('update_bodyparsing')
@click.option('--exec_list')
@click.option('--tgt_dir')
@click.option('--parsing_name')
@click.option('--tgt_parsing_name', default='body_parsing.json')
def update_bodyparsing(exec_list, tgt_dir, parsing_name, tgt_parsing_name):

    def _check_tag_valid(tag):
        valid = True
        if 'tag_valid' in metadata_tgt:
            if tag in metadata_tgt['tag_valid']:
                valid = metadata_tgt['tag_valid'][tag]
            else:
                valid = False
        if not valid:
            for k, v in src['drawable'].items():
                if v == tag:
                    valid = True
                    break
        return valid

    if tgt_parsing_name is None:
        tgt_parsing_name = parsing_name

    exec_src = exec_list
    exec_src_name = osp.basename(exec_src)
    srcd = osp.dirname(exec_src)
    execsrc2tgt = osp.join(srcd, exec_src_name.split('chunk')[0].rstrip('_')) + '.txt.json'
    assert osp.exists(execsrc2tgt), f'{execsrc2tgt} does not exist!'
    execsrc2tgt = json2dict(execsrc2tgt)

    exec_list = load_exec_list(exec_list)

    n_updates = 0
    not_exists = 0
    for srcp in tqdm(exec_list):
        tgt_list = execsrc2tgt[srcp]
        if srcp not in tgt_list:
            tgt_list.append(srcp)
        relsrcp = osp.relpath(srcp, tgt_dir)
        srcp = osp.join(srcd, relsrcp)
        src_parsing = osp.join(srcp, parsing_name)

        if not osp.exists(srcp):
            print(srcp)
            not_exists += 1
            continue

        src = json2dict(src_parsing)

        metadata_src = src['metadata']
        # metadata_src['is_valid'] = True
        if 'tag_valid' not in metadata_src:
            metadata_src['tag_valid'] = {}
        metadata_src['tag_valid']['objects'] = True
        metadata_src['tag_valid']['footwear'] = True
        
        for tgtp in tgt_list:
            tgt_parsing = osp.join(tgtp, tgt_parsing_name)
            # if osp.exists(tgt_parsing):
            #     tgt = json2dict(tgt_parsing)
            #     footwear_valid = True
            #     metadata_tgt = tgt.get('metadata', {})
            #     footwear_valid = _check_tag_valid('footwear')
            #     metadata_src['tag_valid']['footwear'] = footwear_valid
            dict2json(src, tgt_parsing)
            n_updates += 1
    print(not_exists)



@cli.command('render_body_samples')
@click.option('--exec_list')
@click.option('--bg_list')
@click.option('--mask_name', default='bodyparsingv3.json')
@click.option('--save_dir', default='')
@click.option('--rank_to_worldsize', default='', type=str)
@click.option('--save_suffix', default='.png', type=str)
def render_body_samples(exec_list, bg_list, mask_name, save_dir, rank_to_worldsize, save_suffix):

    from live2d.scrap_model import animal_ear_detected, Drawable, ImageProcessor, compose_from_drawables, VALID_BODY_PARTS_V3
    from utils.cv import fgbg_hist_matching, quantize_image, random_crop, rle2mask, mask2rle, img_alpha_blending, resize_short_side_to, batch_save_masks, batch_load_masks
    from utils.torch_utils import seed_everything

    seed_everything(42)

    hist_match_prob = 0.35
    # quantize_prob = 0.25
    color_correction_sampler = NameSampler({'hist_match': hist_match_prob, 'quantize': 0.})
    
    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)
    bg_list = load_exec_list(bg_list)

    tagcluster_bodypart = json2dict('assets/tagcluster_bodypart_v2.json')
    tag2generaltag = {}
    for general_tag, tlist in tagcluster_bodypart.items():
        for t in tlist:
            if t in tag2generaltag and tag2generaltag[t] != general_tag:
                print(f'conflict tag def: {t} - {general_tag}, ' + tag2generaltag[t])
            tag2generaltag[t] = general_tag

    if save_dir != '':
        os.makedirs(save_dir, exist_ok=True)
    render_sample = save_dir != ''

    MAX_TGT_SIZE = 1280
    target_tag_list = VALID_BODY_PARTS_V3 + ['head']

    invalid_lst: list[int] = [2094, 1389, 627, 477, 280, 480]


    for ii, p in enumerate(tqdm(exec_list)):
        try:
            lmodel = Live2DScrapModel(p)
            load_success = lmodel.load_body_parsing(mask_name)
            if not load_success:
                print(f'failed to load body parsing, skip: {p}')
                continue

            metadata = lmodel._body_parsing['metadata']
            is_valid = metadata.get('is_valid', True)
            is_incomplete = metadata.get('is_incomplete', False)
            is_cleaned = metadata.get('cleaned', False)
            tag_valid = metadata.get('tag_valid', {})
            object_valid = True
            foot_valid = True

            if not is_valid:
                continue

            # if is_incomplete:
            #     continue

            # keep_bg = random.random() < 0.3
            keep_bg = False

            if not is_valid:
                continue
            
            valid_drawables: list[Drawable] = []
            body_drawables: list[Drawable] = []
            h, w = lmodel.size()
            x_min, x_max, y_min, y_max = w, 0, h, 0
            for d in lmodel.drawables:
                d.get_img()
                if d.area < 1:
                    continue
                if not keep_bg and d.body_part_tag not in target_tag_list:
                    continue
                valid_drawables.append(d)
                if d.body_part_tag in target_tag_list:
                    body_drawables.append(d)
                dxyxy = d.xyxy
                x_min = min(x_min, dxyxy[0])
                x_max = max(x_max, dxyxy[2])
                y_min = min(y_min, dxyxy[1])
                y_max = max(y_max, dxyxy[3])

            if keep_bg:
                x_min = y_min = 0
                x_max = w
                y_max = h

            ch, cw = y_max - y_min, x_max - x_min
            scale = min(MAX_TGT_SIZE / max(ch, cw), 1)
            nh, nw = ch, cw
            if scale < 1:
                nh = int(round(nh * scale))
                nw = int(round(nw * scale))
            new_processor = ImageProcessor(target_frame_size=[nw, nh], crop_bbox=[x_min, y_min, x_max, y_max], pad_to_square=False)
            lmodel.final = new_processor(lmodel.final, update_coords_modifiers=True)
            lmodel.final_bbox = [
                new_processor.crop_bbox[0] + x_min,
                new_processor.crop_bbox[1] + y_min,
                new_processor.crop_bbox[0] + x_max,
                new_processor.crop_bbox[1] + y_max
            ]
            for d in valid_drawables:
                d.set_img_processor(new_processor)
                d._final_size = [nh, nw]
                d.load_img(force_reload=True, img=d.img)
            
            h, w = lmodel.size()
            depth_buffer = np.zeros((h, w), dtype=np.uint16)
            base_depth = 1
            init_drawable_visible_map(valid_drawables)
            # part_mask_list, body_final = _compose_body_samples(lmodel)
            
            part_mask_list = []
            if not keep_bg:
                body_final = lmodel.compose_bodypart_drawables(target_tag_list)
            else:
                body_final = compose_from_drawables(valid_drawables)

            for tag in target_tag_list:
                m  = lmodel.compose_bodypart_drawables(tag, mask_only=True, final_visible_mask=True).astype(np.uint8)
                part_mask_list.append(m)
            
            mask = np.zeros((h, w), dtype=bool)
            for d in body_drawables:
                m = d.get_full_mask()
                if np.any(d.bitwise_and(mask, [0, 0, w, h])):
                    base_depth += 1
                    mask = m
                else:
                    mask |= m                
                d.depth = base_depth
                depth_buffer[np.where(m)] = base_depth
            
            depth_dtype = np.uint8
            if base_depth > 255:
                depth_dtype = np.uint16
            depth_buffer = depth_buffer.astype(depth_dtype)

            d = osp.abspath(lmodel.directory).replace('\\', '/').rstrip('/').replace('.', '_DOT_')
            d1 = d.split('/')[-1]
            d2 = d.split('/')[-3]
            savename = d2 + '____' + d1
            savep = osp.join(save_dir, savename)

            masks = part_mask_list
            foot_msk_idx = target_tag_list.index('footwear')
            object_msk_idx = target_tag_list.index('objects')
            leg_msk_idx = target_tag_list.index('legwear')
            masks[leg_msk_idx] = masks[leg_msk_idx] | masks[foot_msk_idx]
            px = py = 0
            
            final_img = body_final

            bgp = random.choice(bg_list)
            fh, fw = final_img.shape[:2]
            bg = imread(bgp)
            fsize = min(max(h, w), MAX_TGT_SIZE)
            fsze_max = int(round(fsize * 1.5))
            target_bg_size = random.randint(fsize, fsze_max)
            bg = resize_short_side_to(bg, target_bg_size)

            target_bg_w = target_bg_h = target_bg_size
            if fh > fw:
                target_bg_w = random.randint(fw, target_bg_size)
            elif fw > fh:
                target_bg_h = random.randint(fh, target_bg_size)

            bg = random_crop(imread(bgp), (target_bg_h, target_bg_w))
            
            px = py = 0
            if fh != target_bg_h or fw != target_bg_w:
                if fh != target_bg_h:
                    py = random.randint(0, target_bg_h - fh)
                if fw != target_bg_w:
                    px = random.randint(0, target_bg_w - fw)
                blank_final = np.zeros((target_bg_h, target_bg_w, 4), np.uint8)
                blank_final[py: py + fh, px: px + fw] = final_img
                final_img = blank_final

                depth_blank = np.zeros((target_bg_h, target_bg_w), dtype=depth_dtype)
                depth_blank[py: py + fh, px: px + fw] = depth_buffer
                depth_buffer = depth_blank

                for mi, m in enumerate(masks):
                    blank = np.zeros((target_bg_h, target_bg_w), bool)
                    blank[py: py + fh, px: px + fw] = m
                    masks[mi] = blank
            fh, fw = final_img.shape[:2]

            color_correct = color_correction_sampler.sample()

            if color_correct == 'hist_match':
                fgbg_hist_matching([final_img], bg, fg_only=True)

            wbg = img_alpha_blending([bg, final_img])
            wbg[..., -1] = final_img[..., -1]


            fh, fw = wbg.shape[:2]

            # save_tmp_img(visualize_segs_with_labels(masks, wbg[..., :3], tag_list=target_tag_list, image_weight=0.1))
            imwrite(savep, wbg, quality=100, ext=save_suffix)
            imwrite(savep + '_depth', depth_buffer, quality=100, ext='.png')
            
            mask_meta_list = [{} for _ in range(len(target_tag_list))] # dont use [{}] * len
            mask_meta_list[foot_msk_idx]['is_valid'] = foot_valid
            mask_meta_list[object_msk_idx]['is_valid'] = object_valid
            batch_save_masks(masks, savep + '.json', mask_meta_list=mask_meta_list)
            del masks
            # del wbg
            del depth_buffer

            sample_ann = {'cleaned': is_cleaned, 'is_incomplete': is_incomplete, 'tag_info': {k: {'valid': True, 'exists': False} for k in VALID_BODY_PARTS_V2}, 'final_size': wbg.shape[:2]}
            tag_info = sample_ann['tag_info']
            # tag_info['footwear']['valid'] = foot_valid
            # tag_info['objects']['valid'] = object_valid

            for ii, tag in enumerate(target_tag_list):
                # if tag == 'footwear' and not foot_valid:
                #     continue
                # if tag == 'objects' and not object_valid:
                #     continue
                if tag == 'head':
                    drawables = lmodel.get_body_part_drawables(['face', 'irides', 'eyebrow', 'eyewhite', 'eyelash', 'eyewear', 'ears', 'nose', 'mouth'])
                else:
                    drawables = lmodel.get_body_part_drawables(tag)
                # if tag == 'legwear':
                #     drawables += lmodel.get_body_part_drawables('footwear')
                drawables = [d for d in drawables if d.area >= 1]
                if len(drawables) == 0:
                    continue
                init_drawable_visible_map(drawables)
                x_min, x_max, y_min, y_max = fw, 0, fh, 0
                for d in drawables:
                    dxyxy = d.xyxy
                    x_min = min(x_min, dxyxy[0])
                    x_max = max(x_max, dxyxy[2])
                    y_min = min(y_min, dxyxy[1])
                    y_max = max(y_max, dxyxy[3])
                
                xyxy = [x_min, y_min, x_max, y_max]
                dh, dw = y_max - y_min, x_max - x_min
                part_final = compose_from_drawables(drawables, xyxy=xyxy)
                imwrite(savep + f'_{tag}', part_final, quality=100, ext='.png')
                
                depth_buffer = np.zeros((dh, dw), dtype=depth_dtype)
                for d in drawables:
                    dxyxy = d.xyxy
                    m = d.final_visible_mask
                    depth_buffer[dxyxy[1] - y_min: dxyxy[3] - y_min, dxyxy[0] - x_min: dxyxy[2] - x_min][np.where(m)] = d.depth
                
                xyxy = [x_min + px, y_min + py, x_max + px, y_max + py]
                imwrite(savep + f'_{tag}_depth', depth_buffer, quality=100, ext='.png')
                if tag not in tag_info:
                    tag_info[tag] = {}
                tag_info[tag]['exists'] = True
                tag_info[tag]['xyxy'] = xyxy

                blank = np.zeros_like(wbg)
                blank[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]] = part_final
                # save_tmp_img(wbg)
                # save_tmp_img(img_alpha_blending([wbg, blank]))
                # pass
                

            
            dict2json(sample_ann, savep + '_ann.json')

        except Exception as e:
            # raise
            print(f'Failed to process {p}: {e}')




@cli.command('render_simple_samples')
@click.option('--exec_list')
@click.option('--bg_list')
@click.option('--save_dir', default='')
@click.option('--rank_to_worldsize', default='', type=str)
@click.option('--save_suffix', default='.png', type=str)
def render_simple_samples(exec_list, bg_list, save_dir, rank_to_worldsize, save_suffix):

    from live2d.scrap_model import animal_ear_detected, Drawable, ImageProcessor, compose_from_drawables
    from utils.cv import fgbg_hist_matching, quantize_image, random_crop, rle2mask, mask2rle, img_alpha_blending, resize_short_side_to, batch_save_masks, batch_load_masks
    from utils.torch_utils import seed_everything

    seed_everything(42)

    hist_match_prob = 0.35
    # quantize_prob = 0.25
    color_correction_sampler = NameSampler({'hist_match': hist_match_prob, 'quantize': 0.})

    is_pkl = False
    if exec_list.endswith('.pkl'):
        src_dir = osp.dirname(exec_list)
        import pickle
        is_pkl = True
        with open(exec_list, 'rb') as f:
            exec_list = pickle.load(f)
    
    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)
    bg_list = load_exec_list(bg_list)

    # tagcluster_bodypart = json2dict('assets/tagcluster_bodypart_v2.json')
    # tag2generaltag = {}
    # for general_tag, tlist in tagcluster_bodypart.items():
    #     for t in tlist:
    #         if t in tag2generaltag and tag2generaltag[t] != general_tag:
    #             print(f'conflict tag def: {t} - {general_tag}, ' + tag2generaltag[t])
    #         tag2generaltag[t] = general_tag

    if save_dir != '':
        os.makedirs(save_dir, exist_ok=True)
    render_sample = save_dir != ''

    MAX_TGT_SIZE = 1024

    # valid_taglst = set(list(tag2generaltag.keys()) + ['smile'])

    nsaved = 0
    # nsaved += len(exec_list)
    for ii, p in enumerate(tqdm(exec_list)):
        try:
            if is_pkl:
                tags = p['tags']
                p = osp.join(src_dir, p['file_path'])

            img = Image.open(p).convert('RGBA')

            if not is_pkl:
                from annotators.wdv3_tagger import apply_wdv3_tagger
                img_input = pil_ensure_rgb(img)
                img_input = img_input.resize((448, 448), resample=Image.Resampling.LANCZOS)
                caption, taglist, ratings, character, general = apply_wdv3_tagger(img_input)
                tags = general

            # tags = [tag for tag in tags if tag in valid_taglst]

            img = np.array(img)
            x1, y1, x2, y2 = cv2.boundingRect(cv2.findNonZero((img[..., -1] > 25).astype(np.uint8)))
            x2 += x1
            y2 += y1
            img = img[y1: y2, x1: x2].copy()

            ch, cw = img.shape[:2]
            scale = min(MAX_TGT_SIZE / max(ch, cw), 1)
            nh, nw = ch, cw
            if scale < 1:
                nh = int(round(nh * scale))
                nw = int(round(nw * scale))
                img = cv2.resize(img, dsize=(nw, nh), interpolation=cv2.INTER_AREA)
            
            savename = str(nsaved).zfill(5)
            savep = osp.join(save_dir, savename)

            h, w = img.shape[:2]
            final_img = img

            bgp = random.choice(bg_list)
            fh, fw = final_img.shape[:2]
            bg = imread(bgp)
            fsize = min(max(h, w), MAX_TGT_SIZE)
            fsze_max = int(round(fsize * 1.5))
            target_bg_size = random.randint(fsize, fsze_max)
            bg = resize_short_side_to(bg, target_bg_size)

            target_bg_w = target_bg_h = target_bg_size
            if fh > fw:
                target_bg_w = random.randint(fw, target_bg_size)
            elif fw > fh:
                target_bg_h = random.randint(fh, target_bg_size)

            bg = random_crop(imread(bgp), (target_bg_h, target_bg_w))
            
            px = py = 0
            if fh != target_bg_h or fw != target_bg_w:
                if fh != target_bg_h:
                    py = random.randint(0, target_bg_h - fh)
                if fw != target_bg_w:
                    px = random.randint(0, target_bg_w - fw)
                blank_final = np.zeros((target_bg_h, target_bg_w, 4), np.uint8)
                blank_final[py: py + fh, px: px + fw] = final_img
                final_img = blank_final

            fh, fw = final_img.shape[:2]

            color_correct = color_correction_sampler.sample()

            if color_correct == 'hist_match':
                fgbg_hist_matching([final_img], bg, fg_only=True)

            wbg = img_alpha_blending([bg, final_img])
            wbg[..., -1] = final_img[..., -1]

            fh, fw = wbg.shape[:2]

            # save_tmp_img(visualize_segs_with_labels(masks, wbg[..., :3], tag_list=target_tag_list, image_weight=0.1))
            imwrite(savep, wbg, quality=100, ext=save_suffix)
            with open(savep + '.txt', 'w', encoding='utf8') as f:
                f.write(','.join(tags))

            nsaved += 1


        except Exception as e:
            # raise
            print(f'Failed to process {p}: {e}')




if __name__ == '__main__':
    cli()
import argparse
import sys
import os.path as osp
import os
import gc

from PIL import Image
import pandas as pd
import numpy as np
import cv2

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from utils.io_utils import json2dict, find_all_imgs, load_exec_list, pil_ensure_rgb, dict2json, imglist2imgrid, save_tmp_img
from utils.torch_utils import img2tensor, tensor2img
from metrics.clip_score import img_clip_score,get_img_feats as get_clip_img_feats
from metrics.binary_dice_loss import BinaryDiceLoss
from train.eval_utils import AvgMeter
from utils.ssim_torch import SSIMCriteria
from tqdm import tqdm
from utils.cv import smart_resize, center_square_pad_resize, pad_rgb
from torchmetrics.image.fid import FrechetInceptionDistance
from live2d.scrap_model import VALID_BODY_PARTS_V2

import random
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio

import torch
lpips_model = None
@torch.no_grad()
def calculate_lpip(src, tgt, device='cuda'):
    global lpips_model
    if lpips_model is None:
        lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device=device)
    score = lpips_model(src, tgt)
    return score


import torch
fid_model = None
@torch.no_grad()
def calculate_fid(src, tgt, device='cuda'):
    global fid_model
    if fid_model is None:
        fid_model = FrechetInceptionDistance(normalize=False)
    score = lpips_model(src, tgt)
    return score

valid_metrics = ['PIC','frame_consistency', 'avg_clip_score', 'lpip']

def resize_frame_list(frame_list, target_h, target_w):
    flist = []
    for ii in range(len(frame_list)):
        f = frame_list[ii]
        if f.shape[0] != target_h or f.shape[1] != target_w:
            f = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        flist.append(f)
    return flist

GC_INTERVAL = 32

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=str, default='fid,psnr,ssim,lpip,avg_clip_score,clip_score_img,mask_dice,mask_mse', help=f'Target metrics to run, separated by comma, valid metrics: {valid_metrics}')
    parser.add_argument('--no_progressbar', action='store_true', help=f'Disable progress bar')
    parser.add_argument('--savep', type=str, default=None, help=f'Result saving path')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--target_dir', default=None)
    parser.add_argument('--src_list', default='workspace/datasets/eval_test.txt')
    parser.add_argument('--target_size', default='1024,1024')
    parser.add_argument('--max_num', default=-1, type=int)

    args = parser.parse_args()

    target_dir = args.target_dir
    assert osp.exists(target_dir)

    metric_names = [m.strip() for m in args.metrics.split(',')]
    target_h, target_w = [int(s) for s in args.target_size.split(',')]

    use_progressbar = not args.no_progressbar

    src_list = load_exec_list(args.src_list)
    from utils.torch_utils import seed_everything
    # seed_everything(1)
    # random.shuffle(src_list)
    # src_list = src_list[:470]

    max_num = int(args.max_num)
    if max_num > 0 and len(src_list) > max_num:
        src_list = src_list[:max_num]

    device = args.device
    metrics = AvgMeter()

    srcv_batch = []
    tgtv_batch = []

    if 'ssim' in args.metrics:
        ssim_criteria = SSIMCriteria(channel=3).to(device=device)

    dice = BinaryDiceLoss().to(device=device)

    if 'fid' in args.metrics:
        fid_metrics = FrechetInceptionDistance(normalize=True).to(device=device)


    dtype = torch.float32

    def get_xyxy(img):
        tgt_mask = img[..., -1].copy()
        tgt_mask[tgt_mask <= 10] = 0
        x1, y1, x2, y2 = cv2.boundingRect(cv2.findNonZero(tgt_mask))
        if x2 == 0 or y2 == 0:
            return None
        return [x1, y1, x2 + x1, y2 + y1]

    with torch.no_grad():
        for ii, srcp in enumerate(tqdm(src_list)):
            # if ii < 1500:
            #     continue
            src_dir = osp.dirname(srcp)
            srcname = osp.splitext(osp.basename(srcp))[0]
            src_cls_lst = []
            ann = json2dict(osp.join(src_dir, srcname + '_ann.json'))
            # print(ann.keys())
            src_final_sz = ann['final_size']
            src_tag_info = ann['tag_info']
            tag_img_list = []
            tgt_img_list = []
            src_mask_list = []
            tgt_mask_list = []
            for tag in VALID_BODY_PARTS_V2:
                tag_img = np.zeros(src_final_sz + [4], dtype=np.uint8)
                p = osp.join(src_dir, srcname + f'_{tag}.png')
                src_exists = osp.exists(p)
                if src_exists:
                    xyxy = src_tag_info[tag]['xyxy']
                    tag_img[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]] = np.array(Image.open(p))
                tag_img = center_square_pad_resize(tag_img, target_h)
                tag_img = smart_resize(tag_img, (target_h, target_w))

                tgtp = osp.join(target_dir, srcname, f'{tag}.png')
                if osp.exists(tgtp):
                    tgt_img = smart_resize(np.array(Image.open(tgtp)), (target_h, target_w))
                else:
                    tgt_img = np.zeros_like(tag_img)

                xyxy1 = get_xyxy(tag_img)
                xyxy2 = get_xyxy(tgt_img)
                if xyxy1 is not None and xyxy2 is not None:
                    xyxy = [min(xyxy1[0], xyxy2[0]), min(xyxy1[1], xyxy2[1]), max(xyxy1[2], xyxy2[2]), max(xyxy1[3], xyxy2[3])]
                    tag_img = smart_resize(tag_img[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]], (target_h, target_w))
                    tgt_img = smart_resize(tgt_img[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]], (target_h, target_w))
                    # save_tmp_img(np.concat([tag_img, tgt_img]))
                    # break

                src_mask_list.append(tag_img[..., [-1]].copy())
                tag_img[..., -1][tag_img[..., -1] <= 10] = 0
                tag_img = pil_ensure_rgb(tag_img)

                tgt_mask_list.append(tgt_img[..., [-1]].copy())
                tgt_img[..., -1][tgt_img[..., -1] <= 10] = 0
                tgt_img = pil_ensure_rgb(tgt_img)
                # save_tmp_img(np.concat([tag_img, tgt_img]))
                # break
                
                if src_exists:
                    # tag_img = pad_rgb(tag_img, to_uint8=True)
                    tag_img_list.append(tag_img)
                    tgt_img_list.append(tgt_img[..., :3])
                # save_tmp_img(imglist2imgrid([tag_img[..., :3], tgt_img[..., :3]]), f'tmp/{ii}_{tag}.png')
                
            # continue
            # src_flist = np.array(resize_frame_list(src_flist, target_h=target_h, target_w=target_w))
            src_tensors = torch.cat([
                            img2tensor(f, normalize=True, mean=127.5, std=127.5, device=device, dtype=dtype) \
                        for f in tag_img_list])

            target_tensors = torch.cat([
                            img2tensor(f, normalize=True, mean=127.5, std=127.5, device=device, dtype=dtype) \
                        for f in tgt_img_list])
            
            src_mask_tensors = torch.cat([
                            img2tensor(f, normalize=True, device=device, dtype=dtype) \
                        for f in src_mask_list])

            tgt_mask_tensors = torch.cat([
                            img2tensor(f, normalize=True, device=device, dtype=dtype) \
                        for f in tgt_mask_list])

            score_dict = {}

            # if 'avg_clip_score' in args.metrics or 'clip_score_img' in args.metrics:
            #     clip_score = 0
            #     clip_score_img = 0
            #     text_feats = img_feats = None
            #     for jj, f in enumerate(target_flist):
            #         score, img_feats, text_feats = img_clip_score(f, src_ann['caption_blip'], device=device, text_feats=text_feats)
            #         if 'clip_score_img' in args.metrics:
            #             img_feats2 = get_clip_img_feats(src_flist[jj], device=device)[0]
            #             clip_score_img += torch.nn.functional.cosine_similarity(img_feats, img_feats2, dim=0).item()
            #         clip_score += score
            #     score_dict['clip_score'] = clip_score / len(target_flist)
            #     if 'clip_score_img' in args.metrics:
            #         score_dict['clip_score_img'] = clip_score_img / len(target_flist)

            if 'lpip' in args.metrics:
                lpip_bsz = 4
                lpip_score = 0
                iter_start = 0
                while True:
                    iter_end = min(iter_start + lpip_bsz, len(src_tensors))
                    lpip_score += (iter_end - iter_start) * calculate_lpip(src_tensors[iter_start: iter_end], target_tensors[iter_start: iter_end], device=device).cpu().item()
                    iter_start += lpip_bsz
                    if iter_start >= len(src_tensors):
                        lpip_score = lpip_score / len(src_tensors)
                        break
                
                score_dict['lpip_score'] = lpip_score

            if 'psnr' in args.metrics:
                psnr_metrics = PeakSignalNoiseRatio(data_range=(-1, 1)).to(device=device)
                psnr_metrics.update(target_tensors, src_tensors)
                score_dict['psnr'] = psnr_metrics.compute().item()

            if 'ssim' in args.metrics:
                score_dict['ssim'] = ssim_criteria(
                    src_tensors * 127.5 + 127.5, 
                    target_tensors * 127.5 + 127.5
                ).mean().item()

            if 'fid' in args.metrics:
                fid_metrics.update(src_tensors * 0.5 + 0.5, real=True)
                fid_metrics.update(target_tensors * 0.5 + 0.5, real=False)

            if 'mask_dice' in args.metrics:
                score_dict['mask_dice'] = dice(tgt_mask_tensors, src_mask_tensors).item()

            if 'mask_mse' in args.metrics:
                score_dict['mask_mse'] = torch.nn.functional.mse_loss(tgt_mask_tensors, src_mask_tensors).item()

            metrics.add(score_dict)
    #     # write_video(src_flist, f'local_src{ii}.mp4')
    #     # write_video(target_flist, f'local_tgt{ii}.mp4')

    #     del srcv
    #     del targetv
    #     if (ii + 1) % GC_INTERVAL == 0:
    #         gc.collect()
    #     # if ii > 10:
    #     #     break
    #     # pass

    # if args.savep is None:
    #     args.savep = args.target_dir + '.json'

    rst = metrics.compute()
    if 'fid' in args.metrics:
        rst['fid'] = fid_metrics.compute().item()
    print(rst)
    dict2json(metrics.meter_dict, args.savep)



    
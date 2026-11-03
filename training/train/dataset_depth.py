import os.path as osp
from typing import Dict, List, Union, Tuple, Optional
import random
import traceback
from enum import Enum

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
from einops import rearrange

from utils.cv import rle2mask, pad_rgb, random_hsv, random_pad_img, center_square_pad_resize
from utils.torch_utils import img2tensor, tensor2img
from utils.io_utils import json2dict, load_exec_list, find_all_imgs, imglist_from_dir_or_flist
from .dataset_layerdiff import load_blend_parts, xyxy_from_taglist


MINIMUM_VISIBLE_ALPHA = 25



class DepthDataset(Dataset):

    '''
    valid src_list: filepath, dict(sample_list=filepath), dict(sample_list=filepath, label_dir=...), or a list of these
    '''

    def __init__(
        self, 
        src_list: Union[str, Dict, List],
        target_size: Union[int, Tuple] = 896,
        random_flip=0.,
        random_crop=0.,
        random_crop_ratio=0.3,
        random_hsv=0.,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],    # SAM input mean
        pixel_std: List[float] = [58.395, 57.12, 57.375],       # SAM input std
        depth_cond_prob: float = 0.,
        load_depth_cond: bool = False,
        tag_list: List[str] = None,
        contrl_signals: str = 'depth',
        normalize_depth=False,
        to_depth: bool = False,
        **kwargs) -> None:

        super().__init__()

        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size

        self.tag_list = tag_list

        # parse source files
        sample_list = []
        if not isinstance(src_list, List):
            src_list = [src_list]
        for src in src_list:
            label_dir = None
            label_list = None
            if isinstance(src, str):
                lst = imglist_from_dir_or_flist(src)
            else:
                assert isinstance(src, Dict)
                lst = imglist_from_dir_or_flist(src['sample_list'])
                if 'label_dir' in src:
                    label_dir = src['label_dir']
                if 'label_list' in src:
                    label_list = load_exec_list(src['label_list'])
            if label_dir is not None:
                for imgp in lst:
                    sample_list.append({'img': imgp, 'label_dir': label_dir})
            elif label_list is not None:
                for imgp, labelp in zip(lst, label_list):
                    sample_list.append({'img': imgp, 'label': labelp})
            else:
                for imgp in lst:
                    sample_list.append({'img': imgp})
        self.sample_list = sample_list
        
        self.random_flip = random_flip
        self.random_hsv = random_hsv

        self.random_crop = random_crop
        self.random_crop_ratio = random_crop_ratio

        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

        self.load_depth_cond = load_depth_cond
        self.depth_cond_prob = depth_cond_prob
        self.contrl_signals = contrl_signals
        self.normalize_depth = normalize_depth
        self.to_depth = to_depth

    def get_randomcrop_xyxy(self, input_wh: Tuple):

        def _randomcrop_start_end(crop_ratio, length):
            start = 0
            end = length
            crop_len = int(round(length * crop_ratio))
            if crop_len < 1:
                return start, end
            start = random.randint(0, crop_len)
            end = start + length - crop_len
            return start, end
        
        if self.random_crop and random.random() < self.random_crop:
            crop_ratio = random.uniform(0, self.random_crop_ratio)
            if crop_ratio == 0.:
                return None
            cx1, cx2 = _randomcrop_start_end(crop_ratio, input_wh[0])
            cy1, cy2 = _randomcrop_start_end(crop_ratio, input_wh[1])
            return [cx1, cy1, cx2, cy2]
        else:
            return None
        

    def __len__(self):
        return len(self.sample_list)
    

    def get_sample(self, index: int):

        def _aug_size_transform(img):

            if pad_x != 0 or pad_y != 0:
                img = cv2.copyMakeBorder(img, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value = 0)

            if crop_xyxy is not None:
                img = img[crop_xyxy[1]: crop_xyxy[3], crop_xyxy[0]: crop_xyxy[2]]

            if target_w != img.shape[1] or target_h != img.shape[0]:
                img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)

            if flip_tb:
                img = img[::-1]
            if flip_lr:
                img = img[:, ::-1]

            return img

        target_w, target_h = self.target_size

        flip_lr = flip_tb = False
        if self.random_flip > 0 and random.random() < self.random_flip:
            if random.random() < 0.3:
                flip_tb = True
            if random.random() < 0.8:
                flip_lr = True

        sample = self.sample_list[index]
        imgp = sample['img']
        imgname = osp.join(osp.dirname(imgp), osp.splitext(osp.basename(imgp))[0])

        if 'label_dir' in sample:
            labelp = osp.join(sample['label_dir'], osp.splitext(osp.basename(imgp))[0] + '.png')
        else:
            labelp = imgname + '_depth.png'


        load_control_map = self.load_depth_cond and random.random() < self.depth_cond_prob
        if load_control_map:
            annp = imgname + '_ann.json'
            ann = json2dict(annp)
            final_size = ann['final_size']
            tag_infos = ann['tag_info']
            valid_tag_list = []
            for k, v in tag_infos.items():
                if v['exists']:
                    v['tag'] = k
                    valid_tag_list.append(v)
            target_tag = None
            if len(valid_tag_list) > 0:
                target_tag = random.choice(valid_tag_list)
            if target_tag is not None:
                tagp = imgname + '_' + target_tag['tag'] + '.png'
                tag_depthp = imgname + '_' + target_tag['tag'] + '_depth.png'
                if not osp.exists(tagp) or not osp.exists(tag_depthp):
                    load_control_map = False
                else:
                    masks_ann = json2dict(imgname + '.json')
                    tag_xyxy = target_tag['xyxy']
                    labelp = tag_depthp
                    img = np.array(Image.open(tagp))

        crop_xyxy = None
        pad_x = pad_y = 0

        labels = np.array(Image.open(labelp))

        mh, mw = labels.shape[:2]
        if mh > mw:
            pad_x = mh - mw
        elif mh < mw:
            pad_y = mw - mh
        if pad_x != 0 or pad_y != 0:
            labels = cv2.copyMakeBorder(labels, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value = 0)

        input_wh = (labels.shape[1], labels.shape[0])
        crop_xyxy = self.get_randomcrop_xyxy(input_wh)
        if crop_xyxy is not None:
            labels = labels[crop_xyxy[1]: crop_xyxy[3], crop_xyxy[0]: crop_xyxy[2]]
        
        if labels.shape[1] != target_w or labels.shape[0] != target_h:
            labels = cv2.resize(labels, self.target_size, interpolation=cv2.INTER_NEAREST)

        if flip_tb:
            labels = labels[::-1]
        if flip_lr:
            labels = labels[:, ::-1]

        depth = labels

        if not load_control_map:
            img = np.array(Image.open(imgp))
        
        img = _aug_size_transform(img)

        masks = img[..., -1] > MINIMUM_VISIBLE_ALPHA

        if np.sum(masks) < 1:
            raise Exception('Invalid Mask')

        # img = img[..., :3].copy()
        img = (pad_rgb(img) * 255).astype(np.uint8)

        if self.random_hsv > 0 and random.random() < self.random_hsv:
            img = random_hsv(img)

        
        img = img2tensor(img=img, normalize=True, mean=self.pixel_mean, std=self.pixel_std, dim_order='chw')
        masks = torch.from_numpy(masks)[None]

        depth_np = depth
        depth_max = 1
        normalize_depth = True
        if self.to_depth:
            depth = np.max(depth) - depth + 1
        if self.normalize_depth == 'marigold':
            # depth_min = np.min(depth)
            depth_max = np.max(depth) + 1e-6
            depth = depth_max - depth
            depth = depth / depth_max * 2 - 1
            normalize_depth = False
        elif self.normalize_depth:
            depth_max = depth.max()
        depth = img2tensor(img=depth, normalize=normalize_depth, mean=0, std=depth_max, dim_order='chw')

        batch = {
            'img': img,
            'masks': masks,
            'srcp': imgp,
            'depth': depth
        }

        if self.load_depth_cond:
            if load_control_map:
                # cond_depth = np.array(Image.open(imgname + '_depth.png'))[tag_xyxy[1]: tag_xyxy[3], tag_xyxy[0]: tag_xyxy[2]]
                # cond_depth = _aug_size_transform(cond_depth)
                tag_mask = rle2mask(masks_ann[self.tag_list.index(target_tag['tag'])])
                tag_mask = tag_mask[tag_xyxy[1]: tag_xyxy[3], tag_xyxy[0]: tag_xyxy[2]].astype(np.uint8) * 255
                tag_mask = _aug_size_transform(tag_mask).astype(np.float32) / 255.
                cond_depth = depth_np.astype(np.float32) * tag_mask / depth_max
                control_input = torch.from_numpy(np.stack([cond_depth, tag_mask], axis=0))
            else:
                control_input = torch.zeros((2, img.shape[-2], img.shape[-1]), dtype=torch.float32)
            batch['control_input'] = control_input
        
        return batch

        

    def __getitem__(self, index):

        while True:
            try:
                samplep = self.sample_list[index]['img']
                return self.get_sample(index)
            except:
                print(f'Failed to load {samplep}: ')
                print(traceback.format_exc())
                index = random.randint(0, len(self) - 1)
                continue



class DepthDataset3D(Dataset):

    '''
    valid src_list: filepath, dict(sample_list=filepath), dict(sample_list=filepath, label_dir=...), or a list of these
    '''

    def __init__(
        self, 
        src_list: Union[str, Dict, List],
        target_size: Union[int, Tuple] = 896,
        random_flip=0.,
        random_crop=0.,
        random_crop_ratio=0.3,
        random_hsv=0.,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],    # SAM input mean
        pixel_std: List[float] = [58.395, 57.12, 57.375],       # SAM input std
        depth_cond_prob: float = 0.,
        load_depth_cond: bool = False,
        tag_list: List[str] = None,
        contrl_signals: str = 'depth',
        normalize_depth=False,
        to_depth: bool = False,
        pad_argb: bool = True,
        mix_tag_groups: list = None,
        include_blend_cls = False,
        ngroup_iters=1,
        random_pad_prob=0.,
        **kwargs) -> None:

        super().__init__()

        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size

        self.tag_list = tag_list

        # parse source files
        sample_list = []
        if not isinstance(src_list, List):
            src_list = [src_list]
        for src in src_list:
            label_dir = None
            label_list = None
            if isinstance(src, str):
                lst = imglist_from_dir_or_flist(src)
            else:
                assert isinstance(src, Dict)
                lst = imglist_from_dir_or_flist(src['sample_list'])
                if 'label_dir' in src:
                    label_dir = src['label_dir']
                if 'label_list' in src:
                    label_list = load_exec_list(src['label_list'])
            if label_dir is not None:
                for imgp in lst:
                    sample_list.append({'img': imgp, 'label_dir': label_dir})
            elif label_list is not None:
                for imgp, labelp in zip(lst, label_list):
                    sample_list.append({'img': imgp, 'label': labelp})
            else:
                for imgp in lst:
                    sample_list.append({'img': imgp})
        self.sample_list = sample_list
        
        self.random_flip = random_flip
        self.random_hsv = random_hsv

        self.random_crop = random_crop
        self.random_crop_ratio = random_crop_ratio

        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

        self.load_depth_cond = load_depth_cond
        self.depth_cond_prob = depth_cond_prob
        self.contrl_signals = contrl_signals
        self.normalize_depth = normalize_depth
        self.to_depth = to_depth
        self.pad_argb = pad_argb

        self.include_blend_cls = include_blend_cls
        self.mix_tag_groups = mix_tag_groups
        self.random_pad_prob = random_pad_prob

        self.ngroup = len(mix_tag_groups)
        self._current_group = 0
        self._current_group_iters = 0
        self.ngroup_iters = ngroup_iters

    def __len__(self):
        return len(self.sample_list)
    

    def get_sample(self, index: int):

        def _aug_size_transform(img, upscale_interpolation=cv2.INTER_LINEAR, down_interpolation=cv2.INTER_AREA):
            if flip_tb:
                img = img[::-1]
            if flip_lr:
                img = img[:, ::-1]
            img = center_square_pad_resize(img, target_size=self.target_size[0], upscale_interpolation=upscale_interpolation, downscale_interpolation=down_interpolation)
            return img

        target_w, target_h = self.target_size

        flip_lr = flip_tb = False
        if self.random_flip > 0 and random.random() < self.random_flip:
            flip_lr = True

        sample = self.sample_list[index]
        imgp = sample['img']
        imgname = osp.join(osp.dirname(imgp), osp.splitext(osp.basename(imgp))[0])
        annp = imgname + '_ann.json'
        ann = json2dict(annp)
        labelp = imgname + '_depth.png'

        fullpage = np.array(Image.open(imgp).convert('RGBA'))
        depth = np.array(Image.open(labelp))

        tag_group_info = self.mix_tag_groups[self._current_group]
        target_tags = tag_group_info['tags'].copy()
        blended_rst = load_blend_parts(imgp, tag_group_info['tags'], ann=ann, keep_loaded_parts=True, keep_depth=True)

        crop_tag_list = tag_group_info.get('crop_tag_list', 
                                           [t for t in target_tags if t not in tag_group_info.get('size_skip_tags', [])])
        tx1, ty1, tx2, ty2 = xyxy_from_taglist(ann, crop_tag_list)

        random_pad = random.random() < self.random_pad_prob
        if random_pad:
            fh, fw = fullpage.shape[:2]
            tmax, lmax = ty1, tx1
            rmax = fw - tx2 - 1
            bmax = fh - ty2 - 1
            blended_img, (t, b, l, r) = random_pad_img(blended_rst['img'][ty1: ty2, tx1: tx2], tmax, bmax, lmax, rmax)
            tx1 -= l
            ty1 -= t
            th, tw = blended_img.shape[:2]
            tx2 = tx1 + tw
            ty2 = ty1 + th
            blended_img = blended_rst['img'][ty1: ty2, tx1: tx2].copy()

        def _np_transform(img):
            h, w = img.shape[:2]
            img = _aug_size_transform(img)
            if self.pad_argb:
                img = pad_rgb(img, return_format='argb')
            else:
                img = np.concat([img[..., 3:], img[..., :3]], axis=2).astype(np.float32) / 255.
            img = img2tensor(img=img, dim_order='chw')
            return img

        fullpage = fullpage[ty1: ty2, tx1: tx2].copy()
        cond_full_page = _np_transform(fullpage)
        # img = _aug_size_transform(fullpage)
        depth = _aug_size_transform(depth[ty1: ty2, tx1: tx2].copy())

        fh = fw = target_h
        empty_tensor = torch.zeros((4, fh, fw), dtype=torch.float32)
        tgt_img_list = [empty_tensor] * len(target_tags)
        tgt_depth_list = [torch.ones((1, fh, fw), dtype=torch.float32)] * len(target_tags)
        caption_list = target_tags
        empty_mask = [True] * len(target_tags)

        # img_tensor = img2tensor(img=img, normalize=True, mean=self.pixel_mean, std=self.pixel_std, dim_order='chw')
        depth_max = np.max(depth) + 1e-6
        depth = (depth_max - depth) / depth_max * 2 - 1
        depth = img2tensor(depth, dim_order='chw')

        for part in blended_rst['part_list']:
            tag_img = _np_transform(part['img'][ty1: ty2, tx1: tx2])
            tag_index = target_tags.index(part['tag'])
            empty_mask[tag_index] = False
            tgt_img_list[tag_index] = tag_img
            tag_depth = _aug_size_transform(part['depth'][ty1: ty2, tx1: tx2])
            tag_depth = (depth_max - tag_depth) / depth_max * 2 - 1
            tag_depth = img2tensor(tag_depth, dim_order='chw')
            tgt_depth_list[tag_index] = tag_depth

        if self.include_blend_cls:
            tgt_depth_list.append(depth)
            empty_mask.append(False)
            tgt_img_list.append(cond_full_page)
            caption_list.append('all')

        batch = {
            'cond_full_page': cond_full_page,
            'img_list': torch.stack(tgt_img_list),
            'depth_list': torch.stack(tgt_depth_list),
            'empty_mask': torch.tensor(empty_mask),
            'srcp': imgp,
            'caption_list': '\n'.join(caption_list),
        }

        self._current_group_iters += 1
        if self._current_group_iters % self.ngroup_iters == 0:
            self._current_group_iters = 0
            self._current_group = (self._current_group + 1) % self.ngroup
        

        # if self.load_depth_cond:
        #     if load_control_map:
        #         # cond_depth = np.array(Image.open(imgname + '_depth.png'))[tag_xyxy[1]: tag_xyxy[3], tag_xyxy[0]: tag_xyxy[2]]
        #         # cond_depth = _aug_size_transform(cond_depth)
        #         tag_mask = rle2mask(masks_ann[self.tag_list.index(target_tag['tag'])])
        #         tag_mask = tag_mask[tag_xyxy[1]: tag_xyxy[3], tag_xyxy[0]: tag_xyxy[2]].astype(np.uint8) * 255
        #         tag_mask = _aug_size_transform(tag_mask).astype(np.float32) / 255.
        #         cond_depth = depth_np.astype(np.float32) * tag_mask / depth_max
        #         control_input = torch.from_numpy(np.stack([cond_depth, tag_mask], axis=0))
        #     else:
        #         control_input = torch.zeros((2, img.shape[-2], img.shape[-1]), dtype=torch.float32)
        #     batch['control_input'] = control_input
        
        return batch

        

    def __getitem__(self, index):

        while True:
            try:
                samplep = self.sample_list[index]['img']
                return self.get_sample(index)
            except:
                raise
                print(f'Failed to load {samplep}: ')
                print(traceback.format_exc())
                index = random.randint(0, len(self) - 1)
                continue

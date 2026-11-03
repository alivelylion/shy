import os.path as osp
from typing import Dict, List, Union, Tuple, Optional
import random
import traceback

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
from einops import rearrange

from utils.cv import rle2mask, pad_rgb, random_hsv
from utils.torch_utils import img2tensor, tensor2img
from utils.io_utils import json2dict, load_exec_list, find_all_imgs, imglist_from_dir_or_flist


MINIMUM_VISIBLE_ALPHA = 25


class SemanticSegDataset(Dataset):

    '''
    valid src_list: filepath, dict(sample_list=filepath), dict(sample_list=filepath, label_dir=...), or a list of these
    '''

    def __init__(
        self, 
        src_list: Union[str, Dict, List],
        class_num: int,
        skip_labels = None,
        target_size: Union[int, Tuple] = 1024,
        random_downscale=0.,
        downscale_sizes=(48, 64, 128, 256),
        random_flip=0.,
        random_crop=0.,
        random_crop_ratio=0.3,
        random_hsv=0.,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],    # SAM input mean
        pixel_std: List[float] = [58.395, 57.12, 57.375],       # SAM input std
        tag_list: List[str] = None,
        **kwargs) -> None:

        super().__init__()

        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.class_num = class_num

        if skip_labels is not None:
            assert isinstance(skip_labels, (int, tuple, list, set))
            if isinstance(skip_labels, int):
                skip_labels = [skip_labels]
        self.skip_labels = skip_labels
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

        self.random_downscale = random_downscale
        self.downscale_sizes = [(dsz, dsz) if isinstance(dsz, int) else dsz 
                                for dsz in downscale_sizes]
        
        self.random_flip = random_flip
        self.random_hsv = random_hsv

        self.random_crop = random_crop
        self.random_crop_ratio = random_crop_ratio

        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

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

        target_w, target_h = self.target_size

        flip_lr = flip_tb = False
        if self.random_flip > 0 and random.random() < self.random_flip:
            if random.random() < 0.3:
                flip_tb = True
            if random.random() < 0.8:
                flip_lr = True

        sample = self.sample_list[index]
        imgp = sample['img']
        if 'label_dir' in sample:
            labelp = osp.join(sample['label_dir'], osp.splitext(osp.basename(imgp))[0] + '.png')
        else:
            labelp = osp.join(
                osp.dirname(imgp),
                osp.splitext(osp.basename(imgp))[0] + '_faceseg.png'
            )

        crop_xyxy = None
        pad_x = pad_y = 0

        mask_valid_list = None
        if not osp.exists(labelp):
            labelp = osp.join(
                osp.dirname(imgp),
                osp.splitext(osp.basename(imgp))[0] + '.json'
            )
            if not osp.exists(labelp):
                labelp = labelp + '.gz'
            if osp.exists(labelp):
                masks_ann = json2dict(labelp)
                mask_valid_list = []
                for m in masks_ann:
                    if 'is_valid' in m:
                        mask_valid_list.append(m['is_valid'])
                    else:
                        mask_valid_list.append(True)
                masks = [rle2mask(m, to_bool=False) for m in masks_ann]
                masks = np.stack(masks, axis=2)

            mh, mw = masks.shape[:2]
            if mh > mw:
                pad_x = mh - mw
            elif mh < mw:
                pad_y = mw - mh
            if pad_x != 0 or pad_y != 0:
                masks = cv2.copyMakeBorder(masks, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value = 0)
            
            input_wh = (masks.shape[1], masks.shape[0])
            crop_xyxy = self.get_randomcrop_xyxy(input_wh)
            if crop_xyxy is not None:
                masks = masks[crop_xyxy[1]: crop_xyxy[3], crop_xyxy[0]: crop_xyxy[2]]
            if masks.shape[1] != target_w or masks.shape[0] != target_h:
                masks = cv2.resize(masks, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            if flip_tb:
                masks = masks[::-1]
            if flip_lr:
                masks = masks[:, ::-1]
            masks = rearrange(masks, 'h w c -> c h w')
            if len(masks) == self.class_num - 1:
                masks = np.concat(
                    [np.logical_or.reduce(masks, 0, keepdims=True), masks]
                )
            masks = np.ascontiguousarray(masks)
        else:
            labels = np.array(Image.open(labelp).convert('L'))
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

            masks = labels[None]
            masks = masks == np.arange(0, self.class_num).reshape((-1, 1, 1)) # C H W

        if self.skip_labels is not None:
            _masks = []
            for ii in range(self.class_num):
                if ii in self.skip_labels:
                    continue
                _masks.append(masks[ii])
            masks = np.stack(_masks)

        img = Image.open(imgp).convert('RGB')
        img = np.array(img)
        if pad_x != 0 or pad_y != 0:
            img = cv2.copyMakeBorder(img, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value = 0)

        if crop_xyxy is not None:
            img = img[crop_xyxy[1]: crop_xyxy[3], crop_xyxy[0]: crop_xyxy[2]]

        if self.random_downscale > 0 and random.random() < self.random_downscale:
            dsz = random.choice(self.downscale_sizes)
            if dsz[0] < img.shape[1] or dsz[1] < img.shape[0]:
                img = cv2.resize(img, dsz, interpolation=cv2.INTER_AREA)

        if target_w != img.shape[1] or target_h != img.shape[0]:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)

        if self.random_hsv > 0 and random.random() < self.random_hsv:
            img = random_hsv(img)

        if flip_tb:
            img = img[::-1]
        if flip_lr:
            img = img[:, ::-1]

        img = img2tensor(img=img, normalize=True, mean=self.pixel_mean, std=self.pixel_std, dim_order='chw')
        masks = torch.from_numpy(masks).to(dtype=torch.float32)
        if mask_valid_list is None:
            mask_valid_list = [True] * self.class_num
        
        mask_valid_list = torch.from_numpy(np.array(mask_valid_list, dtype=np.float32))
        return {
            'img': img,
            'masks': masks,
            'srcp': imgp,
            'mask_valid_list': mask_valid_list
        }
        

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


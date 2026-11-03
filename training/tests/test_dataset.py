

import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import numpy as np
from pathlib import Path
from  tqdm import tqdm
import click
import cv2
import torch
from PIL import Image
from utils.torch_utils import seed_everything

from utils.io_utils import find_all_files_recursive, pil_pad_square, pil_ensure_rgb, imglist2imgrid
from utils.visualize import imglist2imgrid_with_tags


@click.group()
def cli():
    """data set test scripts.
    """


@cli.command('test_segs')
@click.option('--config')
@click.option('--save_dir', default=None)
@click.option('--shuffle', is_flag=True, default=False)
def test_segs(config, save_dir, shuffle):

    from train.dataset_seg import SemanticSegDataset
    from utils.torch_utils import img2tensor, tensor2img
    seed_everything(1)

    if save_dir is None:
        save_dir = osp.join(osp.dirname(config), 
                            osp.splitext(osp.basename(config))[0] + '_datasetvis')
    os.makedirs(save_dir, exist_ok=True)

    from omegaconf import OmegaConf
    config = OmegaConf.load(config)
    

    dataset = SemanticSegDataset(**OmegaConf.to_container(config.dataset_args))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config.get('batch_size', 1),
        num_workers=config.get('dataloader_num_workers', 0),
        drop_last=True
    )

    for batch in dataloader:
        img = batch['img'][0]
        masks = batch['masks'][0]
        srcp = batch['srcp'][0]
        img_np = tensor2img(img, mean=dataset.pixel_mean, std=dataset.pixel_std, denormalize=True)
        masks_np = masks.to(device='cpu', dtype=torch.bool).numpy()
        depth_np = tensor2img(depth, mean=0, std=255., denormalize=True)

        pass
        # rst = visualize_face_segs(masks_np, src_img=img_np)
        # Image.fromarray(rst).save(osp.join(save_dir, osp.splitext(osp.basename(srcp))[0]) + '.jpg', q=95)

        pass


@cli.command('test_depth')
@click.option('--config')
@click.option('--save_dir', default=None)
@click.option('--shuffle', is_flag=True, default=False)
def test_depth(config, save_dir, shuffle):

    from train.dataset_depth import DepthDataset
    from utils.torch_utils import img2tensor, tensor2img

    seed_everything(1)

    if save_dir is None:
        save_dir = osp.join(osp.dirname(config), 
                            osp.splitext(osp.basename(config))[0] + '_datasetvis')
    os.makedirs(save_dir, exist_ok=True)

    from omegaconf import OmegaConf
    config = OmegaConf.load(config)
    
    shuffle = True
    dataset = DepthDataset(**OmegaConf.to_container(config.dataset_args))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config.get('batch_size', 1),
        num_workers=config.get('dataloader_num_workers', 0),
        drop_last=True
    )

    for batch in dataloader:
        img = batch['img'][0]
        masks = batch['masks'][0]
        depth = batch['depth'][0]
        srcp = batch['srcp'][0]
        img_np = tensor2img(img, mean=dataset.pixel_mean, std=dataset.pixel_std, denormalize=True)
        masks_np = masks.to(device='cpu', dtype=torch.bool).numpy()[0]
        masks_np = masks_np.astype(np.uint8) * 255
        masks_np = cv2.cvtColor(masks_np, cv2.COLOR_GRAY2RGB)
        
        depth = depth.to(device='cpu', dtype=torch.float32).numpy()[0]
        depth = (depth * 255).astype(np.uint8)
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)

        vis_list = [img_np, masks_np, depth]

        if 'control_input' in batch:
            tag_mask = batch['control_input'][0][1].to(device='cpu', dtype=torch.float32).numpy()
            tag_mask = (tag_mask * 255).astype(np.uint8)
            tag_mask = cv2.cvtColor(tag_mask, cv2.COLOR_GRAY2RGB)
            tag_depth = batch['control_input'][0][0].to(device='cpu', dtype=torch.float32).numpy()
            tag_depth = (tag_depth * 255).astype(np.uint8)
            tag_depth = cv2.cvtColor(tag_depth, cv2.COLOR_GRAY2RGB)
            vis_list.append(tag_mask)
            vis_list.append(tag_depth)

        rst = imglist2imgrid(vis_list, cols=len(vis_list))
        # rst = visualize_face_segs(masks_np, src_img=img_np)
        Image.fromarray(rst).save(osp.join(save_dir, osp.splitext(osp.basename(srcp))[0]) + '.jpg', q=95)

        pass



@cli.command('test_depth3d')
@click.option('--config')
@click.option('--save_dir', default=None)
@click.option('--shuffle', is_flag=True, default=False)
def test_depth3d(config, save_dir, shuffle):

    from train.dataset_depth import DepthDataset3D
    from utils.cv import argb2rgba, img_alpha_blending
    from utils.torch_utils import img2tensor, tensor2img

    seed_everything(1)

    if save_dir is None:
        save_dir = osp.join(osp.dirname(config), 
                            osp.splitext(osp.basename(config))[0] + '_datasetvis')
    os.makedirs(save_dir, exist_ok=True)

    from omegaconf import OmegaConf
    config = OmegaConf.load(config)
    
    shuffle = True
    dataset = DepthDataset3D(**OmegaConf.to_container(config.dataset_args))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config.get('batch_size', 1),
        num_workers=config.get('dataloader_num_workers', 0),
        drop_last=True
    )

    for batch in dataloader:
        img_list = batch['img_list'][0]
        depth_list = batch['depth_list'][0]
        srcp = batch['srcp'][0]
        print(img_list.min(), img_list.max())

        caption_list = batch['caption_list'][0].split('\n')
        img_vis = imglist2imgrid_with_tags([tensor2img(img[1:], denormalize=True) for img in img_list], caption_list, skip_empty=True, fix_size=512)
        depth_vis = imglist2imgrid_with_tags([cv2.cvtColor(tensor2img(img, denormalize=True, mean=127.5, std=127.5), cv2.COLOR_GRAY2RGB) for img in depth_list], caption_list, skip_empty=True, fix_size=512)
        img_vis = np.concat([img_vis, depth_vis])

        blended = img_alpha_blending([{'img': argb2rgba(tensor2img(img, denormalize=True)), 'depth': depth[0].to(device='cpu', dtype=torch.float32).numpy()} for img, depth in zip(img_list, depth_list)])
        # print(caption_list)
        Image.fromarray(blended).save('local_tst.png')
        # Image.fromarray(img_vis).save('local_tst.png')
        # pass
        # depth = batch['depth'][0]
        # srcp = batch['srcp'][0]
        # img_np = tensor2img(img, mean=dataset.pixel_mean, std=dataset.pixel_std, denormalize=True)
        # masks_np = masks.to(device='cpu', dtype=torch.bool).numpy()[0]
        # masks_np = masks_np.astype(np.uint8) * 255
        # masks_np = cv2.cvtColor(masks_np, cv2.COLOR_GRAY2RGB)
        
        # depth = depth.to(device='cpu', dtype=torch.float32).numpy()[0]
        # depth = (depth * 255).astype(np.uint8)
        # depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)

        # vis_list = [img_np, masks_np, depth]

        # rst = imglist2imgrid(vis_list, cols=len(vis_list))
        # # rst = visualize_face_segs(masks_np, src_img=img_np)
        Image.fromarray(img_vis).save(osp.join(save_dir, osp.splitext(osp.basename(srcp))[0]) + '.jpg', q=95)
        # Image.fromarray(img_vis).save('local_tst.jpg', q=95)
        pass



@cli.command('test_layerdiff')
@click.option('--config')
@click.option('--save_dir', default=None)
@click.option('--shuffle', is_flag=True, default=False)
def test_layerdiff(config, save_dir, shuffle):

    from train.dataset_layerdiff import LayerDiffDataset
    from utils.torch_utils import img2tensor, tensor2img
    
    seed_everything(1)

    if save_dir is None:
        save_dir = osp.join(osp.dirname(config), 
                            osp.splitext(osp.basename(config))[0] + '_datasetvis')
    os.makedirs(save_dir, exist_ok=True)

    from omegaconf import OmegaConf
    config = OmegaConf.load(config)
    
    shuffle = True
    dataset = LayerDiffDataset(**OmegaConf.to_container(config.dataset_args))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config.get('batch_size', 1),
        num_workers=config.get('dataloader_num_workers', 0),
        drop_last=True
    )

    def _cvt_img(img):
        img_np = tensor2img(img, mean=dataset.pixel_mean, std=dataset.pixel_std, denormalize=True)
        img_np = np.concat([img_np[..., 1:], img_np[..., [0]]], axis=2)
        return Image.fromarray(img_np)

    for batch in dataloader:
        img = batch['img'][0]
        cond_fullpage = batch['cond_fullpage'][0]
        srcp = batch['srcp'][0]
        caption = batch['caption'][0]
        savename = osp.join(save_dir, osp.splitext(osp.basename(srcp))[0])
                            
        _cvt_img(img).save(savename + '_img.png')
        _cvt_img(cond_fullpage).save(savename + '_cond_fullpage.png')
        if 'cond_tag_img' in batch:
            cond_tag_img = batch['cond_tag_img'][0]
            _cvt_img(cond_tag_img).save(savename + '_cond_tag_img.png')
        with open(savename + '.txt', 'w', encoding='utf8') as f:
            f.write(caption)
        pass




@cli.command('test_layerdiff3d')
@click.option('--config')
@click.option('--save_dir', default=None)
@click.option('--shuffle', is_flag=True, default=False)
def test_layerdiff3d(config, save_dir, shuffle):

    from train.dataset_layerdiff import LayerDiffDataset3D
    from utils.torch_utils import img2tensor, tensor2img
    from utils.visualize import pil_draw_text
    from utils.cv import visualize_rgba
    from tqdm import tqdm
    from utils.torchcv import pad_rgb_torch

    from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline, UNet2DConditionModel
    from modules.layerdiffuse.vae import vae_encode, TransparentVAEEncoder, TransparentVAEDecoder
    from modules.layerdiffuse.utils import patch_transvae_sd
    from utils.torch_utils import init_model_from_pretrained, tensor2img, img2tensor, seed_everything
    from diffusers import AutoencoderKL


    device='cuda'

    trans_vae_encoder: TransparentVAEEncoder = init_model_from_pretrained(
        'lllyasviel/LayerDiffuse_Diffusers',
        TransparentVAEEncoder, weights_name='ld_diffusers_sdxl_vae_transparent_encoder.safetensors',
        patch_state_dict_func=patch_transvae_sd
    ).to(device=device, dtype=torch.float32)

    trans_vae_decoder: TransparentVAEDecoder = init_model_from_pretrained(
        'lllyasviel/LayerDiffuse_Diffusers',
        TransparentVAEDecoder, weights_name='ld_diffusers_sdxl_vae_transparent_decoder.safetensors',
        patch_state_dict_func=patch_transvae_sd
    ).to(device=device, dtype=torch.float32)

    vae = AutoencoderKL.from_pretrained('cagliostrolab/animagine-xl-4.0', subfolder='vae').to(device=device, dtype=torch.float32)
    
    seed_everything(1)

    if save_dir is None:
        save_dir = osp.join(osp.dirname(config), 
                            osp.splitext(osp.basename(config))[0] + '_datasetvis')
    os.makedirs(save_dir, exist_ok=True)

    from omegaconf import OmegaConf
    config = OmegaConf.load(config)
    
    shuffle = True
    dataset = LayerDiffDataset3D(**OmegaConf.to_container(config.dataset_args))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config.get('batch_size', 1),
        num_workers=config.get('dataloader_num_workers', 0),
        drop_last=True
    )

    def _cvt_img(img):
        img_np = tensor2img(img, mean=dataset.pixel_mean, std=dataset.pixel_std, denormalize=True)
        img_np[..., [0]] = 255
        img_np = np.concat([img_np[..., 1:], img_np[..., [0]]], axis=2)
        return Image.fromarray(visualize_rgba(img_np))

    for batch in dataloader:
        img_list = batch['img_list'][0]
        caption_list = batch['caption_list'][0].split('\n')
        cond_fullpage = batch['cond_fullpage'][0]
        empty_mask = batch['empty_mask'][0]

        if not dataset.pad_argb:
            img_list = pad_rgb_torch(img_list.to(device='cuda'), return_format='argb')
        # vis_img_list = [_cvt_img(cond_fullpage)] + [_cvt_img(img) for jj, img in enumerate(img_list) if not empty_mask[jj]]

        vis_img_list = [_cvt_img(cond_fullpage)]
        for jj, img in enumerate(tqdm(img_list)):
            if empty_mask[jj]:
                continue
            img = img[None].to(device=device, dtype=torch.float32)
            latent = vae_encode(vae, trans_vae_encoder, img, use_offset=False)

            # latents = latents.to(dtype=vae.dtype, device=vae.device) / 
            result_list, vis_list_batch = trans_vae_decoder(vae, latent / vae.config.scaling_factor)
            vis_img_list.append(vis_list_batch[0])

        caption_list = ['condition page'] + [c for jj, c in enumerate(caption_list) if not empty_mask[jj]]
        imglist2imgrid_with_tags(vis_img_list, caption_list, skip_empty=True, fix_size=512, output_type='pil').save('local_tst.png')
        pass
        # srcp = batch['srcp'][0]
        # caption = batch['caption'][0]
        # savename = osp.join(save_dir, osp.splitext(osp.basename(srcp))[0])
                            
        # _cvt_img(img).save(savename + '_img.png')
        # _cvt_img(cond_fullpage).save(savename + '_cond_fullpage.png')
        # if 'cond_tag_img' in batch:
        #     cond_tag_img = batch['cond_tag_img'][0]
        #     _cvt_img(cond_tag_img).save(savename + '_cond_tag_img.png')
        # with open(savename + '.txt', 'w', encoding='utf8') as f:
        #     f.write(caption)
        # pass


if __name__ == '__main__':
    cli()
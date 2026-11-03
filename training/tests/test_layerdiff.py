
from numpy._typing._array_like import NDArray
import os
import os.path as osp
import sys
from typing import Any
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from omegaconf import OmegaConf
import numpy as np
from pathlib import Path
from  tqdm import tqdm
import click
import cv2
import torch
from omegaconf import OmegaConf
from PIL import Image
from utils.torch_utils import seed_everything
from utils.cv import checkerboard, checkerboard_vis, batch_load_masks, pad_rgb
from utils.io_utils import find_all_imgs, pil_pad_square, pil_ensure_rgb, imglist2imgrid, json2dict, dict2json


@click.group()
def cli():
    """layer diff test scripts.
    """


@cli.command('test_layerdiff')
@click.option('--config')
@click.option('--rank_to_worldsize', default=None)
def test_layerdiff(config, rank_to_worldsize):

    import random

    from safetensors.torch import load_file
    from utils.cv import visualize_rgba, center_square_pad_resize
    from utils.torch_utils import init_model_from_pretrained, tensor2img, seed_everything
    from modules.layerdiffuse.vae import TransparentVAEDecoder, TransparentVAEEncoder, vae_encode
    from modules.layerdiffuse.utils import patch_transvae_sd, patch_unet_convin
    from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
    from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
    from live2d.scrap_model import VALID_BODY_PARTS_V2
    from utils.io_utils import load_exec_list

    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps, rescale_noise_cfg
    from train.dataset_layerdiff import LayerDiffDataset
    from utils.torch_utils import img2tensor, tensor2img
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
    from diffusers import (
        AutoencoderKL,
        EulerDiscreteScheduler,
        UNet2DConditionModel,
    )

    config = OmegaConf.load(config)
    if rank_to_worldsize is not None:
        config.rank_to_worldsize = rank_to_worldsize
    
    seed_list = config.get('seed_list', None)
    device = 'cuda'
    dtype = torch.bfloat16

    args = config
    save_dir = config.save_dir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    seed = config.get('seed', 1)
    
    seed_everything(seed)
    is_3d = args.get('is_3d', False)
    pretrained_model_name_or_path = args.pretrained_model_name_or_path
    
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        revision=None,
        variant=None,
    )
    scheduler = EulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    trans_vae_encoder: TransparentVAEEncoder = init_model_from_pretrained(
        'lllyasviel/LayerDiffuse_Diffusers',
        TransparentVAEEncoder, weights_name='ld_diffusers_sdxl_vae_transparent_encoder.safetensors',
        patch_state_dict_func=patch_transvae_sd
    )

    trans_vae_decoder: TransparentVAEDecoder = init_model_from_pretrained(
        'lllyasviel/LayerDiffuse_Diffusers',
        TransparentVAEDecoder, weights_name='ld_diffusers_sdxl_vae_transparent_decoder.safetensors',
        patch_state_dict_func=patch_transvae_sd
    )

    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        use_fast=False,
    )

    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )

    if config.vae_ckpt:
        sd = load_file(config.vae_ckpt)
        td_sd = {}
        vae_sd = {}
        for k, v in sd.items():
            if k.startswith('trans_decoder.'):
                td_sd[k.lstrip('trans_decoder.')] = v
            elif k.startswith('vae.'):
                vae_sd[k.replace('vae.', '')] = v
        trans_vae_decoder.load_state_dict(td_sd)
        vae.load_state_dict(vae_sd)
        vae = vae.to(device=device, dtype=torch.float32)
        trans_vae_encoder = trans_vae_encoder.to(device=device, dtype=torch.float32)
        trans_vae_decoder = trans_vae_decoder.to(device=device, dtype=torch.float32)
    else:
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

        vae = AutoencoderKL.from_pretrained(
            "cagliostrolab/animagine-xl-4.0",
            subfolder="vae").to(device=device, dtype=torch.float32)

    unet_cls = UNetFrameConditionModel if is_3d else UNet2DConditionModel
    unet = unet_cls.from_pretrained(
        **OmegaConf.to_container(config.pretrained_unet)
    )
    unet = unet.to(device=device, dtype=dtype)
    unet.eval()

    weight_dtype = unet.dtype
    rng = torch.Generator(device=unet.device).manual_seed(seed)
    
    pipeline: KDiffusionStableDiffusionXLPipeline = KDiffusionStableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        unet=unet,
        scheduler=scheduler
    )

    target_tag_list = VALID_BODY_PARTS_V2

    exec_list = []
    rank_to_worldsize = config.get('rank_to_worldsize', None)
    create_sub_dir = config.get('create_sub_dir', False)

    if isinstance(config.exec_list, str):  
        if osp.isdir(config.exec_list):
            exec_list = find_all_imgs(config.exec_list, abs_path=True)
        else:
            exec_list = load_exec_list(config.exec_list, rank_to_worldsize=rank_to_worldsize)
    else:
        for p in config.exec_list:
            exec_list += load_exec_list(p, rank_to_worldsize=rank_to_worldsize)
    if config.shuffle:
        random.shuffle(exec_list)

    mask_cond = config.get('mask_cond', False)

    if seed_list is not None:
        num_data = len(exec_list)
        exec_list = exec_list * len(seed_list)

    for ii, p in enumerate(tqdm(exec_list)):
        if mask_cond:
            mask_list = batch_load_masks(p + '_masks.json')
        if create_sub_dir:
            # saved = osp.join(save_dir, osp.basename(osp.dirname(srcp)), srcname)
            saved = osp.join(save_dir, osp.basename(osp.dirname(p)), osp.splitext(osp.basename(p))[0])
        else:
            saved = osp.join(save_dir, osp.splitext(osp.basename(p))[0])
        current_seed = None
        if seed_list is not None:
            current_seed = seed_list[ii // num_data]
            saved += f'_seed{current_seed}'
            seed_everything(current_seed)
        os.makedirs(saved, exist_ok=True)
        fullpage = center_square_pad_resize(np.array(Image.open(p).convert('RGBA')), 1024)
        Image.fromarray(fullpage).save(osp.join(saved, 'src_img.png'))
        page_alpha = img2tensor(fullpage[..., -1] / 255., device=vae.device, dtype=vae.dtype)[0][..., None]
        fullpage = fullpage[..., :3]
        c_concat = np.concatenate([np.full_like(fullpage[..., :1], fill_value=255), fullpage], axis=2)
        c_concat = img2tensor(c_concat, normalize=True)
        if trans_vae_decoder.model.config.in_channels == 6:
            rgb_cond = c_concat[:, 1:].to(device=vae.device, dtype=vae.dtype)
        else:
            rgb_cond = None
        empty_tensor = torch.zeros_like(c_concat)
        c_concat = vae_encode(vae, trans_vae_encoder, c_concat, use_offset=False).to(device=device, dtype=dtype)

        if unet.config.in_channels == 12:
            c_concat = torch.cat([c_concat, vae_encode(vae, trans_vae_encoder, empty_tensor).to(device=device, dtype=dtype)], dim=1)

        vis_list = [fullpage]
        # target_tag_list = ['hair']

        res_list = []
        for tag in target_tag_list:
            if is_3d:
                tag = target_tag_list

            if mask_cond:
                visible_mask = mask_list[VALID_BODY_PARTS_V2.index(tag)].astype(np.uint8) * 255
                visible_mask = center_square_pad_resize(visible_mask, 1024)
                visible_mask = visible_mask[..., None]
                
                cond_tag_img = fullpage * (visible_mask.astype(np.float32) / 255.)
                # cond_tag_img[..., [-1]] = visible_mask.astype(np.float32)
                cond_tag_img = np.concatenate([cond_tag_img, visible_mask], axis=2)
                cond_tag_img: NDArray[Any] = np.concatenate([visible_mask, (pad_rgb(cond_tag_img) * 255).astype(np.uint8)], axis=2)
                cond_tag_img = img2tensor(img=cond_tag_img, normalize=True, mean=[0., 0., 0., 0.], std=[255., 255., 255., 255.], dim_order='chw')
                c_concat[:, -4:] = vae_encode(vae, trans_vae_encoder, cond_tag_img[None].to(dtype=vae.dtype, device=vae.device), use_offset=False).to(dtype=weight_dtype)


            latents = pipeline(
                strength=1.0,
                num_inference_steps=config.num_inference_steps,
                batch_size=1,
                generator=rng,
                guidance_scale=args.guidance_scale, c_concat=c_concat,
                prompt=tag,
                negative_prompt=''
            ).images
            if latents.ndim == 5:
                latents = latents[0]

            latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor

            for latent in latents:
                latent = latent[None]
                # latent = scheduler.add_noise(latent, torch.randn_like(latent), timesteps=torch.tensor([1], device=latent.device))
                result_list, vis_list_batch = trans_vae_decoder(vae, latent, rgb_cond=rgb_cond, mask=page_alpha)
                vis_list += vis_list_batch
                res_list += result_list

            if is_3d:
                break

        for rst, tag in zip(res_list, target_tag_list):
            savename = osp.join(saved, f'{tag}.png')
            Image.fromarray(rst).save(savename)


@cli.command('test_layerdiff_vae')
@click.option('--config')
@click.option('--save_dir', default=None)
@click.option('--shuffle', is_flag=True, default=False)
def test_layerdiff_vae(config, save_dir, shuffle):

    from train.dataset_layerdiff import LayerDiffDataset
    from utils.torch_utils import img2tensor, tensor2img
    from utils.torch_utils import init_model_from_pretrained, tensor2img, seed_everything
    from modules.layerdiffuse.vae import TransparentVAEDecoder, TransparentVAEEncoder, vae_encode
    from modules.layerdiffuse.utils import patch_unet_convin, patch_transvae_sd
    from diffusers import AutoencoderKL
    
    seed_everything(1)

    if save_dir is None:
        save_dir = osp.join(osp.dirname(config), 
                            osp.splitext(osp.basename(config))[0] + '_vaedecodevis')
    os.makedirs(save_dir, exist_ok=True)

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

    vae = AutoencoderKL.from_pretrained(
        'cagliostrolab/animagine-xl-4.0',
        subfolder="vae",
    )
    trans_vae_encoder: TransparentVAEEncoder = init_model_from_pretrained(
        'lllyasviel/LayerDiffuse_Diffusers',
        TransparentVAEEncoder, weights_name='ld_diffusers_sdxl_vae_transparent_encoder.safetensors',
        patch_state_dict_func=patch_transvae_sd
    )

    trans_vae_decoder: TransparentVAEDecoder = init_model_from_pretrained(
        'lllyasviel/LayerDiffuse_Diffusers',
        TransparentVAEDecoder, weights_name='ld_diffusers_sdxl_vae_transparent_decoder.safetensors',
        patch_state_dict_func=patch_transvae_sd
    )
    device = 'cuda'

    vae.requires_grad_(False)
    trans_vae_encoder.requires_grad_(False)
    trans_vae_decoder.requires_grad_(False)
    vae.to(device, dtype=torch.float32)
    trans_vae_decoder.to(device, dtype=torch.float32)
    trans_vae_encoder.to(device, dtype=torch.float32)

    def _cvt_img(img):
        img_np = tensor2img(img, mean=dataset.pixel_mean, std=dataset.pixel_std, denormalize=True)
        img_np = np.concatenate([img_np[..., 1:], img_np[..., [0, 0, 0]]])
        return Image.fromarray(img_np)

    for batch in dataloader:
        img = batch['img'][0]
        cond_fullpage = batch['cond_fullpage'][0]
        cond_tag_img = batch['cond_tag_img'][0]
        srcp = batch['srcp'][0]
        caption = batch['caption'][0]
        savename = osp.join(save_dir, osp.splitext(osp.basename(srcp))[0])
                            
        latent_scaled = vae_encode(vae, trans_vae_encoder, img[None].to(device=device, dtype=vae.dtype), use_offset=True)
        latents = latent_scaled.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
        result_list, vis_list_batch = trans_vae_decoder(vae, latents)
        # trans_vae_decoder()
        Image.fromarray(vis_list_batch[0]).save(savename + '_vaedecoded.png')
        _cvt_img(img).save(savename + '_img.png')
        _cvt_img(cond_fullpage).save(savename + '_cond_fullpage.png')
        _cvt_img(cond_tag_img).save(savename + '_cond_tag_img.png')
        
        with open(savename + '.txt', 'w', encoding='utf8') as f:
            f.write(caption)


@cli.command('marigold_blend')
@click.option('--config')
@click.option('--rank_to_worldsize', default=None)
def marigold_blend(config, rank_to_worldsize):
    from modules.marigold import MarigoldDepthPipeline
    from modules.marigold.marigold_depth_pipeline import encode_empty_text, encode_argb_list
    from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
    from live2d.scrap_model import VALID_BODY_PARTS_V2
    from utils.cv import img_alpha_blending, center_square_pad_resize, argb2rgba
    from utils.torch_utils import img2tensor, tensor2img
    from utils.io_utils import load_exec_list
    # model = MarigoldDepthPipeline.from
    config = OmegaConf.load(config)

    if rank_to_worldsize is not None:
        config.rank_to_worldsize = rank_to_worldsize

    device = 'cuda'
    dtype = torch.bfloat16
    unet = UNetFrameConditionModel.from_pretrained(config.depth_unet)
    pipe = MarigoldDepthPipeline.from_pretrained('prs-eth/marigold-depth-v1-1', unet=unet)
    pipe.empty_text_embed = encode_empty_text()
    pipe.to(device=device, dtype=dtype)

    # srcp = 'workspace/datasets/live2d_bodysamples_chunk3/sekai_jxl_crop____08shizuku_archery_t03-NONE-8.png'
    # srcp = 'workspace/datasets/live2d_bodysamples_chunk3/ARUZ_jxl_crop____aierdeliqi_4-NONE-5.png'
    # part_dir = 'local_configs/layerdiff3d_womix20k_vaewgan1024_10k'
    save_dir = config.save_dir
    exec_list = config.exec_list

    rank_to_worldsize = config.get('rank_to_worldsize', None)
    create_sub_dir = config.get('create_sub_dir', False)

    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)

    seed_list = config.get('seed_list', None)
    if seed_list is not None:
        nlist = []
        for p in exec_list:
            nlist += [p] * len(seed_list)
        exec_list = nlist

    for i, srcp in enumerate(tqdm(exec_list)):
        srcname = osp.basename(osp.splitext(srcp)[0])
        img_list = []
        vis_list = []
        caption_list = []
        exist_list = []
        empty_array = np.zeros((1024, 1024, 4), dtype=np.uint8)
        blended_alpha = np.zeros((1024, 1024), dtype=np.float32)
        fullpage = fullpage = center_square_pad_resize(np.array(Image.open(srcp).convert('RGBA')), 1024)
        
        if create_sub_dir:
            saved = osp.join(save_dir, osp.basename(osp.dirname(srcp)), srcname)
        else:
            saved = osp.join(save_dir, srcname)        
            if seed_list is not None:
                saved = saved + '_seed'+str(seed_list[i % len(seed_list)])
            

        for tag in VALID_BODY_PARTS_V2:
            tagp = osp.join(saved, f'{tag}.png')
            if osp.exists(tagp):
                exist_list.append(True)
                caption_list.append(tag)
                tag_arr = np.array(Image.open(tagp))
                tag_arr[..., -1][tag_arr[..., -1] < 15] = 0
                blended_alpha += tag_arr[..., -1].astype(np.float32) / 255
                img_list.append(tag_arr)
            else:
                img_list.append(empty_array)
                exist_list.append(False)
        blended_alpha = np.clip(blended_alpha, 0, 1) * 255
        blended_alpha = blended_alpha.astype(np.uint8)
        fullpage[..., -1] = blended_alpha
        img_list.append(fullpage)

        vae = pipe.vae

        def _np_transform(img):
            img = np.concatenate([img[..., 3:], img[..., :3]], axis=2).astype(np.float32) / 255.
            img = img2tensor(img=img, dim_order='chw')
            return img
        
        img_list_tensor = torch.stack([_np_transform(img) for img in img_list])
        cond_full_page = img_list_tensor[-1][None]
        ncls = img_list_tensor.shape[0]
        with torch.no_grad():
            rgb_latent = [encode_argb_list(vae, img[None, None].to(device=device, dtype=dtype), pad_argb=True, dtype=dtype) for img in img_list_tensor]
            rgb_latent = torch.cat(rgb_latent, dim=1)
            rgb_cond_latent = encode_argb_list(vae, cond_full_page[None], pad_argb=True, dtype=dtype)
            rgb_latent = torch.cat(
                [rgb_cond_latent.expand(-1, ncls, -1, -1, -1), rgb_latent], dim=2
            )

        pipe_out = pipe(
            # tensor2img(img, 'pil', denormalize=True, mean=127.5, std=127.5),
            color_map=None,
            cond_latent=rgb_latent[0],
        )
        depth_pred: np.ndarray = pipe_out.depth_tensor
        
        depth_pred = depth_pred.to(device='cpu', dtype=torch.float32).numpy()
        drawables = [{'img': argb2rgba(tensor2img(img, denormalize=True)), 'depth': depth} for img, depth in zip(img_list_tensor, depth_pred)]
        drawables = drawables[:-1]
        blended = img_alpha_blending(drawables)
        # vis_list.append(checkerboard_vis(img=blended))

        infop = osp.join(saved, 'info.json')
        if osp.exists(infop):
            info = json2dict(infop)
        else:
            info = {'parts': {}}

        parts = info['parts']
        for ii, depth in enumerate(depth_pred[:-1]):
            depth_max, depth_min = depth.max(), depth.min()
            depth = np.clip((depth - depth_min) / (depth_max - depth_min + 1e-7) * 255, 0, 255).astype(np.uint8)
            # depth = depth[..., None][..., [-1] * 3].copy()
            tag = VALID_BODY_PARTS_V2[ii]
            parts_info = parts.get(tag, {})
            savep = osp.join(saved, f'{tag}_depth.png')
            Image.fromarray(depth).save(savep)
            vis_list.append(depth)
            parts[tag] = parts_info
            parts_info['depth_max'] = depth_max
            parts_info['depth_min'] = depth_min

        dict2json(info, infop)
        Image.fromarray(blended).save(osp.join(saved, 'reconstruction.png'))

        # for tagimg in img_list[:-1]:
        #     vis_list.append(checkerboard_vis(tagimg))
        # caption_list.insert(0, 'reconstruction')
        # caption_list.insert(0, 'input')
        # vis_list.insert(0, fullpage[..., :3])

        # imglist2imgrid_with_tags(vis_list, caption_list, fix_size=512, output_type='pil').save(osp.join(save_dir, osp.basename(f'{srcname}.png')))
        # # blended = img_alpha_blending(drawables)

        # img_list.append()
        # Image.fromarray(np.concat([blended, fullpage])).save('local_tst.png')

if __name__ == '__main__':
    cli()
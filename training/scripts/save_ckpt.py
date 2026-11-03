
import os
import random
import os.path as osp

import click
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
)

from utils.torch_utils import img2tensor, tensor2img
from utils.torch_utils import init_model_from_pretrained, tensor2img, seed_everything
from modules.layerdiffuse.vae import TransparentVAEDecoder, TransparentVAEEncoder, vae_encode, TransparentVAE
from modules.layerdiffuse.utils import patch_transvae_sd, patch_unet_convin
from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel

from modules.marigold import MarigoldDepthPipeline


@click.group()
def cli():
    """live2d scripts.
    """

@cli.command('save_vae')
@click.option('--ckpt')
@click.option('--savep')
def save_vae(ckpt, savep):
    vae = AutoencoderKL.from_pretrained(
        "cagliostrolab/animagine-xl-4.0",
        subfolder="vae",
        revision=None,
        variant=None,
    )
    trans_vae = TransparentVAE.from_pretrained(
        'layerdifforg/seethroughv0.0.1_layerdiff3d',
        subfolder="trans_vae"
    )

    td_sd = {}
    vae_sd = {}
    sd = load_file(ckpt)
    for k, v in sd.items():
        if k.startswith('trans_decoder.'):
            td_sd[k.lstrip('trans_decoder.')] = v
        elif k.startswith('vae.'):
            vae_sd[k.replace('vae.', '')] = v

    os.makedirs(savep, exist_ok=True)


    if len(vae_sd) > 0:
        vae.load_state_dict(vae_sd)
    vae.save_pretrained(osp.join(savep, 'vae'))

    if len(td_sd) > 0:
        trans_vae.decoder.load_state_dict(td_sd)
    trans_vae.save_pretrained(osp.join(savep, 'trans_vae'))
    

if __name__ == '__main__':
    cli()
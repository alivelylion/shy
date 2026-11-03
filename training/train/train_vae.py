import sys
import os.path as osp
import argparse
import logging
import math
import os
import shutil
import datetime
import gc
import random
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from PIL import Image
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import transformers
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils.import_utils import is_xformers_available
# from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
import cv2

from train.eval_utils import AvgMeter
from train.loss_vae import LPIPSWithDiscriminator, ReconLoss, ConvNextType
from modules.layerdiffuse.vae import TransparentVAEEncoder, TransparentVAEDecoder, vae_encode
from modules.layerdiffuse.utils import patch_transvae_sd, conv_add_channels, patch_unet_convin
from utils.io_utils import flatten_dict, imglist2imgrid, json2dict, dict2json
from train.dataset_layerdiff import LayerDiffVAEDataset
from utils.torch_utils import init_model_from_pretrained, tensor2img, seed_everything, img2tensor
from utils.torchcv import pad_rgb_torch
from utils.visualize import imglist2imgrid_with_tags
from utils.cv import argb2rgba, img_alpha_blending, checkerboard, center_square_pad_resize, pad_rgb


logger = get_logger(__name__, log_level="INFO")


class CompositeVAE(nn.Module):

    def __init__(self, vae: AutoencoderKL, trans_encoder: TransparentVAEEncoder, trans_decoder: TransparentVAEDecoder, loss_args=None):
        super().__init__()
        self.vae = vae
        self.trans_encoder = trans_encoder
        self.trans_decoder = trans_decoder
        self.train_vae_decoder = False

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def init_trainable(self, train_vae_decoder = True, train_trans_decoder=True):
        self.vae.requires_grad_(False)
        self.vae.decoder.requires_grad_(train_vae_decoder)
        self.trans_encoder.requires_grad_(False)
        if self.trans_decoder is not None:
            self.trans_decoder.requires_grad_(train_trans_decoder)
        self.train_vae_decoder = self.train_vae_decoder
    
    def train(self, mode = True):
        if self.train_vae_decoder:
            self.vae.decoder.train(mode)
        if self.trans_decoder is not None:
            self.trans_decoder.train(mode)
        return self

    def get_last_layer(self):
        if self.trans_decoder is not None:
            return self.trans_decoder.model.conv_out.weight

    def forward(self, argb, use_offset=False, scheduler: DDPMScheduler =None, noise=None, noise_step=0, rgb_cond=None):
        '''
        argb: (-1, 1)
        '''
        with torch.no_grad():
            latent = vae_encode(self.vae, trans_vae_encoder=self.trans_encoder, argb_tensor=argb, use_offset=use_offset, scale=False).clone().detach()
            if noise is not None and scheduler is not None and isinstance(noise_step, torch.Tensor):
                for ii, l in enumerate(latent):
                    step = noise_step[ii]
                    if step > 0:
                        latent[ii] = scheduler.add_noise(l[None] * self.vae.config.scaling_factor, noise[[ii]], noise_step[[ii]])[0] / self.vae.config.scaling_factor
        pixel = self.vae.decode(latent).sample
        pixel = pixel * 0.5 + 0.5
        if self.trans_decoder is not None:
            if rgb_cond is not None:
                pixel = torch.concat([pixel, rgb_cond], dim=-3)
            pixel = self.trans_decoder.model(pixel, latent)
        return pixel


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # model
    parser.add_argument(
        "--model_args", default=None
    )

    parser.add_argument(
        "--train_trans_decoder", default=True
    )

    parser.add_argument("--loss_args", default=None)
    parser.add_argument("--loss_type", default='vaeloss_v2')

    # optimizer & lr scheduler & hyperparameters
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_scheduler_step_rules",
        type=str,
        default=None
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--alpha_reweight",
        default=1.0, type=float
    )

    # dataset
    parser.add_argument(
        "--dataset_args",
        default=None,
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--valset_args",
        default=None,
    )

    # logging, saving, reports & resume
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=10000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--val_mask_thr",
        type=int,
        default=0.
    )
    parser.add_argument(
        "--visualization_steps",
        type=int,
        default=3000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--max_visualize_samples",
        type=int,
        default=8,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_l2d_vae",
    )
    parser.add_argument(
        "--tracker_init_kwargs",
        type=str,
        default=None,
    )

    # performance
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    parser.add_argument(
        "--local_rank",
        type=str,
        default=0,
    )

    # added args
    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--init_from_ckpt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        type=bool,
        default=True
    )

    parser.add_argument("--pretrained_vae", default=None)

    args = parser.parse_args()
    args = OmegaConf.create(vars(args))
    if args.config is not None:
        config = OmegaConf.load(args.config)
        args.merge_with(config)

    if args.output_dir is None:
        if args.config is not None:
            args.output_dir = osp.join('workspace/training_output',
                                       osp.splitext(osp.basename(args.config))[0])
        else: 
            args.output_dir = osp.join('workspace/training_output', args.tracker_project_name)
        

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_args is None:
        args.dataset_args = {}

    if args.model_args is None:
        args.model_args = {}

    if args.loss_args is None:
        args.loss_args = {}


    return args

def checkerboard_vis(y: torch.Tensor):
    y = y.clip(0, 1).movedim(1, -1)
    if y.shape[-1] == 4:
        alpha = y[..., :1]
        fg = y[..., 1:]
    else:
        fg = y
        alpha = torch.ones_like(y[..., [0]])
    B, H, W, C = fg.shape
    cb = checkerboard(shape=(H // 64, W // 64))
    cb = cv2.resize(cb, (W, H), interpolation=cv2.INTER_NEAREST)
    cb = (0.5 + (cb - 0.5) * 0.1)[None, ..., None]
    cb = torch.from_numpy(cb).to(fg)

    vis = (fg * alpha + cb * (1 - alpha))[0]
    vis = (vis * 255.0).detach().cpu().float().numpy().clip(0, 255).astype(np.uint8)
    return vis

@torch.inference_mode()
def visualize_batch(args, batch, model, dataset, accelerator: Accelerator, global_step):

    model = accelerator.unwrap_model(model)
    model.eval()
    weight_dtype = model.dtype

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise = torch.randn((1, 4, dataset.target_size // 8, dataset.target_size // 8), dtype=weight_dtype, device=accelerator.device)
    for ii, srcp in enumerate(args.visualize_src):

        def _infer_pairs(p, tag=None):
            img = np.array(Image.open(p))
            img = center_square_pad_resize(img, target_size=dataset.target_size)
            img = pad_rgb(img, return_format='argb')
            img = img2tensor(img=img, dim_order='bchw')
            rgb_cond = None
            if args.dataset_args.load_rgb_cond:
                rgb_cond = rgb_fullpage
                if tag is not None:
                    tx1, ty1, tx2, ty2 = ann['tag_info'][tag]['xyxy']
                    rgb_cond = rgb_fullpage[ty1: ty2, tx1: tx2]
                rgb_cond = center_square_pad_resize(rgb_cond, target_size=dataset.target_size)
                rgb_cond = img2tensor(rgb_cond, normalize=True, dtype=weight_dtype, device=accelerator.device)
            noise_step = torch.tensor([ii], dtype=torch.long, device=accelerator.device)
            recon = model(img.to(dtype=weight_dtype), scheduler=noise_scheduler, noise_step=0, noise=noise, rgb_cond=rgb_cond)
            from utils.io_utils import save_tmp_img
            # save_tmp_img(np.concatenate([checkerboard_vis(img), checkerboard_vis(recon)], axis=1))
            return np.concatenate([checkerboard_vis(img), checkerboard_vis(recon)], axis=1)
        
        rgb_fullpage = np.array(Image.open(srcp))[..., :3]
        vis_list = []
        vis_list.append(_infer_pairs(srcp, ))
        srcname = osp.splitext(srcp)[0]
        ann = json2dict(srcname + '_ann.json')

        valid_taglist = {'head', 'front hair', 'back hair', 'headwear', 'irides', 'topwear', 'handwear', 'bottomwear', 'legwear'}
        nvalid = len(valid_taglist)
        for tag in dataset.tag_list:
            p = srcname + f'_{tag}.png'
            if tag not in valid_taglist:
                continue
            if not osp.exists(p):
                continue
            vis_list.append(_infer_pairs(p, tag))

        # if ii == 0:
        #     vis_list.append(checkerboard_vis(batch['img'][[0]]))
        # vis = imglist2imgrid(vis_list)
        sample_saved = False
        if args.report_to == 'wandb':
            try:
                import wandb
                wandb_tracker = accelerator.get_tracker("wandb")
                for jj, vis in enumerate(vis_list):
                    wandb_tracker.log({f'train/samples': wandb.Image(vis)}, step=global_step + ii * nvalid + jj)
                sample_saved = True
            except:
                pass
        if not sample_saved:
            for jj, vis in enumerate(vis_list):
                Image.fromarray(vis).save(osp.join(args.output_dir, 'step-' + str(global_step + ii * nvalid + jj).zfill(4) + '.jpg'), q=97)

    model.train()
    gc.collect()
    torch.cuda.empty_cache()

    return vis



def init_vae(args):
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
    )
    trans_vae_encoder: TransparentVAEEncoder = init_model_from_pretrained(
        'lllyasviel/LayerDiffuse_Diffusers',
        TransparentVAEEncoder, weights_name='ld_diffusers_sdxl_vae_transparent_encoder.safetensors',
        patch_state_dict_func=patch_transvae_sd
    )

    # trans_vae_decoder: TransparentVAEDecoder = init_model_from_pretrained(
    #     'lllyasviel/LayerDiffuse_Diffusers',
    #     TransparentVAEDecoder, weights_name='ld_diffusers_sdxl_vae_transparent_decoder.safetensors',
    #     patch_state_dict_func=patch_transvae_sd
    # )
    if args.train_trans_decoder:
        trans_vae_decoder: TransparentVAEDecoder = init_model_from_pretrained(
            'lllyasviel/LayerDiffuse_Diffusers',
            TransparentVAEDecoder, weights_name='ld_diffusers_sdxl_vae_transparent_decoder.safetensors',
            patch_state_dict_func=patch_transvae_sd
        )
    else:
        trans_vae_decoder = None
    if args.dataset_args.get('load_rgb_cond', False):
        patch_unet_convin(trans_vae_decoder.model, target_in_channels=6)
    model = CompositeVAE(vae, trans_vae_encoder, trans_vae_decoder).to(dtype=torch.float32)
    if args.get('pretrained_ckpt', False):
        from safetensors.torch import load_file
        sd = load_file('workspace/training_output/vae_1280_lossv2_wotrans/checkpoint-10000.safetensors')
        print(model.load_state_dict(sd, strict=False))
    model.init_trainable(args.train_vae_decoder, args.train_trans_decoder)
    return model



LOSS_TYPE = "mse"
LPIPS_NET = "vgg"
USE_CONVNEXT = True

DLR = 1e-3

NEW_LATENT_DIM = None
PRETRAIN = False
NOSE_COMPOSE = True

class VAELossV2(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.recon_loss=ReconLoss(
            loss_type=LOSS_TYPE,
            lpips_net=LPIPS_NET,
            convnext_type=ConvNextType.TINY if USE_CONVNEXT else None,
            convnext_kwargs={
                "feature_layers": [2, 6, 10, 14],
                "use_gram": False,
                "input_range": (-1, 1),
                "device": "cuda",
            },
            loss_weights={
                LOSS_TYPE: 0.25,
                "lpips": 0.3,
                "convnext": 0.45,
            },
            noise_compose=NOSE_COMPOSE
        )
        self.recon_loss_weight = 0.5
        self.disc_factor = 0.

    def forward(self, x, x_rec, *args, **kwargs):
        loss = 0
        if x_rec.shape[1] == 3 and x.shape[1] == 4:
            x = x[:, 1:]
        recon_loss = self.recon_loss(x, x_rec)
        loss += recon_loss * self.recon_loss_weight
        return loss, {
            'recon_loss': recon_loss.detach()
        }


def main():
    args = parse_args()
    from accelerate import DistributedDataParallelKwargs

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )
    use_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED
    if use_deepspeed:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_batch_size

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        seed_everything(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    initial_global_step = 0
    if args.resume_from_checkpoint:
        raise Exception('Not implemented')

    if args.gradient_checkpointing:
        raise Exception('Not implemented')

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_name = args.optimizer.lower()
    if optimizer_name == 'adamw':

        # Initialize the optimizer
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW
    elif optimizer_name == 'muon':
        pass

    model = init_vae(args)
    model.train()


    if args.loss_type == 'vaeloss_v2':
        vae_loss = VAELossV2(**OmegaConf.to_container(args.loss_args))
    else:
        vae_loss = LPIPSWithDiscriminator(**OmegaConf.to_container(args.loss_args))
        vae_loss.train()


    ganloss = vae_loss.disc_factor > 0

    if args.pretrained_vae is not None:
        from safetensors.torch import load_file
        model.load_state_dict(load_file(args.pretrained_vae))

    # for param in model.img_adapter.parameters(): param.data = param.data.contiguous()

    parameters = [p for p in model.parameters() if p.requires_grad]
    parameters_with_lr = [{"params": parameters, "lr": args.learning_rate}]
    optimizer = optimizer_cls(
        parameters_with_lr,
        **OmegaConf.to_container(args.optimizer_args)
    )

    train_dataset = LayerDiffVAEDataset(**OmegaConf.to_container(args.dataset_args))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        step_rules=args.lr_scheduler_step_rules
    )

    if ganloss:
        optimizer_D = optimizer_cls(
            [{"params": vae_loss.parameters(), "lr": args.learning_rate}] , **OmegaConf.to_container(args.optimizer_D_args)
        )

        lr_scheduler_D = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_D,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            step_rules=args.lr_scheduler_step_rules
        )
        vae_loss, optimizer_D, lr_scheduler_D = accelerator.prepare(
            vae_loss, optimizer_D, lr_scheduler_D
        )
    else:
        vae_loss = vae_loss.to(device='cuda')

    # Prepare everything with our `accelerator`.
    get_last_layer = model.get_last_layer
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    train_dataloader = accelerator.prepare_data_loader(train_dataloader, device_placement=False)
    # vae.to(accelerator.device, dtype=torch.float32)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if accelerator.is_main_process:
        dcfg = OmegaConf.to_container(args)
        tracker_init_kwargs = dcfg['tracker_init_kwargs']
        if tracker_init_kwargs is None:
            tracker_init_kwargs = {}
        # otherwise accelerate refuse the config dict
        dcfg = flatten_dict(dcfg, separator='.')
        for k, v in dcfg.items():
            if isinstance(v, list):
                if len(v) > 0 and isinstance(v[0], (int, float)):
                    dcfg[k] = torch.tensor(v)
                elif isinstance(v, list):
                    dcfg[k] = None
                elif isinstance(v, dict):
                    dcfg[k] = None
                else:
                    dcfg[k] = v
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        if args.report_to == 'wandb' and 'wandb' not in tracker_init_kwargs:
            tracker_init_kwargs['wandb'] = {}
        if 'wandb' in tracker_init_kwargs:
            runname_prefix = osp.basename(args.output_dir)
            tracker_init_kwargs['wandb']['name'] = runname_prefix + '_' + now
            tracker_init_kwargs['wandb']['dir'] = args.output_dir
        accelerator.init_trackers(args.tracker_project_name, dcfg, tracker_init_kwargs)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    loss_dict = {}

    torch.cuda.empty_cache()

    # if args.diff_noise_generator:
    noise_generator = None
    rank = int(os.environ.get('LOCAL_RANK',-1))
    if args.seed:
        rank += args.seed
    noise_generator = torch.Generator(device=accelerator.device)
    noise_generator.manual_seed(rank)

    D_step = False

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # if accelerator.is_main_process:
            #     visualize_batch(args, batch, model, train_dataset, accelerator, global_step)
            img = batch['img']
            rgb_cond = None
            if 'rgb_cond' in batch:
                rgb_cond = batch['rgb_cond'].to(device=accelerator.device, dtype=weight_dtype)
            #     vis = np.concat([checkerboard_vis(img), tensor2img(rgb_cond, denormalize=True)])
            #     Image.fromarray(vis).save('local_tst.png')
            #     pass
            bsz, c, im_h, im_w = img.shape
            if not train_dataset.pad_argb:
                with torch.no_grad():
                    img = pad_rgb_torch(img.to(device=accelerator.device, dtype=torch.float32), return_format='argb')
            else:
                img = img.to(device=accelerator.device, dtype=torch.float32)

            noise = None
            noise_step = 0
            if args.noise_max_step > 1:

                # timesteps = torch.randint(
                #     1, args.noise_max_step, (bsz,), device=accelerator.device, generator=noise_generator
                # )
                # for ii in range(bsz):
                #     if random.random() > args.noise_prob:
                #         timesteps[ii] = 0

                noise_step = torch.randint(
                    1, args.noise_max_step, (bsz,), device=accelerator.device, generator=noise_generator
                )
                for ii in range(bsz):
                    if random.random() > args.noise_prob:
                        noise_step[ii] = 0
                noise = torch.randn((bsz, c, train_dataset.target_size // 8, train_dataset.target_size // 8), dtype=weight_dtype, device=accelerator.device)
            
            if ganloss:
                D_step = True
            else:
                D_step = False
            if D_step:
                with accelerator.accumulate(vae_loss):
                    with torch.no_grad():
                        preds = model(img.to(dtype=weight_dtype), scheduler=noise_scheduler, noise=noise, noise_step=noise_step, rgb_cond=rgb_cond)
                    loss, loss_log = vae_loss(img.to(dtype=weight_dtype), preds, optimizer_idx=1, global_step=global_step + 1, last_layer=get_last_layer(), rgb_cond=rgb_cond, alpha_reweight=args.alpha_reweight)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(vae_loss.parameters(), args.max_grad_norm)
                        D_step = False
                    optimizer_D.step()
                    lr_scheduler_D.step()
                    optimizer.zero_grad()
                    optimizer_D.zero_grad()
                    loss_log['dloss'] = loss

                # Gather the losses across all processes for logging
                for loss_key, loss_val in loss_log.items():
                    if loss_key not in loss_dict:
                        loss_dict[loss_key] = 0
                    avg_loss = accelerator.gather(loss_val.repeat(args.train_batch_size)).mean()
                    loss_dict[loss_key] += avg_loss.item() / args.gradient_accumulation_steps

            if not D_step:
                with accelerator.accumulate(model):
                    preds = model(img.to(dtype=weight_dtype), scheduler=noise_scheduler, noise=noise, noise_step=noise_step, rgb_cond=rgb_cond)
                    # print(img.shape, preds.shape)
                    # print(img.min(), img.max(), preds.min(), preds.max())
                    loss, loss_log = vae_loss(img.to(dtype=weight_dtype), preds, optimizer_idx=0, global_step=global_step + 1, last_layer=get_last_layer(), rgb_cond=rgb_cond, alpha_reweight=args.alpha_reweight)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        if ganloss:
                            D_step = True
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    if ganloss:
                        optimizer_D.zero_grad()
                    loss_log['gloss'] = loss

                # Gather the losses across all processes for logging
                for loss_key, loss_val in loss_log.items():
                    if loss_key not in loss_dict:
                        loss_dict[loss_key] = 0
                    avg_loss = accelerator.gather(loss_val.repeat(args.train_batch_size)).mean()
                    loss_dict[loss_key] += avg_loss.item() / args.gradient_accumulation_steps

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if D_step or not ganloss:
                    progress_bar.update(1)
                    global_step += 1
                    if accelerator.is_main_process:
                        if global_step % args.visualization_steps == 0:
                            visualize_batch(args, batch, model, train_dataset, accelerator, global_step)

                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None and accelerator.is_main_process:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            # sd = accelerator.unwrap_model(model).state_dict()
                            from safetensors.torch import save_file
                            save_file(accelerator.unwrap_model(model).state_dict(), save_path + '.safetensors')
                            if ganloss:
                                save_file(accelerator.unwrap_model(vae_loss).state_dict(), save_path + '_desc.safetensors')
                            logger.info(f"Saved state to {save_path}")

                accelerator.log(loss_dict, step=global_step)
                loss_dict = {}

                # global_step = 0


                # if global_step % args.validation_steps == 0 and args.valset_args is not None:
                #     log_dict = depth_benchmark_marigold(
                #         config=args, 
                #         model=accelerator.unwrap_model(model), 
                #         accelerator=accelerator,
                #         save_dir=osp.join(args.output_dir, f'step{str(global_step).zfill(4)}_benchmark'),
                #         visualize=True,
                #         vae=vae,
                #         empty_text_embed=empty_text_tensor[[0]]
                #     )
                #     to_log_dict = {}
                #     for k, val in log_dict.items():
                #         to_log_dict['val/' + k] = val
                #     if accelerator.is_main_process:
                #         print(f'step {global_step} val result: {to_log_dict}')
                #     accelerator.log(to_log_dict)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()
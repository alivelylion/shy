import sys
import os.path as osp
import argparse
import logging
import math
import os
import shutil
import datetime
import gc
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from PIL import Image
from omegaconf import OmegaConf
import torch
import transformers
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
import cv2

from modules.depth_anything_v2.adapter import DepthAnythingV2, ExtendDepthAnythingV2
from train.loss_depth import affine_invariant_loss, grad_match_loss, SiLogLoss
from train.benchmark import depth_benchmark
from utils.io_utils import flatten_dict, imglist2imgrid
from train.dataset_depth import DepthDataset
from utils.torch_utils import init_model_from_pretrained, tensor2img, seed_everything


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # model
    parser.add_argument(
        "--model_args", default=None
    )

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
        default="train_l2d_depth",
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
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
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
        "--pretrained_model",
        type=str,
        default=None
    )

    parser.add_argument(
        "--loss_gradient_matching_scale",
        type=float,
        default=1.
    )

    parser.add_argument(
        "--loss_siglog_scale",
        type=float,
        default=0.
    )

    parser.add_argument(
        "--loss_affine_invariant_scale",
        type=float,
        default=1.
    )

    parser.add_argument(
        "--ignore_mask",
        type=bool,
        default=False
    )

    parser.add_argument(
        "--scale_before_loss",
        type=bool,
        default=True
    )


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

    return args


@torch.inference_mode()
def visualize_batch(args, batch, model, dataset, accelerator: Accelerator, global_step):

    def _visualize_depth(depth: torch.Tensor, mask):
        depth_min = torch.min(depth[mask])
        depth_max = torch.max(depth[mask])
        depth = (depth - depth_min) / (depth_max - depth_min)
        depth = depth.to(device='cpu', dtype=torch.float32).numpy()
        if depth.ndim == 3:
            depth = depth[0]
        depth = (depth * 255).astype(np.uint8)
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        return depth

    model = accelerator.unwrap_model(model)
    weight_dtype = model.dtype
    img = batch['img'].to(dtype=weight_dtype)
    masks = batch['masks'].to(dtype=weight_dtype)
    bsz, _, im_h, im_w = img.shape
    
    control_input = None
    if 'control_input' in batch:
        control_input = batch['control_input'].to(dtype=weight_dtype)

    preds = model(img, control_input=control_input)

    vis_list = []
    for ii in range(min(args.max_visualize_samples, bsz)):
        img = batch['img'][ii]
        masks = batch['masks'][ii]
        depth = batch['depth'][ii]
        depth_preds = preds[ii]

        img_np = tensor2img(img, mean=dataset.pixel_mean, std=dataset.pixel_std, denormalize=True)
        depth_rgb = _visualize_depth(depth, masks)
        depth_pred_rgb = _visualize_depth(depth_preds, masks)
        vis_list += [img_np, depth_rgb, depth_pred_rgb]

    vis = imglist2imgrid(vis_list, cols=len(vis_list))

    sample_saved = False
    if args.report_to == 'wandb':
        try:
            import wandb
            wandb_tracker = accelerator.get_tracker("wandb")
            wandb_tracker.log({f'train/samples': wandb.Image(vis)}, step=global_step)
            sample_saved = True
        except:
            pass
    if not sample_saved:
        Image.fromarray(vis).save(osp.join(args.output_dir, 'step-' + str(global_step).zfill(4) + '.jpg'), q=97)

    return vis


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    use_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED

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


    # some how init model here
    # model = None
    


    with_ctrl = 'with_ctrl' in args.model_type

    if "depthanythingv2" in args.model_type:
        if "metric" in args.model_type:
            pretrained_model_name_or_path = "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large"
            weights_name = "depth_anything_v2_metric_hypersim_vitl.pth"
        else:
            pretrained_model_name_or_path = "depth-anything/Depth-Anything-V2-Large"
            weights_name = "depth_anything_v2_vitl.pth"

        # if args.model_type == 'depthanythingv2_':

        if "ctrl" in args.model_type:
            if not args.init_from_ckpt:
                model: ExtendDepthAnythingV2 = ExtendDepthAnythingV2(
                        depth_model=init_model_from_pretrained(
                            pretrained_model_name_or_path=pretrained_model_name_or_path,
                            module_cls=DepthAnythingV2,
                            model_args=OmegaConf.to_container(args.model_args),
                            weights_name=weights_name
                        )
                    )
                model.init_adapters()
            else:
                model: ExtendDepthAnythingV2 = init_model_from_pretrained(
                    pretrained_model_name_or_path=args.init_from_ckpt,
                    module_cls=ExtendDepthAnythingV2,
                    download_from_hf=False,
                    model_args=OmegaConf.to_container(args.model_args)
                )
        else:
            model: DepthAnythingV2 = init_model_from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                module_cls=DepthAnythingV2,
                model_args=OmegaConf.to_container(args.model_args),
                weights_name=weights_name
            )

    model.train()

    # for param in model.img_adapter.parameters(): param.data = param.data.contiguous()

    parameters = [p for p in model.parameters() if p.requires_grad]
    print(len(parameters))
    parameters_with_lr = [{"params": parameters, "lr": args.learning_rate}]
    optimizer = optimizer_cls(
        parameters_with_lr,
        **OmegaConf.to_container(args.optimizer_args)
    )


    train_dataset = DepthDataset(**OmegaConf.to_container(args.dataset_args))
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

    # Prepare everything with our `accelerator`.

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

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

    torch.cuda.empty_cache()

    def _scale_depth(depth, mask):
        depth_max = depth[mask].max() + 1e-6
        return depth / depth_max

    if args.loss_siglog_scale > 0:
        siglog_loss = SiLogLoss().to('cuda')

    loss_dict = {}
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # if accelerator.is_main_process:
            #     visualize_batch(args, batch, model, train_dataset, accelerator, global_step)
                
            with accelerator.accumulate(model):

                img = batch['img'].to(dtype=weight_dtype)
                masks = batch['masks']
                depth = batch['depth']

                # mask_valid_list = batch['mask_valid_list']
                bsz, _, im_h, im_w = img.shape
                num_cls = masks.shape[1]

                if args.ignore_mask:
                    masks = torch.ones_like(masks)
                    batch['masks'] = masks

                control_input = None
                if with_ctrl:
                    control_input = batch['control_input'].to(dtype=weight_dtype)

                # with torch.autocast(device_type='cuda', enabled=weight_dtype != torch.float32):
                preds = model(img, control_input=control_input)

                # print(depth.max(), depth.min(), preds.max(), preds.min())

                if args.scale_before_loss:
                    depth = _scale_depth(depth, masks)
                    preds = _scale_depth(preds, masks)                    

                # mask_valid_list = mask_valid_list.float()
                loss = 0.
                loss_dict_local = {}
                if args.loss_gradient_matching_scale > 0:
                    loss_gm = grad_match_loss(preds, depth, masks) * args.loss_gradient_matching_scale
                    loss += loss_gm
                    loss_dict_local['loss_gm'] = loss_gm
                if args.loss_affine_invariant_scale > 0:
                    loss_af = affine_invariant_loss(preds, depth, masks) * args.loss_affine_invariant_scale
                    loss += loss_af
                    loss_dict_local['loss_af'] = loss_af

                if args.loss_siglog_scale > 0:
                    loss_siglog = siglog_loss(preds, depth, masks) * args.loss_siglog_scale
                    loss += loss_siglog
                    loss_dict_local['loss_siglog'] = loss_siglog

                # Gather the losses across all processes for logging
                for loss_key, loss_val in loss_dict_local.items():
                    if loss_key not in loss_dict:
                        loss_dict[loss_key] = 0
                    avg_loss = accelerator.gather(loss_val.repeat(args.train_batch_size)).mean()
                    loss_dict[loss_key] += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(loss_dict, step=global_step)
                loss_dict = {}

                # global_step = 0
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
                        logger.info(f"Saved state to {save_path}")

                if accelerator.is_main_process:
                    if global_step % args.visualization_steps == 0:
                        visualize_batch(args, batch, model, train_dataset, accelerator, global_step)

                if global_step % args.validation_steps == 0 and args.valset_args is not None:
                    log_dict = depth_benchmark(
                        config=args, 
                        model=accelerator.unwrap_model(model), 
                        accelerator=accelerator,
                        save_dir=osp.join(args.output_dir, f'step{str(global_step).zfill(4)}_benchmark'),
                        visualize=True
                    )
                    to_log_dict = {}
                    for k, val in log_dict.items():
                        to_log_dict['val/' + k] = val
                    if accelerator.is_main_process:
                        print(f'step {global_step} val result: {to_log_dict}')
                    accelerator.log(to_log_dict)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()
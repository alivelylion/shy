import sys
import os.path as osp
import os
import gc
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from PIL import Image
from einops import rearrange, reduce
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from tqdm import tqdm
import click
import cv2

from train.eval_utils import AvgMeter
from utils.io_utils import flatten_dict, imglist2imgrid, dict2json
from train.dataset_seg import SemanticSegDataset
from train.dataset_depth import DepthDataset

from utils.torch_utils import init_model_from_pretrained, fix_params, tensor2img
from utils.visualize import visualize_segs, visualize_segs_with_labels



@click.group()
def cli():
    """live2d scripts.
    """




@cli.command('facedet_benchmark')
@click.option('--config')
@click.option('--ckpt', default=None)
@click.option('--save_dir', default=None)
@click.option('--visualize', default=False, is_flag=True)
@click.option('--accelerator', default=None)
@click.option('--model', default=None)
@click.option('--ckpt_prefix', default=None)
@click.option('--ckpt_suffix', default='.pt,.safetensors,.pth,.ckpt')
def semsam_benchmark_cli(
    **kwargs
):
    ckpt_prefix = kwargs.pop('ckpt_prefix')
    if ckpt_prefix is None:
        return semsam_benchmark(**kwargs)
    else:
        
        ckpt_suffix = kwargs.pop('ckpt_suffix')
        ckpt_suffix = ckpt_suffix.split(',')
        d = osp.dirname(ckpt_prefix)
        ckpt_lst = []
        for dn in os.listdir(d):
            if os.path.splitext(dn)[-1] in ckpt_suffix:
                ckpt_lst.append(osp.join(d, dn))
        kwargs.pop('ckpt')
        config = kwargs['config']
        input_save_dir = kwargs.pop('save_dir')
        for ckpt in ckpt_lst:
            print(f'running benchmark for {ckpt}')
            model_name = osp.basename(osp.splitext(ckpt)[0])
            if input_save_dir is not None:
                save_dir = osp.join(input_save_dir, model_name + '_benchmark')
            else:
                save_dir = osp.join(osp.splitext(config)[0] + '_benchmark', f'{model_name}')
            
            semsam_benchmark(ckpt=ckpt, save_dir=save_dir, **kwargs, )

        pass


def semsam_benchmark(
    config,
    ckpt=None,
    save_dir=None,
    visualize=False,
    accelerator=None,
    model=None
):
    
    from modules.semanticsam import SemanticSam, Sam

    if save_dir is None:
        assert isinstance(config, str)
        save_dir = osp.splitext(config)[0] + '_benchmark'
    os.makedirs(save_dir, exist_ok=True)

    device = 'cuda'
    if isinstance(config, str):
        config = OmegaConf.load(config)

    mask_thr = getattr(config, 'val_mask_thr', 0.0)
    val_visualize_interval = getattr(config, 'val_visualize_interval', 4)
    
    dataset = SemanticSegDataset(**OmegaConf.to_container(config.valset_args))

    if model is None:
        model: SemanticSam = init_model_from_pretrained(
            pretrained_model_name_or_path=ckpt,
            module_cls=SemanticSam,
            model_args=OmegaConf.to_container(config.model_args),
            device=device
        ).eval()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=getattr(config, 'val_batch_size', 1),
        num_workers=getattr(config, 'dataloader_num_workers', 1),
        drop_last=True
    )
    if accelerator is not None:
        dataloader = accelerator.prepare_data_loader(dataloader)

    weight_dtype = model.dtype
    log_meter = AvgMeter()

    for ii, batch in enumerate(tqdm(dataloader, desc='running validation...', disable=accelerator is not None and not accelerator.is_main_process)):
        img = batch['img'].to(dtype=weight_dtype, device=device)
        masks = batch['masks'].to(dtype=weight_dtype, device=device)
        bsz, _, im_h, im_w = img.shape
        num_cls = masks.shape[1]

        with torch.inference_mode():
            low_res_masks, iou_predictions = model(img=img)
        mh, mw = masks.shape[-2:]

        if low_res_masks.shape[-1] != mw or low_res_masks.shape[-2] != mh:
            mask_preds = F.interpolate(low_res_masks, (mh, mw), mode='bilinear')
        else:
            mask_preds = low_res_masks

        if visualize and (accelerator is None or accelerator.is_main_process) \
            and (ii + 1) % val_visualize_interval == 0:
            vis_list = []
            for jj in range(bsz):
                img_np = tensor2img(img[jj], mean=dataset.pixel_mean, std=dataset.pixel_std, denormalize=True)
                masks_np = masks[jj].to(device='cpu', dtype=torch.bool).numpy()
                rst_gt = visualize_segs_with_labels(masks_np, tag_list=None, src_img=img_np, image_weight=0.)
                masks_np = (mask_preds[jj] > mask_thr).to(device='cpu', dtype=torch.bool).numpy()
                reference_img = np.concat([img_np, rst_gt], axis=1)
                rst_preds = visualize_segs_with_labels(masks_np, src_img=img_np, tag_list=dataset.tag_list, image_weight=0., reference_img=reference_img)
                vis_list.append(rst_preds)

            vis = imglist2imgrid(vis_list, cols=1)
            Image.fromarray(vis).save(osp.join(save_dir, str(ii).zfill(3) + '.jpg'), q=97)

        mask_preds = mask_preds > mask_thr
        masks = masks > mask_thr

        mask_preds = rearrange(mask_preds, 'b c h w -> b c (h w)')
        masks = rearrange(masks, 'b c h w -> b c (h w)')

        unions = reduce(mask_preds | masks, 'b c n -> b c', 'sum')
        zero_unions_msk = unions == 0
        unions[zero_unions_msk] = 1
        intersections = reduce(mask_preds & masks, 'b c n -> b c', 'sum')
        intersections[zero_unions_msk] = 1
        miou = reduce(intersections / (unions + 1e-6), 'b c -> b', 'mean')
        pixel_precision = reduce((mask_preds == masks).to(torch.float32), 'b c n -> b', 'mean')

        loss_dict = {}
        loss_dict['pixel_precision'] = pixel_precision
        loss_dict['miou'] = miou

        if accelerator is not None:
            loss_dict_gather = accelerator.gather_for_metrics(loss_dict)
            log_meter.add(loss_dict_gather)
        else:
            log_meter.add(loss_dict)
    
    del dataloader
    gc.collect()
    torch.cuda.empty_cache()

    log_dict = log_meter.compute()
    if accelerator is None or accelerator.is_main_process:
        print(log_dict)
        dict2json(log_dict, osp.join(save_dir, 'benchmark.json'))
    return log_dict


@cli.command('plot_benchmark_list')
@click.option('--src_dir')
@click.option('--savep', default=None)
def plot_benchmark_list_cli(**kwargs):
    plot_benchmark_list(**kwargs)

def plot_benchmark_list(
    src_dir,
    savep
):
    import matplotlib.pyplot as plt
    from utils.io_utils import json2dict

    src_list = os.listdir(src_dir)
    # src_list.sort(key = lambda x: int(x.split('-')[-1]))

    benchmark_list = []
    for src in src_list:
        step = src.split('_')[0]
        if 'step' not in step:
            continue
        step = step.lstrip('step')
        if not step.isdigit():
            continue
        step = int(step)
        benchmark = json2dict(osp.join(src_dir, src, 'benchmark.json'))
        benchmark['step'] = step
        benchmark_list.append(benchmark)

    benchmark_list.sort(key = lambda x: x['step'])
    metrics = list(benchmark_list[0].keys())
    metrics.remove('step')
    fig, ax_lst = plt.subplots(len(metrics), 1, layout='constrained', figsize=(int(round(len(benchmark_list) * 0.5)), 20), dpi=300)
    step_lst = [b['step'] for b in benchmark_list]

    cmap = plt.cm.viridis 

    for ii, ax in enumerate(ax_lst):
        metric = metrics[ii]
        values = [b[metric] for b in benchmark_list]
        ax.plot(step_lst, values, color=cmap(ii / (len(ax_lst) - 1)))
        ax.set_ylabel(metric)

        for jj, v in enumerate(values):
            v = f"{values[jj]:.4f}"
            if v.startswith('0.'):
                v = v[1:]
            ax.text(step_lst[jj], values[jj], v, ha='center', va='bottom')

    if savep is None:
        savep = osp.join(src_dir, 'plot.png')
    fig.savefig(savep)
    print(f'plot saved to {savep}')


@torch.inference_mode()
def depth_benchmark(
    config,
    ckpt=None,
    save_dir=None,
    visualize=False,
    accelerator=None,
    model=None,
    ignore_mask=False
):
    def _visualize_depth(depth: torch.Tensor, mask):
        depth_max = torch.max(depth[mask])
        depth = depth / depth_max
        depth = depth.to(device='cpu', dtype=torch.float32).numpy()
        if depth.ndim == 3:
            depth = depth[0]
        depth = (depth * 255).astype(np.uint8)
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        return depth

    from modules.depth_anything_v2.adapter import DepthAnythingV2, ExtendDepthAnythingV2
    from train.loss_depth import compute_metrics

    if save_dir is None:
        assert isinstance(config, str)
        save_dir = osp.splitext(config)[0] + '_benchmark'
    os.makedirs(save_dir, exist_ok=True)

    device = 'cuda'
    if isinstance(config, str):
        config = OmegaConf.load(config)

    mask_thr = getattr(config, 'val_mask_thr', 0.0)
    val_visualize_interval = getattr(config, 'val_visualize_interval', 4)
    
    dataset = DepthDataset(**OmegaConf.to_container(config.valset_args))

    if model is None:
        model: DepthDataset = init_model_from_pretrained(
            pretrained_model_name_or_path=ckpt,
            module_cls=DepthDataset,
            model_args=OmegaConf.to_container(config.model_args),
            device=device
        ).eval()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=getattr(config, 'val_batch_size', 1),
        num_workers=getattr(config, 'dataloader_num_workers', 1),
        drop_last=True
    )
    if accelerator is not None:
        dataloader = accelerator.prepare_data_loader(dataloader)

    weight_dtype = model.dtype
    log_meter = AvgMeter()

    for ii, batch in enumerate(tqdm(dataloader, desc='running validation...', disable=accelerator is not None and not accelerator.is_main_process)):
        img = batch['img'].to(dtype=weight_dtype, device=device)
        masks = batch['masks']

        if ignore_mask:
            masks = torch.ones_like(masks)
            batch['masks'] = masks

        bsz, _, im_h, im_w = img.shape

        with torch.inference_mode():
            control_input = None
            if 'control_input' in batch:
                control_input = batch['control_input'].to(dtype=weight_dtype)
            preds = model(img, control_input=control_input)

        mh, mw = masks.shape[-2:]

        if visualize and (accelerator is None or accelerator.is_main_process) \
            and (ii + 1) % val_visualize_interval == 0:
            vis_list = []
            for jj in range(bsz):
                img_np = tensor2img(img[jj], mean=dataset.pixel_mean, std=dataset.pixel_std, denormalize=True)

                depth = batch['depth'][jj]
                depth_preds = preds[jj]
                mask = masks[jj]

                depth_rgb = _visualize_depth(depth, mask)
                depth_pred_rgb = _visualize_depth(depth_preds, mask)
                vis_list += [img_np, depth_rgb, depth_pred_rgb]

            vis = imglist2imgrid(vis_list, cols=1)
            Image.fromarray(vis).save(osp.join(save_dir, str(ii).zfill(3) + '.jpg'), q=97)


        loss_dict = compute_metrics(batch['depth'], preds, mask=masks)
        # mask_preds = mask_preds > mask_thr
        # masks = masks > mask_thr

        # mask_preds = rearrange(mask_preds, 'b c h w -> b c (h w)')
        # masks = rearrange(masks, 'b c h w -> b c (h w)')

        # unions = reduce(mask_preds | masks, 'b c n -> b c', 'sum')
        # zero_unions_msk = unions == 0
        # unions[zero_unions_msk] = 1
        # intersections = reduce(mask_preds & masks, 'b c n -> b c', 'sum')
        # intersections[zero_unions_msk] = 1
        # miou = reduce(intersections / (unions + 1e-6), 'b c -> b', 'mean')
        # pixel_precision = reduce((mask_preds == masks).to(torch.float32), 'b c n -> b', 'mean')

        # loss_dict = {}
        # loss_dict['pixel_precision'] = pixel_precision
        # loss_dict['miou'] = miou

        if accelerator is not None:
            loss_dict_gather = accelerator.gather_for_metrics(loss_dict)
            # print(loss_dict_gather)
            log_meter.add(loss_dict_gather)
        else:
            log_meter.add(loss_dict)
    
    del dataloader
    gc.collect()
    torch.cuda.empty_cache()

    log_dict = log_meter.compute()
    if accelerator is None or accelerator.is_main_process:
        print(log_dict)
        dict2json(log_dict, osp.join(save_dir, 'benchmark.json'))
    return log_dict



if __name__ == '__main__':
    cli()
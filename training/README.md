# Training

Training scripts for the See-through layer decomposition models: LayerDiff, Marigold depth,
transparent VAE, and body part segmentation.

This codebase produces the **V3 model** with 23 body-part tag training. To reproduce the
results from the paper, refer to the
[v0.0.1 model](https://huggingface.co/layerdifforg/seethroughv0.0.1_layerdiff3d).

## Environment Setup

Uses the unified `see_through` conda env. See the [root README](../README.md) for setup.

### Optional extras

```bash
# DeepSpeed ZeRO for multi-GPU training (required for LayerDiff)
pip install -r requirements-training-deepspeed.txt

# Experiment tracking
pip install wandb

# 8-bit Adam optimizer (used with --use_8bit_adam in config)
pip install bitsandbytes

# FID / LPIPS / PSNR benchmarks (required by scripts/benchmark.py)
pip install torchmetrics
```

## Data Preparation

Training data is prepared in three stages:

1. **Extract Live2D model layers** using
   [CubismPartExtr](https://github.com/shitagaki-lab/CubismPartExtr) — converts `.moc3` model
   files into per-drawable RGBA images.

2. **Parse and label the extracted layers** using `parse_live2d.py` and the annotation UI — runs
   SAM segmentation and assigns body-part tags. See
   [README_datapipeline.md](../README_datapipeline.md) for the full walkthrough.

3. **Render training samples** using `scripts/data_pipeline.py` — composites labeled layers onto
   background images with augmentation to produce the final training data.

Training data should be placed under `workspace/datasets/` (gitignored). The sample list paths
in the YAML configs (e.g. `workspace/datasets/l2d_bodysamples_v3.txt`) point to text files
listing the training samples, one per line.

## Testbed

Our training was conducted on **8x NVIDIA H200** GPUs. LayerDiff and LayerDiff 3D require
multi-GPU training with DeepSpeed ZeRO-2; other models can be trained on a single GPU.

## Training Pipelines

The two main model families each follow a three-stage pipeline: train a 2D model, convert its
UNet weights to 3D, then fine-tune the 3D model.

**Marigold depth** (SD 1.5 scale):

```
train_marigold_depth.py  -->  cvt_marigold2d_to_3d.py  -->  train_marigold3d.py
    (2D depth model)          (UNet weight conversion)       (3D depth model)
```

**LayerDiff** (SDXL scale):

```
train_layerdiff.py  -->  cvt_layerdiff2d_to_3d.py  -->  train_layerdiff3d.py
   (2D layer model)       (UNet weight conversion)       (3D layer model)
```

**Auxiliary models** (single-stage, single-GPU):

| Script | Purpose |
|--------|---------|
| `train/train_depth.py` | Depth Anything V2 adapter |
| `train/train_vae.py` | Transparent VAE encoder/decoder |
| `train/train_partseg.py` | SAM-HQ body part segmentation |

### Usage

Always run from the repository root:

```bash
cd /path/to/see-through
conda activate see_through

# Multi-GPU training with DeepSpeed (LayerDiff example)
accelerate launch --config_file training/configs/test_ddp_4gpu.json \
  training/train/train_layerdiff.py \
  --config training/configs/test_layerdiff.yaml
```

Accelerate config files are provided for 4-GPU (`test_ddp_4gpu.json`) and 8-GPU
(`ddp_bf16.json`) setups. Adjust `num_processes` to match your GPU count.

## Script Inventory

### Training scripts (`train/`)

| Script | Purpose |
|--------|---------|
| `train_layerdiff.py` | LayerDiff fine-tuning (SDXL, multi-GPU DeepSpeed) |
| `train_layerdiff3d.py` | LayerDiff 3D training (SDXL, multi-GPU DeepSpeed) |
| `train_marigold_depth.py` | Marigold 2D depth estimation |
| `train_marigold3d.py` | Marigold 3D depth |
| `train_partseg.py` | Body part segmentation (SAM-HQ) |
| `train_depth.py` | Depth Anything V2 adapter |
| `train_vae.py` | Transparent VAE encoder/decoder |
| `dataset_layerdiff.py` | LayerDiff dataset loader |
| `dataset_depth.py` | Depth dataset loader |
| `dataset_seg.py` | Segmentation dataset loader |
| `loss_depth.py` | Depth training losses |
| `loss_vae.py` | VAE training losses (LPIPS + ConvNeXt perceptual) |
| `loss_mask_samhq.py` | SAM-HQ mask losses |
| `eval_utils.py` | Evaluation utilities |
| `kepler.py` | Kepler codebook quantizer (VQ-VAE) |
| `benchmark.py` | In-training benchmark utilities |

### Utility scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `cvt_marigold2d_to_3d.py` | Convert Marigold 2D UNet weights to 3D |
| `cvt_layerdiff2d_to_3d.py` | Convert LayerDiff 2D UNet weights to 3D |
| `data_pipeline.py` | Training data rendering and augmentation |
| `benchmark.py` | FID / LPIPS / PSNR evaluation (requires `torchmetrics`) |
| `save_ckpt.py` | Checkpoint format conversion |
| `ckpt.py` | Checkpoint utilities |
| `hf.py` | HuggingFace Hub upload/download helpers |

### Metrics (`metrics/`)

| Script | Purpose |
|--------|---------|
| `clip_score.py` | CLIP-based similarity scoring |
| `binary_dice_loss.py` | Binary Dice loss for segmentation |

### Configs (`configs/`)

| Config | Purpose |
|--------|---------|
| `test_layerdiff.yaml` | LayerDiff training config |
| `test_layerdiff3d.yaml` | LayerDiff 3D training config |
| `test_marigold_depth.yaml` | Marigold 2D depth config |
| `test_marigold3d.yaml` | Marigold 3D depth config |
| `test_depth.yaml` | Depth Anything adapter config |
| `test_vae.yaml` | Transparent VAE config |
| `test_partseg.yaml` | Body part segmentation config |
| `finetune_layerdiff_iter2.yaml` | LayerDiff fine-tuning with multi-source data |
| `test_ddp_4gpu.json` | Accelerate config for 4-GPU DeepSpeed |
| `ddp_bf16.json` | Accelerate config for 8-GPU DeepSpeed |

"""Run the author's paper-version Ours Full pipeline on an image folder."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "8"
os.environ["HF_HOME"] = str(ROOT / "workspace/.cache/huggingface")
os.environ["TORCH_HOME"] = str(ROOT / "workspace/.cache/torch")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
sys.path.insert(0, str(ROOT / "common"))
os.chdir(ROOT)


def input_fingerprint(source: Path) -> dict:
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'sha256': digest, 'recipe': 'ours-full-v0.0.1-seed42-layer1024x30-depth768-auto-v1'}


def matches_input(result_dir: Path, fingerprint: dict) -> bool:
    try:
        info = json.loads((result_dir / 'info.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return isinstance(info, dict) and info.get('inference_input') == fingerprint


def store_input_fingerprint(result_dir: Path, fingerprint: dict) -> None:
    path = result_dir / 'info.json'
    info = json.loads(path.read_text(encoding='utf-8'))
    info['inference_input'] = fingerprint
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stream-offload", action="store_true", help="Overlap UNet transfers and CUDA compute")
    parser.add_argument("--only", help="Run only the image with this exact filename")
    args = parser.parse_args()
    source_root = args.input_dir.resolve(strict=True)
    output_root = args.output_dir.resolve()
    if not source_root.is_dir():
        parser.error("--input-dir must be a directory")
    if output_root == source_root or source_root in output_root.parents:
        parser.error("The output directory must be outside the input directory")
    images = sorted(
        p for p in source_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        and (args.only is None or p.name == args.only)
    )
    if not images:
        parser.error("No input images found")
    destinations = [
        (p.relative_to(source_root).parent / p.stem.rstrip(" .")).as_posix().casefold()
        for p in images
    ]
    if len(destinations) != len(set(destinations)):
        parser.error("Input images with the same stem would overwrite each other's outputs")
    output_root.mkdir(parents=True, exist_ok=True)

    import torch
    from diffusers import DiffusionPipeline
    from PIL import Image
    from psd_tools import PSDImage
    from utils import inference_utils
    from utils.torch_utils import seed_everything

    torch.set_num_threads(8)
    if args.stream_offload:
        original_group_offload = DiffusionPipeline.enable_group_offload

        def streamed_group_offload(self, *positional, **keywords):
            keywords.update(
                use_stream=True, record_stream=True,
                exclude_modules=[key for key in self.components if key != "unet"],
            )
            return original_group_offload(self, *positional, **keywords)

        DiffusionPipeline.enable_group_offload = streamed_group_offload
    model_layer = str(ROOT / "workspace/models/layerdiff_v0.0.1")
    model_depth = str(ROOT / "workspace/models/marigold_v0.0.1")
    tags = inference_utils.VALID_BODY_PARTS_V2
    failures: list[str] = []
    started = time.perf_counter()
    print(f"BATCH_START images={len(images)} output={output_root}", flush=True)
    print("MODEL=v0.0.1 TAGS=19 RESOLUTION=1024 DEPTH=768 STEPS=30 SEED=42 GROUP_OFFLOAD=true", flush=True)
    print(f"STREAM_OFFLOAD={args.stream_offload}", flush=True)

    for index, source in enumerate(images, 1):
        relative = source.relative_to(source_root)
        save_dir = output_root / relative.parent
        safe_stem = source.stem.rstrip(" .")
        inference_source = source
        if safe_stem != source.stem:
            # Win32 strips trailing dots when creating directories. Use a local
            # read-only input alias so the author's basename-based paths agree.
            alias_dir = ROOT / "workspace/.batch_inputs/ours_full" / relative.parent
            alias_dir.mkdir(parents=True, exist_ok=True)
            inference_source = alias_dir / f"{safe_stem}{source.suffix}"
            if not inference_source.exists():
                os.link(source, inference_source)
            elif not inference_source.samefile(source):
                raise RuntimeError(f"Input alias already belongs to a different file: {inference_source}")
        result_dir = save_dir / safe_stem
        psd_path = save_dir / f"{safe_stem}.psd"
        depth_psd = save_dir / f"{safe_stem}_depth.psd"
        item_started = time.perf_counter()
        print(f"IMAGE_START {index}/{len(images)} {relative}", flush=True)
        try:
            fingerprint = input_fingerprint(source)
            input_matches = matches_input(result_dir, fingerprint)
            if not input_matches and (result_dir / 'info.json').is_file():
                # Invalidate the old identity before any PNG is overwritten.
                # An interrupted replacement must not validate the old cache.
                store_input_fingerprint(result_dir, {})
            layer_ready = input_matches and all((result_dir / f"{tag}.png").is_file() for tag in tags)
            depth_ready = input_matches and all((result_dir / f"{tag}_depth.png").is_file() for tag in tags)
            if not layer_ready:
                depth_ready = False
                print(f"STAGE {index}/{len(images)} layer_decomposition", flush=True)
                seed_everything(42)
                inference_utils.apply_layerdiff(
                    str(inference_source), model_layer, save_dir=str(save_dir),
                    seed=42, resolution=1024, num_inference_steps=30,
                    group_offload=True, disable_progressbar=True,
                )
                print(f"LAYER_DONE {index}/{len(images)} seconds={time.perf_counter() - item_started:.2f}", flush=True)
            if not depth_ready or not (result_dir / "reconstruction.png").is_file():
                print(f"STAGE {index}/{len(images)} pseudo_depth", flush=True)
                seed_everything(42)
                inference_utils.apply_marigold(
                    str(inference_source), model_depth, save_dir=str(save_dir),
                    seed=42, resolution=768, num_inference_steps=-1,
                    group_offload=True, disable_progressbar=True,
                )
            if not layer_ready or not depth_ready or not psd_path.is_file() or not depth_psd.is_file():
                print(f"STAGE {index}/{len(images)} psd_export", flush=True)
                inference_utils.further_extr(
                    str(result_dir), rotate=False, save_to_psd=True, tblr_split=False,
                )
            for tag in tags:
                for filename in (f"{tag}.png", f"{tag}_depth.png"):
                    with Image.open(result_dir / filename) as image:
                        image.verify()
            with Image.open(result_dir / "reconstruction.png") as image:
                image.verify()
            psd = PSDImage.open(psd_path)
            depth_document = PSDImage.open(depth_psd)
            store_input_fingerprint(result_dir, fingerprint)
            print(
                f"IMAGE_DONE {index}/{len(images)} seconds={time.perf_counter() - item_started:.2f} "
                f"psd_layers={len(psd)} depth_layers={len(depth_document)} "
                f"peak_vram_gib={torch.cuda.max_memory_allocated() / 2**30:.3f} {relative}",
                flush=True,
            )
            del psd, depth_document
        except Exception:
            failures.append(str(relative))
            print(f"IMAGE_FAILED {index}/{len(images)} {relative}", flush=True)
            traceback.print_exc()
            # Reinitialise after a failure so one input cannot poison the rest.
            inference_utils.layerdiff_pipeline = None
            inference_utils.marigold_pipeline = None
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    print(
        f"BATCH_DONE completed={len(images) - len(failures)} failed={len(failures)} "
        f"seconds={time.perf_counter() - started:.2f}", flush=True,
    )
    for failed in failures:
        print(f"FAILED_INPUT {failed}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

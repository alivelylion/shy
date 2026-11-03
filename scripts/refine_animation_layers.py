"""Refine existing Ours Full outputs without rerunning diffusion inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "8"
os.environ["TORCH_HOME"] = str(ROOT / "workspace/.cache/torch")
os.environ["HF_HOME"] = str(ROOT / "workspace/.cache/huggingface")
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / "workspace/.cache/ultralytics")
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ["YOLO_OFFLINE"] = "true"
sys.path[:0] = [str(ROOT), str(ROOT / "common")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Optional separate output root; default writes into each original result's animation subfolder")
    parser.add_argument("--pose-model", type=Path, default=ROOT / "workspace/models/pose/yolo11s-pose.pt")
    parser.add_argument("--only", nargs="+", help="Only process result subdirectories with these prefixes")
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--example-rules", action="store_true", help="Opt into filename-specific rules for the previously reviewed examples")
    parser.add_argument("--no-lama", action="store_true", help="Nearest-material preview only, without learned inpainting")
    args = parser.parse_args()
    source = args.input_dir.resolve(strict=True)
    output = args.output_dir.resolve() if args.output_dir else None
    if output == source:
        parser.error("Omit --output-dir to refine inside the original result folders")
    if (source / "src_img.png").is_file():
        items = [source]
    else:
        items = sorted(p.parent for p in source.rglob("src_img.png") if (p.parent / "info.json").is_file())
    if args.only:
        items = [p for p in items if p.name.startswith(tuple(args.only))]
    if not items:
        parser.error("No Ours Full outputs found")
    from utils.animation_parts import refine_for_animation
    import torch
    torch.set_num_threads(8)
    failures = 0
    print(f"REFINE_START images={len(items)} output={output}", flush=True)
    for i, item in enumerate(items, 1):
        started = time.perf_counter()
        destination = output / (item.relative_to(source) if item != source else Path(item.name)) if output else item / "动画拆层"
        print(f"IMAGE_START {i}/{len(items)} {item.name}", flush=True)
        try:
            info = refine_for_animation(item, destination, args.pose_model, use_lama=not args.no_lama,
                                        previews=not args.no_previews, use_example_rules=args.example_rules)
            print(f"IMAGE_DONE {i}/{len(items)} layers={len(info.get('layers', []))} cloth={len(info['cloth_splits'])} arms={len(info['arms'])} legs={len(info['legs'])} reused={info.get('reused',False)} unchanged={info.get('unchanged',False)} seconds={time.perf_counter()-started:.1f} warnings={info['warnings']}", flush=True)
        except Exception:
            failures += 1
            traceback.print_exc()
    print(f"REFINE_DONE completed={len(items)-failures} failed={failures}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())

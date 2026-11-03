"""Audit every delivered character against source foreground; optionally repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "8"
os.environ["TORCH_HOME"] = str(ROOT / "workspace/.cache/torch")
sys.path[:0] = [str(ROOT), str(ROOT / "common")]


def verify_export(src, target, layers: list, output: Path) -> dict:
    """Read the deliverables themselves; in-memory coverage is insufficient."""
    import numpy as np
    from PIL import Image
    from psd_tools import PSDImage
    from utils.animation_parts import composite, _on_checker
    from utils.animation_integrity import measure_completeness

    errors = []
    coverage = measure_completeness(target, layers)
    if coverage["missing_pixels"]:
        errors.append(f"missing_foreground={coverage['missing_pixels']}")
    rendered = composite(layers)
    expected = src.copy()
    expected[~target] = 0
    if not np.array_equal(rendered[..., 3], expected[..., 3]):
        errors.append("composite_alpha")
    maximum = int(np.abs(rendered[..., :3].astype(int)-expected[..., :3].astype(int))[target].max()) if target.any() else 0
    if maximum > 1:
        errors.append(f"composite_rgb={maximum}")
    names = [part.name for part in layers]
    if len(names) != len(set(names)):
        errors.append("duplicate_layer_names")
    parents = {part.name: part.parent for part in layers}
    for part in layers:
        if part.rgba.shape != src.shape or part.depth.shape != target.shape:
            errors.append(f"canvas:{part.name}")
        if part.completed is not None and part.completed.shape != target.shape:
            errors.append(f"completion_canvas:{part.name}")
        if part.pivot is not None and (len(part.pivot) != 2 or not np.isfinite(part.pivot).all()):
            errors.append(f"pivot:{part.name}")
        seen = {part.name}
        parent = part.parent
        while parent:
            if parent not in parents or parent in seen:
                errors.append(f"parent_hierarchy:{part.name}")
                break
            seen.add(parent)
            parent = parents[parent]
    psd = PSDImage.open(output / "动画拆层.psd")
    if psd.size != (src.shape[1], src.shape[0]):
        errors.append("psd_canvas")
    exported = {part.name: part for part in psd}
    if len(psd) != len(layers) or set(exported) != set(names):
        errors.append("psd_layer_names")
    for part in layers:
        if part.name not in exported:
            continue
        item = exported[part.name]
        png = np.asarray(Image.fromarray(part.rgba).crop(item.bbox))
        actual = np.asarray(item.topil().convert("RGBA"))
        if png.shape != actual.shape or not np.array_equal(actual, png):
            errors.append(f"psd_png_rgba:{part.name}")
    preview = output / "静态合成.png"
    if not preview.is_file() or not np.array_equal(np.asarray(Image.open(preview).convert("RGB")), np.asarray(_on_checker(rendered, 1024))):
        errors.append("static_preview")
    if not (output / "部件总览.png").is_file():
        errors.append("parts_overview")
    return {"errors": errors, "maximum_visible_rgb_error": maximum, "checked_psd_layers": len(psd), **coverage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    source, output = args.input_dir.resolve(strict=True), args.output_dir.resolve(strict=True)
    if source == output:
        parser.error("Preserve the original Ours Full outputs")
    from utils.animation_parts import load_layers, load_refined, export_layers, save_previews
    from utils.animation_integrity import ensure_complete, foreground_mask, read_hints, measure_completeness
    import torch
    torch.set_num_threads(8)
    failures = 0
    inputs = sorted(p for p in source.iterdir() if p.is_dir() and (p / "info.json").exists())
    if args.only:
        inputs = [p for p in inputs if p.name.startswith(tuple(args.only))]
    print(f"INTEGRITY_START images={len(inputs)} repair={args.repair}", flush=True)
    for index, item in enumerate(inputs, 1):
        started = time.perf_counter()
        print(f"CHECK_START {index}/{len(inputs)} {item.name}", flush=True)
        src, originals = load_layers(item)
        cached = load_refined(output / item.name)
        if cached is None:
            layers = originals
            info = {"input":str(item),"arms":[],"legs":[],"instances":[],"cloth_splits":[],"warnings":[]}
        else:
            layers, info = cached
        hints, _ = read_hints(item.name)
        target, basis = foreground_mask(src, originals, hints)
        before = measure_completeness(target, layers)
        checked = verify_export(src, target, layers, output/item.name)
        if args.repair and checked["errors"]:
            print(f"REPAIR {item.name} errors={checked['errors']}", flush=True)
            layers, info = ensure_complete(src, originals, layers, info, item)
            info["refinement_version"] = max(5, info.get("refinement_version", 0))
            for key in ("arms","legs","instances","cloth_splits","warnings"):
                info.setdefault(key, [])
            info.pop("unchanged", None)
            export_layers(layers, output/item.name, info)
            save_previews(layers, output/item.name)
            layers, info = load_refined(output/item.name)
            checked = verify_export(src, target, layers, output/item.name)
        failures += int(bool(checked["errors"]))
        if not checked["errors"] and args.repair:
            info.setdefault("integrity", {}).update({"source_sha256":hashlib.sha256(src.tobytes()).hexdigest(),
                                                      "disk_rgba_and_psd_layers_verified":True})
            (output/item.name/"关节与图层.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"CHECK_DONE {index}/{len(inputs)} before={before['coverage']:.6f} after={checked['coverage']:.6f} missing={checked['missing_pixels']} rgb_error={checked['maximum_visible_rgb_error']} psd_layers={checked['checked_psd_layers']} errors={checked['errors']} basis={basis} seconds={time.perf_counter()-started:.1f}", flush=True)
    motion_files = list(output.rglob("*.gif"))+list(output.rglob("*_最大幅度.png"))
    failures += int(bool(motion_files))
    print(f"MOTION_FILES {len(motion_files)}", flush=True)
    print(f"INTEGRITY_DONE checked={len(inputs)} incomplete={failures}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())

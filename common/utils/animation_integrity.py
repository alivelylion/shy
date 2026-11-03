"""Source-aware completeness checks for animation deliverables.

Completeness means preserving the observed foreground, not proving anatomical
segmentation or animation quality. Source-assisted repairs are identified in the
existing layer manifest rather than being attributed to the author's model.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


REVISION = 1
PROJECT = Path(__file__).resolve().parents[2]


def read_hints(name: str, *, use_example_rules: bool = True) -> tuple[dict, str]:
    path = PROJECT / "workspace/animation_integrity_hints.json"
    data = path.read_bytes() if use_example_rules and path.is_file() else b"{}"
    hints = json.loads(data).get(name, {})
    digest = hashlib.sha256(json.dumps(hints, sort_keys=True).encode()).hexdigest()[:16]
    return hints, digest


def polygon(shape: tuple[int, int], points: list[list[float]]) -> np.ndarray:
    result = np.zeros(shape, np.uint8)
    cv2.fillPoly(result, [np.round(points).astype(np.int32)], 1)
    return result > 0


def foreground_mask(src: np.ndarray, originals: list, hints: dict) -> tuple[np.ndarray, str]:
    alpha = src[..., 3] > 10
    x, y, w, h = cv2.boundingRect(alpha.astype(np.uint8))
    rectangular = alpha.sum()/max(w*h, 1) > 0.98
    if hints.get("foreground_polygon"):
        initial = polygon(alpha.shape, hints["foreground_polygon"]) & alpha
        if hints.get("refine_boundary", False):
            # Refine only a narrow band. Colour alone cannot separate Doraemon's
            # blue body from a blue background.
            mask = np.full(alpha.shape, cv2.GC_BGD, np.uint8)
            mask[cv2.dilate(initial.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0] = cv2.GC_PR_BGD
            mask[initial] = cv2.GC_PR_FGD
            mask[cv2.erode(initial.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = cv2.GC_FGD
            cv2.setRNGSeed(0)
            cv2.grabCut(src[..., :3].copy(), mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
            initial = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)) & alpha
        return initial, "reviewed_foreground_outline"
    if not rectangular:
        return alpha, "source_transparency"
    union = np.logical_or.reduce([p.mask for p in originals])
    return union & alpha, "author_foreground_for_opaque_source"


def _source_instances(src: np.ndarray, target: np.ndarray, layers: list, hints: dict, use_lama: bool) -> tuple[list, dict]:
    from utils.animation_parts import Layer, _complete_rgb, split_limb_chains
    events = {"instances": [], "arms": [], "legs": []}
    for instance in hints.get("source_instances", []):
        retained = [(p.name, p.mask.copy()) for p in layers if p.name in instance.get("retain_tags", [])]
        scope = target.copy()
        if instance.get("box"):
            scope[:] = False
            x1, y1, x2, y2 = instance["box"]
            scope[y1:y2, x1:x2] = target[y1:y2, x1:x2]
        if instance.get("replace_all", False):
            layers = []
        else:
            for layer in layers:
                layer.rgba[scope] = 0
                if layer.completed is not None:
                    layer.completed[scope] = False
        definitions = instance.get("regions", [])
        chains = instance.get("chains", [])
        slots: list[dict] = []
        for region in definitions:
            mask = polygon(target.shape, region["polygon"]) & scope if region.get("polygon") else scope.copy()
            slots.append({"definition": region, "mask": mask, "kind": "region"})
        for chain in chains:
            mask = polygon(target.shape, chain["polygon"]) & scope
            slots.append({"definition": chain, "mask": mask, "kind": chain["kind"]})
        for old_name, old_mask in retained:
            mask = old_mask & scope
            if 4 <= mask.sum() < scope.sum()*0.2:
                slots.append({"definition": {"name":old_name,"role":"original","parent":instance.get("head_name"),
                                               "label":old_name}, "mask":mask, "kind":"region"})
        owner = np.full(target.shape, -1, np.int16)
        for i, slot in enumerate(slots):
            owner[slot["mask"]] = i
        pieces = []
        for i, slot in enumerate(slots):
            definition = slot["definition"]
            known = owner == i
            if known.sum() < 4:
                continue
            depth_value = int(230-(i+1)*190/max(len(slots), 1))
            depth = np.full(target.shape, depth_value, np.uint8)
            name = definition["name"]
            if slot["kind"] == "region":
                support = slot["mask"] if definition.get("complete", False) else known
                rgba, completed = _complete_rgb(src, known, support, use_lama)
                pieces.append(Layer(name, rgba, depth, f"source:{instance['name']}",
                                    pivot=definition.get("pivot"), parent=definition.get("parent"),
                                    role=definition.get("role", "source_region"), completed=completed,
                                    meta={"label": definition.get("label", name), "provenance": "reviewed_source_regions"}))
            else:
                rgba = src.copy(); rgba[~known] = 0
                limb = Layer(name, rgba, depth, f"source:{instance['name']}")
                split, limb_events = split_limb_chains(limb, [dict(definition)], slot["kind"], use_lama)
                for piece in split:
                    if piece.role in {"arm_upper", "leg_thigh"}:
                        piece.parent = instance.get("body_name")
                    piece.meta["provenance"] = "reviewed_source_regions"
                pieces.extend(split)
                events["arms" if slot["kind"] == "arm" else "legs"].extend(limb_events)
        # Do not silently discard a pixel that falls outside reviewed part polygons.
        remaining = scope & (owner < 0)
        if remaining.any():
            rgba = src.copy(); rgba[~remaining] = 0
            pieces.append(Layer(f"{instance['name']}_source_remainder", rgba, np.full(target.shape, 245, np.uint8),
                                f"source:{instance['name']}", role="source_unclassified",
                                meta={"label": "原图其余部件", "needs_joint_review": True}))
        layers.extend(pieces)
        events["instances"].append({"name": instance["name"], "parts": len(pieces), "method": "source_regions_and_author_LaMa",
                                    "reviewed_regions": True})
    return [p for p in layers if p.mask.any()], events


def _trim_full_character_duplicates(layers: list, target: np.ndarray) -> list[str]:
    """A VAE 'footwear' prediction can accidentally repeat the entire character."""
    removed = []
    feet = [p for p in layers if p.role == "leg_foot"]
    for layer in layers:
        if layer.source == "footwear" and layer.mask.sum() > target.sum()*0.45 and feet:
            support = np.logical_or.reduce([p.mask for p in feet])
            support = cv2.dilate(support.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            discarded = layer.mask & ~support
            if discarded.any():
                layer.rgba[discarded] = 0
                removed.append(layer.name)
    if any(p.role in {"arm_forearm", "leg_shin"} for p in layers):
        for layer in layers:
            if layer.source != "objects" or layer.mask.sum() < target.sum()*0.8:
                continue
            other = [p.mask for p in layers if p is not layer and p.source != "objects"]
            covered = np.logical_or.reduce(other) if other else np.zeros_like(target)
            if (covered & layer.mask).sum() > layer.mask.sum()*0.85:
                layer.rgba[covered] = 0
                removed.append(layer.name)
    return removed


def _owner_map(layers: list, target: np.ndarray) -> np.ndarray:
    h, w = target.shape
    owners = np.full((h, w), -1, np.int32)
    boxes = [cv2.boundingRect(p.rgba[..., 3]) for p in layers]
    count = len(layers)
    for y in range(0, h, 128):
        for x in range(0, w, 128):
            y2, x2 = min(h, y+128), min(w, x+128)
            if not target[y:y2, x:x2].any():
                continue
            ids = [i for i, (bx, by, bw, bh) in enumerate(boxes) if bw and bh and bx < x2 and bx+bw > x and by < y2 and by+bh > y]
            if not ids:
                continue
            costs = np.stack([np.where(layers[i].rgba[y:y2, x:x2, 3] > 10,
                                      layers[i].depth[y:y2, x:x2].astype(np.int32)*(count+1)+(count-i),
                                      2**30) for i in ids])
            best = np.argmin(costs, axis=0)
            local = np.asarray(ids, np.int32)[best]
            local[costs.min(axis=0) == 2**30] = -1
            local[~target[y:y2, x:x2]] = -1
            owners[y:y2, x:x2] = local
    return owners


def measure_completeness(target: np.ndarray, layers: list) -> dict:
    union = np.logical_or.reduce([p.mask for p in layers]) if layers else np.zeros_like(target)
    missing = target & ~union
    n, _, stats, _ = cv2.connectedComponentsWithStats(missing.astype(np.uint8), 8)
    return {"foreground_pixels": int(target.sum()), "missing_pixels": int(missing.sum()),
            "coverage": float(1-missing.sum()/max(target.sum(), 1)),
            "largest_missing_component": int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0}


def ensure_complete(src: np.ndarray, originals: list, layers: list, metadata: dict, source: Path,
                    use_lama: bool = True, *, use_example_rules: bool = False) -> tuple[list, dict]:
    from utils.animation_parts import Layer, composite
    hints, digest = read_hints(source.name, use_example_rules=use_example_rules)
    target, basis = foreground_mask(src, originals, hints)
    before = measure_completeness(target, layers)
    layers, events = _source_instances(src, target, layers, hints, use_lama)
    duplicates = _trim_full_character_duplicates(layers, target)
    for layer in layers:
        layer.rgba[~target] = 0
        if layer.completed is not None:
            layer.completed &= target
    union = np.logical_or.reduce([p.mask for p in layers]) if layers else np.zeros_like(target)
    missing = target & ~union
    restored = int(missing.sum())
    if missing.any():
        # Exact observed pixels are retained even where the author missed an ear,
        # accessory, robot torso or an entire body. This layer is explicitly
        # unclassified, and is not counted as a successful anatomical split.
        existing = next((p for p in layers if p.name == "source_visible_remainder"), None)
        if existing is not None:
            existing.rgba[missing] = src[missing]
            existing.depth[missing] = 248
        else:
            rgba = src.copy(); rgba[~missing] = 0
            layers.append(Layer("source_visible_remainder", rgba, np.full(target.shape, 248, np.uint8),
                                "source", role="source_unclassified", meta={"label": "原图未分类可见部件", "needs_joint_review": True}))
    layers = [p for p in layers if p.mask.any()]
    owners = _owner_map(layers, target)
    if (owners[target] < 0).any():
        raise ValueError("Foreground ownership is incomplete")
    edge = target & (src[..., 3] < 255)
    for i, layer in enumerate(layers):
        selected = owners == i
        layer.rgba[selected] = src[selected]
        layer.rgba[edge & ~selected, 3] = 0
    after = measure_completeness(target, layers)
    if after["missing_pixels"]:
        raise ValueError(f"Foreground still incomplete: {after}")
    rendered = composite(layers)
    # Mask coverage alone is insufficient: verify the actual rendered RGBA.
    expected = src.copy(); expected[~target] = 0
    if not np.array_equal(rendered[..., 3], expected[..., 3]):
        raise ValueError("Rest-pose alpha differs from the observed foreground")
    rgb_error = np.abs(rendered[..., :3].astype(int)-expected[..., :3].astype(int))
    maximum = int(rgb_error[target].max()) if target.any() else 0
    if maximum > 1:
        raise ValueError(f"Rest-pose source pixel error {maximum}")
    if events["instances"]:
        metadata["source_repairs"] = events["instances"]
        if hints.get("replace_all_metadata", False):
            metadata["arms"] = events["arms"]
            metadata["legs"] = events["legs"]
        else:
            metadata["source_repair_arms"] = events["arms"]
            metadata["source_repair_legs"] = events["legs"]
    unclassified = [p.name for p in layers if p.role == "source_unclassified"]
    warnings = [w for w in metadata.get("warnings", []) if not w.startswith("Visible source regions")]
    if unclassified:
        warnings.append("Visible source regions retained without verified joint segmentation: "+", ".join(unclassified))
    metadata["warnings"] = warnings
    metadata["completion_revision"] = max(2, metadata.get("completion_revision", 0))
    metadata["integrity"] = {"revision": REVISION, "hints_digest": digest, "foreground_basis": basis,
                             "source_sha256": hashlib.sha256(src.tobytes()).hexdigest(),
                             "before": before, "after": after, "source_remainder_pixels": restored,
                             "rest_pose_rgba_checked": True, "maximum_visible_rgb_error": maximum,
                             "trimmed_duplicate_layers": duplicates,
                             "no_unclassified_foreground": not bool(unclassified),
                             "joint_segmentation_verified": False}
    return layers, metadata

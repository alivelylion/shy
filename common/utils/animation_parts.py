"""Optional motion-oriented refinement of the author's RGBA/depth layers.

Depth and pose are estimates. Uncertain splits are retained as original layers;
completion masks explicitly identify inferred pixels. This is not a Spine rig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize


PROJECT = Path(__file__).resolve().parents[2]


@dataclass(eq=False)
class Layer:
    name: str
    rgba: np.ndarray
    depth: np.ndarray
    source: str
    pivot: list[float] | None = None
    parent: str | None = None
    role: str = "original"
    completed: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
    visibility_mask: np.ndarray | None = None

    @property
    def mask(self) -> np.ndarray:
        return self.rgba[..., 3] > 10

    @property
    def z(self) -> float:
        return float(np.median(self.depth[self.mask])) if self.mask.any() else 255.0


def _components(mask: np.ndarray, minimum: int = 32) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum
    return keep[labels]


def _bbox(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int]:
    y, x = np.where(mask)
    if not len(x):
        return (0, 0, 0, 0)
    h, w = mask.shape
    return max(0, int(x.min())-pad), max(0, int(y.min())-pad), min(w, int(x.max())+pad+1), min(h, int(y.max())+pad+1)


def _cut(layer: Layer, mask: np.ndarray, name: str, **kwargs: object) -> Layer:
    rgba = layer.rgba.copy()
    rgba[~mask] = 0
    return Layer(name, rgba, layer.depth.copy(), layer.source, **kwargs)


def load_layers(source: Path) -> tuple[np.ndarray, list[Layer]]:
    src = np.array(Image.open(source / "src_img.png").convert("RGBA"))
    info = json.loads((source / "info.json").read_text(encoding="utf-8"))
    layers: list[Layer] = []
    for tag in info["parts"]:
        p = source / f"{tag}.png"
        d = source / f"{tag}_depth.png"
        if not p.is_file() or not d.is_file():
            raise FileNotFoundError(f"Missing RGBA/depth pair: {p}")
        rgba = np.array(Image.open(p).convert("RGBA"))
        depth = np.array(Image.open(d).convert("L"))
        if rgba.shape != src.shape or depth.shape != src.shape[:2]:
            raise ValueError(f"Expected full-canvas input layers: {p}")
        # The transparent VAE leaves a faint alpha=1 frame/background.
        clean = _components(rgba[..., 3] > 10, minimum=20)
        rgba[~clean] = 0
        if clean.sum() >= 24:
            layers.append(Layer(tag, rgba, depth, tag))
    return src, layers


def _two_surfaces(layer: Layer) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Require separated depth modes, coherent regions and distinct appearance."""
    mask = layer.mask
    valid = (layer.rgba[..., 3] >= 128) & (layer.depth < 245)
    values = layer.depth[valid]
    if len(values) < 500 or np.ptp(values) < 22:
        return None
    threshold, _ = cv2.threshold(values[:, None], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    groups = [valid & (layer.depth <= threshold), valid & (layer.depth > threshold)]
    if min(g.sum() for g in groups) < max(160, 0.065*valid.sum()):
        return None
    medians = [float(np.median(layer.depth[g])) for g in groups]
    spreads = [float(np.percentile(layer.depth[g], 85)-np.percentile(layer.depth[g], 15)) for g in groups]
    gap = medians[1]-medians[0]
    if gap < 22 or max(spreads) > gap*0.65:
        return None
    lab = cv2.cvtColor(layer.rgba[..., :3], cv2.COLOR_RGB2LAB)
    colors = [np.median(lab[g], axis=0) for g in groups]
    if np.linalg.norm(colors[0]-colors[1]) < 15:
        return None
    seeds = [_components(g, max(40, int(valid.sum()*0.003))) for g in groups]
    if not all(g.any() for g in seeds):
        return None
    # Alpha boundaries with white depth are assigned to the nearest confident surface.
    d0 = ndi.distance_transform_edt(~seeds[0])
    d1 = ndi.distance_transform_edt(~seeds[1])
    near = mask & (d0 <= d1)
    far = mask & ~near
    # Reflected/antialiased exterior edges must not contaminate the lining context.
    color0 = np.linalg.norm(lab.astype(float)-colors[0], axis=-1)
    color1 = np.linalg.norm(lab.astype(float)-colors[1], axis=-1)
    exterior_edge = far & (color0+8 < color1)
    near |= exterior_edge
    far &= ~exterior_edge
    x1, y1, x2, y2 = _bbox(mask)
    fy, fx = np.where(far)
    # A tiny trim or belt is not evidence for an independently moving inner cloth.
    if fy.max()-fy.min() < (y2-y1)*0.45 or fx.max()-fx.min() < (x2-x1)*0.22:
        return None
    return near, far, {"depth_centers": medians, "depth_gap": gap, "rule": "depth_modes_color_spatial_support"}


def _complete_rgb(rgba: np.ndarray, known: np.ndarray, support: np.ndarray, use_lama: bool) -> tuple[np.ndarray, np.ndarray]:
    """Complete only a bounded support; preserve every known RGBA pixel exactly."""
    unknown = support & ~known
    result = rgba.copy()
    result[~support] = 0
    if not unknown.any() or not known.any():
        return result, unknown
    x1, y1, x2, y2 = _bbox(support | known, 24)
    sl = np.s_[y1:y2, x1:x2]
    local_known = known[sl]
    local_unknown = unknown[sl]
    trusted = local_known & (rgba[sl][..., 3] >= 220)
    core = cv2.erode(trusted.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if core.sum() >= 12:
        trusted = core
    if not trusted.any():
        trusted = local_known
    nearest = ndi.distance_transform_edt(~trusted, return_distances=False, return_indices=True)
    # Use the target material as context, avoiding exterior cloth leaking into its lining.
    rgb = rgba[sl][..., :3][tuple(nearest)].copy()
    rgb[trusted] = rgba[sl][..., :3][trusted]
    if use_lama and local_unknown.sum() >= 16:
        from annotators.lama_inpainter import apply_inpaint
        rgb = apply_inpaint(rgb, local_unknown.astype(np.uint8)*255, precision=__import__("torch").float32)
    target = result[sl]
    target[local_unknown, :3] = rgb[local_unknown]
    target[local_unknown, 3] = 255
    result[known & support] = rgba[known & support]
    return result, unknown


def split_cloth(layer: Layer, use_lama: bool = True) -> tuple[list[Layer], dict] | None:
    separated = _two_surfaces(layer)
    if separated is None:
        return None
    near, far, evidence = separated
    # Fill within the existing garment silhouette; never a full rectangular alpha.
    support = ndi.binary_fill_holes(layer.mask)
    for y in np.flatnonzero(support.any(axis=1)):
        xs = np.flatnonzero(support[y])
        support[y, xs[0]:xs[-1]+1] = True
    outer = _cut(layer, near, f"{layer.name}_outer", role="cloth_outer")
    inner_rgba, completion = _complete_rgb(layer.rgba, far, support, use_lama)
    inner_depth = layer.depth.copy()
    inner_depth[completion] = round(evidence["depth_centers"][1])
    inner = Layer(f"{layer.name}_inner", inner_rgba, inner_depth, layer.source, role="cloth_inner", completed=completion)
    x1, y1, x2, _ = _bbox(support)
    outer.pivot = [(x1+x2)/2, float(y1)]
    inner.pivot = outer.pivot.copy()
    inner.meta.update(evidence)
    outer.meta.update(evidence)
    return [inner, outer], {**evidence, "source": layer.name, "inferred_pixels": int(completion.sum())}


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b-a
    t = np.clip(((points-a)*ab).sum(axis=-1)/max(float(ab@ab), 1.0), 0, 1)
    return np.linalg.norm(points-(a+t[..., None]*ab), axis=-1)


_pose_models: dict[str, object] = {}


def detect_poses(src: np.ndarray, model: Path) -> list[np.ndarray]:
    from ultralytics import YOLO
    key = str(model.resolve())
    if key not in _pose_models:
        _pose_models[key] = YOLO(key)
    result = _pose_models[key].predict(src[..., :3][..., ::-1], imgsz=1024, device=0, conf=0.3, iou=0.35, verbose=False)[0]
    if result.keypoints is None:
        return []
    return [k for k in result.keypoints.data.cpu().numpy() if (k[5:11, 2] > 0.4).sum() >= 4]


def _snap_joint(point: np.ndarray, skeleton: np.ndarray, radius: float) -> np.ndarray:
    y, x = np.where(skeleton)
    if not len(x):
        return point.copy()
    points = np.column_stack([x, y])
    distance = np.linalg.norm(points-point, axis=1)
    return points[distance.argmin()].astype(float) if distance.min() <= radius else point.copy()


def _depth_groups(mask: np.ndarray, depth: np.ndarray, min_gap: float) -> tuple[np.ndarray, np.ndarray] | None:
    core = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    core &= depth < 240
    if core.sum() < 160:
        return None
    threshold, _ = cv2.threshold(depth[core, None], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    seeds = [core & (depth <= threshold), core & (depth > threshold)]
    if min(s.sum() for s in seeds) < max(24, core.sum()*0.07):
        return None
    centers = [float(np.median(depth[s])) for s in seeds]
    gap = centers[1]-centers[0]
    spreads = [float(np.percentile(depth[s], 85)-np.percentile(depth[s], 15)) for s in seeds]
    if gap < min_gap or max(spreads) > gap*0.8:
        return None
    d0, d1 = [ndi.distance_transform_edt(~s) for s in seeds]
    first = mask & (d0 <= d1)
    return first, mask & ~first


def _arm_masks(mask: np.ndarray, joints: np.ndarray, depth: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, float] | None:
    shoulder, elbow, wrist = joints[:, :2].copy()
    length = float(np.linalg.norm(shoulder-elbow)+np.linalg.norm(elbow-wrist))
    if length < 28:
        return None
    skeleton = skeletonize(mask)
    shoulder = _snap_joint(shoulder, skeleton, length*0.18)
    elbow = _snap_joint(elbow, skeleton, length*0.18)
    widths = ndi.distance_transform_edt(mask)
    ex, ey = np.round(elbow).astype(int)
    radius = float(np.clip(widths[ey, ex], 3, length*0.12))
    yy, xx = np.indices(mask.shape)
    disk = (xx-elbow[0])**2+(yy-elbow[1])**2 <= (radius*1.4)**2
    branch, _ = ndi.label(skeleton & ~disk, structure=np.ones((3, 3)))
    sy, sx = np.where(skeleton & ~disk)
    if len(sx) < 4:
        return None
    sp = np.column_stack([sx, sy])
    upper_id = branch[sy[np.linalg.norm(sp-shoulder, axis=1).argmin()], sx[np.linalg.norm(sp-shoulder, axis=1).argmin()]]
    upper_path = branch == upper_id
    distal = skeleton & ~disk & ~upper_path
    if distal.any():
        wrist = _snap_joint(wrist, distal, length*0.25)
    else:
        wrist = _snap_joint(wrist, skeleton, length*0.18)
    if np.linalg.norm(elbow-wrist) < 10:
        return None
    points = np.stack([xx, yy], axis=-1)
    if distal.any():
        upper = mask & (ndi.distance_transform_edt(~upper_path) <= ndi.distance_transform_edt(~distal))
    else:
        upper = mask & (_segment_distance(points, shoulder, elbow) <= _segment_distance(points, elbow, wrist))
    # Bent arms can touch at the shoulder/hand, closing the skeleton into a loop.
    # A strong depth discontinuity then supplies the missing upper/forearm boundary.
    depth_split = _depth_groups(mask, depth, 35)
    if depth_split is not None:
        distances = [ndi.distance_transform_edt(~g) for g in depth_split]
        sx, sy = np.clip(np.round(shoulder).astype(int), [0, 0], [mask.shape[1]-1, mask.shape[0]-1])
        upper_index = int(distances[1][sy, sx] < distances[0][sy, sx])
        depth_upper = depth_split[upper_index]
        if (depth_upper & upper).sum() > depth_upper.sum()*0.70:
            upper = _components(depth_upper, max(40, int(mask.sum()*0.01)))
    fore_and_hand = mask & ~upper
    wrist = _snap_joint(joints[2, :2], skeletonize(fore_and_hand), length*0.25)
    axis = (wrist-elbow)/np.linalg.norm(wrist-elbow)
    beyond_wrist = ((points-wrist)*axis).sum(axis=-1) > 0
    hand = fore_and_hand & beyond_wrist
    forearm = fore_and_hand & ~hand
    distal_split = _depth_groups(fore_and_hand, depth, 12)
    if distal_split is not None:
        # Hand depth is useful only if the proposed component is distal to the elbow.
        distal_axis = (wrist-elbow)/np.linalg.norm(wrist-elbow)
        scores = [float(np.mean(((points[g]-elbow)*distal_axis).sum(axis=-1))) for g in distal_split]
        hand_index = int(scores[1] > scores[0])
        proposed_hand = distal_split[hand_index]
        if (proposed_hand & beyond_wrist).sum() > proposed_hand.sum()*0.45:
            hand, forearm = proposed_hand, distal_split[1-hand_index]
            boundary = (cv2.dilate(hand.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & forearm
            by, bx = np.where(boundary & (np.linalg.norm(points-wrist, axis=-1) < length*0.25))
            if len(bx) >= 2:
                wrist = np.array([float(np.mean(bx)), float(np.mean(by))])
    if min(upper.sum(), forearm.sum()) < max(32, mask.sum()*0.06):
        return None
    # Rounded overlapping joint caps, retained beneath the neighbouring segment.
    elbow_cap = (xx-elbow[0])**2+(yy-elbow[1])**2 <= (radius*1.08)**2
    wx, wy = np.clip(np.round(wrist).astype(int), [0, 0], [mask.shape[1]-1, mask.shape[0]-1])
    wr = float(np.clip(widths[wy, wx], 2, radius))
    wrist_cap = (xx-wrist[0])**2+(yy-wrist[1])**2 <= (wr*1.05)**2
    upper |= elbow_cap
    forearm |= elbow_cap
    if hand.sum() >= 32:
        forearm |= wrist_cap
        hand |= wrist_cap
    else:
        forearm |= hand
        hand[:] = False
    return [upper, forearm, hand], np.stack([shoulder, elbow, wrist]), radius


def split_arms(layer: Layer, poses: list[np.ndarray], use_lama: bool) -> tuple[list[Layer], list[dict]]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(layer.mask.astype(np.uint8), 8)
    pieces: list[Layer] = []
    events: list[dict] = []
    used = np.zeros(layer.mask.shape, bool)
    candidates: list[tuple[float, int, int, str, np.ndarray]] = []
    for person, pose in enumerate(poses):
        for side, ids in (("left", [5, 7, 9]), ("right", [6, 8, 10])):
            joints = pose[ids]
            if joints[:, 2].min() < 0.5:
                continue
            for label in range(1, n):
                if stats[label, cv2.CC_STAT_AREA] < 120:
                    continue
                comp = labels == label
                y, x = np.where(comp)
                points = np.column_stack([x[::4], y[::4]])
                distances = np.minimum(_segment_distance(points, joints[0, :2], joints[1, :2]), _segment_distance(points, joints[1, :2], joints[2, :2]))
                length = np.linalg.norm(joints[0, :2]-joints[1, :2])+np.linalg.norm(joints[1, :2]-joints[2, :2])
                joint_dist = ndi.distance_transform_edt(~comp)
                coords = np.round(joints[:, :2]).astype(int)
                if (coords < 0).any() or (coords[:, 0] >= comp.shape[1]).any() or (coords[:, 1] >= comp.shape[0]).any():
                    continue
                offset = joint_dist[coords[:, 1], coords[:, 0]]
                if offset.max() > max(16, length*0.14) or np.median(distances) > max(12, length*0.18):
                    continue
                score = float((np.median(distances)+offset.mean())/max(float(length), 1.0))
                candidates.append((score, label, person, side, joints))
    consumed: set[int] = set()
    assigned: set[tuple[int, str]] = set()
    for score, label, person, side, joints in sorted(candidates, key=lambda c: c[0]):
        if label in consumed or (person, side) in assigned:
            continue
        comp = labels == label
        split = _arm_masks(comp, joints, layer.depth)
        if split is None:
            continue
        masks, joints_xy, radius = split
        prefix = f"arm_{side}" if len(poses) == 1 else f"person{person+1}_arm_{side}"
        for i, (name, support) in enumerate(zip(("upper", "forearm", "hand"), masks)):
            if support.sum() < 24:
                continue
            known = comp & support
            rgba, completed = _complete_rgb(layer.rgba, known, support, use_lama)
            depth = layer.depth.copy()
            depth[completed] = round(float(np.median(depth[known])))
            piece = Layer(f"{prefix}_{name}", rgba, depth, layer.source,
                          pivot=joints_xy[min(i, 2)].tolist(),
                          parent=f"{prefix}_{('upper', 'forearm')[i-1]}" if i else None,
                          role=f"arm_{name}", completed=completed)
            piece.meta = {"joint_radius": radius, "pose_confidence": joints[:, 2].tolist(), "assignment_cost": score}
            pieces.append(piece)
        consumed.add(label)
        assigned.add((person, side))
        used |= comp
        events.append({"arm": prefix, "joints": joints_xy.tolist(), "confidence": joints[:, 2].tolist()})
    remainder = layer.mask & ~used
    if remainder.sum() >= 20:
        pieces.append(_cut(layer, remainder, "handwear_unsplit", role="unresolved_arm"))
    return pieces or [layer], events


def infer_limb_chains(layer: Layer, kind: str, poses: list[np.ndarray]) -> list[dict]:
    """Fit a smooth path to each completed limb, rather than its occluded source."""
    n, labels, stats, centers = cv2.connectedComponentsWithStats(layer.mask.astype(np.uint8), 8)
    chains: list[dict] = []
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] < 100:
            continue
        mask = labels == label
        sk = skeletonize(mask)
        sy, sx = np.where(sk)
        if len(sx) < 12:
            continue
        xy = np.column_stack([sx, sy]).astype(float)
        index = np.full(mask.shape, -1, np.int32)
        index[sy, sx] = np.arange(len(sx))
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = sx+dx, sy+dy
            ok = (nx >= 0) & (nx < mask.shape[1]) & (ny >= 0) & (ny < mask.shape[0])
            ids = np.flatnonzero(ok)
            dest = index[ny[ok], nx[ok]]
            keep = dest >= 0
            rows.append(ids[keep]); cols.append(dest[keep]); weights.append(np.full(keep.sum(), math.hypot(dx, dy)))
        graph = coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(len(sx), len(sx))).tocsr()
        proximal = xy[np.argmin(xy[:, 1])]
        hints: list[np.ndarray] = []
        for pose in poses:
            for idx in ([5, 6] if kind == "arm" else [11, 12]):
                if pose[idx, 2] > 0.3:
                    if np.linalg.norm(xy-pose[idx, :2], axis=1).min() < max(25, stats[label, 3]*0.3):
                        hints.append(pose[idx, :2])
        if hints:
            proximal = min(hints, key=lambda p: np.linalg.norm(xy-p, axis=1).min())
        start = int(np.linalg.norm(xy-proximal, axis=1).argmin())
        distance, predecessor = dijkstra(graph, directed=False, indices=start, return_predecessors=True)
        finite = np.isfinite(distance)
        end = int(np.argmax(np.where(finite, distance, -1)))
        path = [end]
        while path[-1] != start:
            nxt = int(predecessor[path[-1]])
            if nxt < 0:
                break
            path.append(nxt)
        path = path[::-1]
        curve = xy[path]
        progress = distance[path]/max(distance[end], 1)
        # Prefer an actual bend; otherwise use conservative anatomical proportions.
        valid = (progress > 0.25) & (progress < 0.72)
        scores = np.zeros(len(curve))
        for k in np.flatnonzero(valid):
            scores[k] = (np.linalg.norm(curve[k]-curve[0])+np.linalg.norm(curve[-1]-curve[k])-np.linalg.norm(curve[-1]-curve[0]))
        bend = int(scores.argmax())
        if scores[bend] < distance[end]*0.04:
            bend = int(np.abs(progress-0.50).argmin())
        distal_joint = int(np.abs(progress-0.86).argmin())
        points = [curve[0].tolist(), curve[bend].tolist(), curve[distal_joint].tolist(), curve[-1].tolist()]
        chains.append({"name": f"{kind}_component{len(chains)+1}", "points": points,
                       "provenance": "completed_mask_skeleton_estimate", "component": label,
                       "needs_joint_review": True})
    return chains


def _skeleton_partition(mask: np.ndarray, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Propagate joint ownership along branches (e.g. armour or elbow spikes)."""
    sk = skeletonize(mask)
    sy, sx = np.where(sk)
    if len(sx) < 12:
        return None
    xy = np.column_stack([sx, sy]).astype(float)
    index = np.full(mask.shape, -1, np.int32); index[sy, sx] = np.arange(len(sx))
    rows, cols, weights = [], [], []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        nx, ny = sx+dx, sy+dy
        valid = (nx >= 0) & (nx < mask.shape[1]) & (ny >= 0) & (ny < mask.shape[0])
        ids = np.flatnonzero(valid); dest = index[ny[valid], nx[valid]]; keep = dest >= 0
        rows.append(ids[keep]); cols.append(dest[keep]); weights.append(np.full(keep.sum(), math.hypot(dx, dy)))
    graph = coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(len(sx), len(sx))).tocsr()
    start, end = [int(np.linalg.norm(xy-p, axis=1).argmin()) for p in (joints[0], joints[-1])]
    distances, previous = dijkstra(graph, directed=False, indices=start, return_predecessors=True)
    if not np.isfinite(distances[end]) or distances[end] < 15:
        return None
    path = [end]
    while path[-1] != start:
        nxt = int(previous[path[-1]])
        if nxt < 0:
            return None
        path.append(nxt)
    path = np.asarray(path[::-1], dtype=int)
    curve = xy[path]
    cut1 = int(np.linalg.norm(curve-joints[1], axis=1).argmin())
    cut2 = int(np.linalg.norm(curve-joints[2], axis=1).argmin())
    if cut1 < 2 or cut2 <= cut1+2:
        return None
    label_sets = [path[:cut1+1], path[cut1+1:cut2+1], path[cut2+1:]]
    if not all(len(ids) for ids in label_sets):
        return None
    fields = np.stack([dijkstra(graph, directed=False, indices=ids, min_only=True) for ids in label_sets])
    owners = fields.argmin(axis=0).astype(np.uint8)
    nearest = ndi.distance_transform_edt(~sk, return_distances=False, return_indices=True)
    full = np.zeros(mask.shape, np.uint8); full[sy, sx] = owners
    full = full[tuple(nearest)]
    adjusted = joints.copy(); adjusted[1] = curve[cut1]; adjusted[2] = curve[cut2]
    return full, adjusted


def split_limb_chains(layer: Layer, chains: list[dict], kind: str, use_lama: bool) -> tuple[list[Layer], list[dict]]:
    if not chains:
        return [layer], []
    yy, xx = np.indices(layer.mask.shape)
    points = np.stack([xx, yy], axis=-1)
    ys, xs = np.where(layer.mask)
    pixels = np.column_stack([xs, ys])
    fields = []
    lengths = []
    for chain in chains:
        joints = np.asarray(chain["points"], dtype=float)
        if len(joints) == 3:
            joints = np.vstack([joints, joints[-1]+(joints[-1]-joints[-2])*0.25])
        chain["points"] = joints.tolist()
        fields.append(np.minimum.reduce([_segment_distance(pixels, a, b) for a, b in zip(joints[:-1], joints[1:])]))
        lengths.append(float(np.linalg.norm(np.diff(joints[:3], axis=0), axis=1).sum()))
    assignment = np.argmin(np.stack(fields), axis=0)
    pieces: list[Layer] = []
    events: list[dict] = []
    used = np.zeros(layer.mask.shape, bool)
    for ci, chain in enumerate(chains):
        joints = np.asarray(chain["points"], float)
        reviewed = chain.get("provenance", "reviewed_image_coordinates") == "reviewed_image_coordinates"
        selected = assignment == ci
        if not reviewed:
            selected &= fields[ci] < max(18, lengths[ci]*0.35)
        mask = np.zeros(layer.mask.shape, bool)
        mask[ys[selected], xs[selected]] = True
        if mask.sum() < 60:
            continue
        width = ndi.distance_transform_edt(mask)
        halfwidth = float(np.clip(np.percentile(width[mask], 92)*1.15, 3, lengths[ci]*0.15))
        ds = np.stack([_segment_distance(points, joints[i], joints[i+1]) for i in range(3)])
        owner = ds.argmin(axis=0)
        topology = _skeleton_partition(mask, joints)
        if topology is not None:
            owner, joints = topology
        axis = joints[2]-joints[1]
        beyond = ((points-joints[2])*axis).sum(axis=-1) >= 0
        masks = [mask & (owner == 0), mask & (owner == 1), mask & (owner == 2)] if topology is not None else [mask & (owner == 0), mask & (owner != 0) & ~beyond, mask & beyond]
        roles = ("upper", "forearm", "hand") if kind == "arm" else ("thigh", "shin", "foot")
        # Include a capsule only when the source already shows part of this limb.
        # Fully unseen limbs are not invented from an unrelated person's texture.
        for bi in range(2):
            if masks[bi].sum() < 24 and chain.get("complete_missing", False):
                masks[bi] |= _segment_distance(points, joints[bi], joints[bi+1]) <= halfwidth
        if min(masks[0].sum(), masks[1].sum()) < 20:
            continue
        for bi, (role, base) in enumerate(zip(roles, masks)):
            if base.sum() < 12:
                continue
            support = base.copy()
            for ji in ([1] if bi == 0 else [1, 2] if bi == 1 else [2]):
                r = halfwidth*(0.8 if ji == 2 else 1.05)
                joint_xy = np.clip(np.round(joints[ji]).astype(int), [0, 0], [mask.shape[1]-1, mask.shape[0]-1])
                r = max(r, float(width[joint_xy[1], joint_xy[0]])*1.3)
                support |= np.sum((points-joints[ji])**2, axis=-1) <= r*r
            known = mask & support
            # For a missing proximal section use the same limb as the colour context.
            context = known if known.sum() > 10 else mask
            rgba, completion = _complete_rgb(layer.rgba, context, support, use_lama)
            depth = layer.depth.copy()
            depth[completion] = round(float(np.median(depth[context])))
            name = f"{chain['name']}_{role}"
            piece = Layer(name, rgba, depth, layer.source, pivot=joints[bi].tolist(),
                          parent=f"{chain['name']}_{roles[bi-1]}" if bi else None,
                          role=f"{kind}_{role}", completed=completion)
            piece.meta = {"joint_radius": halfwidth, "joint_provenance": chain.get("provenance", "reviewed_image_coordinates"),
                          "needs_joint_review": chain.get("needs_joint_review", False)}
            pieces.append(piece)
        used |= mask
        events.append({"limb": chain["name"], "joints": joints.tolist(), "provenance": chain.get("provenance", "reviewed_image_coordinates")})
    remainder = layer.mask & ~used
    if remainder.sum() > 20:
        pieces.append(_cut(layer, remainder, f"{layer.source}_unsplit", role=f"unresolved_{kind}"))
    return pieces or [layer], events


def load_refined(output: Path) -> tuple[list[Layer], dict] | None:
    manifest = output / "关节与图层.json"
    if not manifest.is_file():
        return None
    info = json.loads(manifest.read_text(encoding="utf-8"))
    layers: list[Layer] = []
    for item in info["layers"]:
        name = item["name"]
        rgba = np.array(Image.open(output / item["file"]).convert("RGBA"))
        depth = np.array(Image.open(output / f"{name}_depth.png").convert("L"))
        cp = output / f"{name}_补全区域.png"
        completion = np.array(Image.open(cp)) > 0 if cp.is_file() else None
        layers.append(Layer(name, rgba, depth, item["source"], item.get("pivot_xy"), item.get("parent"), item["role"], completion,
                            {k: item[k] for k in ("joint_radius", "joint_provenance", "needs_joint_review", "label", "provenance") if k in item}))
    return layers, info


def _polygon(shape: tuple[int, int], xy: list[list[float]]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(xy).astype(np.int32)], 1)
    return mask > 0


def extract_scene_instances(src: np.ndarray, layers: list[Layer], instances: list[dict], use_lama: bool) -> tuple[list[Layer], list[dict]]:
    events: list[dict] = []
    shape = src.shape[:2]
    for instance in instances:
        name = instance["name"]
        initial = _polygon(shape, instance["polygon"])
        # Reviewed outlines seed an edge-aware refinement. It is not animal pose inference.
        x1, y1, x2, y2 = _bbox(initial, 12)
        local = initial[y1:y2, x1:x2].astype(np.uint8)
        gc = np.full(local.shape, cv2.GC_BGD, np.uint8)
        gc[cv2.dilate(local, np.ones((7, 7), np.uint8)) > 0] = cv2.GC_PR_BGD
        gc[local > 0] = cv2.GC_PR_FGD
        gc[cv2.erode(local, np.ones((3, 3), np.uint8)) > 0] = cv2.GC_FGD
        cv2.grabCut(src[y1:y2, x1:x2, :3].copy(), gc, None, np.zeros((1, 65), float), np.zeros((1, 65), float), 3, cv2.GC_INIT_WITH_MASK)
        mask = np.zeros(shape, bool)
        mask[y1:y2, x1:x2] = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
        mask = _components(mask, 12)
        rgba = src.copy(); rgba[~mask] = 0
        depth = np.full(shape, 145, np.uint8)
        original = Layer(name, rgba, depth, name)
        head = _polygon(shape, instance["head_polygon"]) & mask
        tail = _polygon(shape, instance["tail_polygon"]) & mask
        limb_mask = np.zeros(shape, bool)
        for chain in instance["legs"]:
            limb_mask |= _polygon(shape, chain["polygon"]) & mask
        body = mask & ~head & ~tail & ~limb_mask
        all_points = np.stack(np.indices(shape)[::-1], axis=-1)
        support = body.copy()
        for chain in instance["legs"]:
            support |= np.sum((all_points-np.array(chain["points"][0]))**2, axis=-1) < 8**2
        head_pivot = instance.get("head_pivot", np.mean(np.asarray(instance["head_polygon"], float), axis=0).tolist())
        support |= np.sum((all_points-np.asarray(head_pivot))**2, axis=-1) < 8**2
        body_rgba, completed = _complete_rgb(rgba, body, support, use_lama)
        pieces = [Layer(f"{name}_body", body_rgba, depth.copy(), name, role="animal_body", completed=completed)]
        pieces.extend([_cut(original, head, f"{name}_head", role="animal_head", pivot=head_pivot, parent=f"{name}_body"),
                       _cut(original, tail, f"{name}_tail", role="animal_tail", parent=f"{name}_body")])
        limb = _cut(original, limb_mask, f"{name}_limbs")
        limb.source = name
        split, event = split_limb_chains(limb, instance["legs"], "leg", use_lama)
        pieces.extend(split)
        for piece in pieces:
            if piece.role == "leg_thigh":
                piece.parent = f"{name}_body"
        erase = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        for layer in layers:
            layer.rgba[erase] = 0
        if use_lama:
            from annotators.lama_inpainter import apply_inpaint
            local_rgb = apply_inpaint(src[y1:y2, x1:x2, :3], erase[y1:y2, x1:x2].astype(np.uint8)*255)
            patch = np.zeros_like(src)
            patch[y1:y2, x1:x2, :3] = local_rgb
            patch[erase, 3] = 255
            pieces.insert(0, Layer(f"{name}_background_patch", patch, np.full(shape, 245, np.uint8), "background", role="background_completion", completed=erase))
        layers.extend(pieces)
        events.append({"name": name, "method": "reviewed_outline_grabcut_author_lama", "visible_limb_chains": event,
                       "warning": "Only the three visible cow legs are split; the fully hidden fourth leg is not verified or fabricated."})
    return layers, events


def _remove_duplicate_cloth(layers: list[Layer], garment: Layer) -> dict | None:
    top = next((p for p in layers if p.name == "topwear"), None)
    if top is None:
        return None
    overlap = top.mask & garment.mask & (top.rgba[..., 3] > 100) & (garment.rgba[..., 3] > 100)
    if overlap.sum() < garment.mask.sum()*0.55:
        return None
    difference = np.linalg.norm(top.rgba[..., :3].astype(float)-garment.rgba[..., :3].astype(float), axis=-1)
    if np.median(difference[overlap]) > 60:
        return None
    # Restrict ownership cleanup to the garment silhouette and its immediate boundary.
    # Transfer the better continuous-dress texture before removing its duplicate.
    # This avoids exposing bottomwear's generated horizontal cut-line at the waist.
    near = cv2.dilate(garment.mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ymin = _bbox(garment.mask)[1]
    near[:ymin] = False
    transfer = near & top.mask
    garment.rgba[transfer] = top.rgba[transfer]
    garment.depth[transfer] = top.depth[transfer]
    remove = near
    count = int((top.mask & remove).sum())
    top.rgba[remove] = 0
    return {"source": "topwear", "owner": garment.name, "removed_pixels": count}


def composite(layers: list[Layer]) -> np.ndarray:
    """Per-pixel depth composition; median depth alone loses crossing hair/limbs."""
    layers = [p for p in layers if p.meta.get('default_visible', True)]
    if not layers:
        raise ValueError("No non-empty layers")
    if len(layers) <= 40:
        return _composite_dense(layers)
    # Character sheets have many small disjoint parts. Sort only layers touching
    # each tile instead of allocating/sorting N full-page copies of empty pixels.
    h, w = layers[0].rgba.shape[:2]
    result = np.zeros((h, w, 4), np.uint8)
    boxes = [cv2.boundingRect(p.rgba[..., 3]) for p in layers]
    for y in range(0, h, 128):
        for x in range(0, w, 128):
            y2, x2 = min(h, y+128), min(w, x+128)
            active = [Layer(p.name, p.rgba[y:y2, x:x2], p.depth[y:y2, x:x2], p.source,
                            visibility_mask=None if p.visibility_mask is None else p.visibility_mask[y:y2, x:x2])
                      for p, (bx, by, bw, bh) in zip(layers, boxes)
                      if bw and bh and bx < x2 and bx+bw > x and by < y2 and by+bh > y]
            if active:
                result[y:y2, x:x2] = _composite_dense(active)
    return result


def _composite_dense(layers: list[Layer]) -> np.ndarray:
    shape = layers[0].rgba.shape[:2]
    rgba = np.stack([p.rgba for p in layers])
    for i, p in enumerate(layers):
        if p.visibility_mask is not None:
            rgba[i, ..., 3] = np.round(rgba[i, ..., 3].astype(float)*p.visibility_mask/255).astype(np.uint8)
    depths = np.stack([np.where(p.rgba[..., 3] > 0, p.depth.astype(float), 256) for p in layers])
    # Stable ordering ensures identical overlap pixels remain deterministic.
    order = np.argsort(-depths, axis=0, kind="stable")
    rgb_premult = np.zeros((*shape, 3), np.float32)
    alpha = np.zeros(shape, np.float32)
    for i in range(len(layers)):
        item = np.take_along_axis(rgba, order[i][None, ..., None], axis=0)[0].astype(np.float32)/255
        a = item[..., 3]
        rgb_premult = item[..., :3]*a[..., None]+rgb_premult*(1-a[..., None])
        alpha = a+alpha*(1-a)
    rgb = rgb_premult/np.maximum(alpha[..., None], 1e-8)
    return np.round(np.dstack([rgb, alpha])*255).clip(0, 255).astype(np.uint8)


def _warp(layer: Layer, matrix: np.ndarray) -> Layer:
    h, w = layer.mask.shape
    # Premultiply before interpolation to avoid dark/grey outlines on transparent edges.
    premult = layer.rgba.astype(np.float32)/255
    premult[..., :3] *= premult[..., 3:4]
    out = cv2.warpAffine(premult, matrix, (w, h), flags=cv2.INTER_LINEAR)
    out[..., :3] /= np.maximum(out[..., 3:4], 1e-8)
    rgba = np.round(out*255).clip(0, 255).astype(np.uint8)
    depth = cv2.warpAffine(layer.depth, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
    visibility = None if layer.visibility_mask is None else cv2.warpAffine(layer.visibility_mask, matrix, (w, h))
    return Layer(layer.name, rgba, depth, layer.source, layer.pivot, layer.parent, layer.role,
                 meta=layer.meta.copy(), visibility_mask=visibility)


def _on_checker(rgba: np.ndarray, size: int = 640) -> Image.Image:
    h, w = rgba.shape[:2]
    yy, xx = np.indices((h, w))
    bg = np.where(((xx//24+yy//24) % 2)[..., None], 232, 210).astype(np.uint8)
    bg = np.repeat(bg, 3, axis=2)
    a = rgba[..., 3:4].astype(float)/255
    image = Image.fromarray(np.round(rgba[..., :3]*a+bg*(1-a)).astype(np.uint8))
    image.thumbnail((size, size))
    return image


def save_previews(layers: list[Layer], output: Path, export_motion: bool = False) -> None:
    """Export static overviews; motion files are opt-in for internal inspection."""
    _on_checker(composite(layers), 1024).save(output / "静态合成.png")
    # Show the complete deliverable, including unchanged head/body/props. An
    # overview containing only newly split limbs misrepresents what is exported.
    featured = [p for p in layers if p.role != "background_completion" and p.mask.any()]
    if featured:
        contact = Image.new("RGB", (4*210, math.ceil(len(featured)/4)*260), "#eeeeee")
        font_path = Path("C:/Windows/Fonts/msyh.ttc")
        font = ImageFont.truetype(str(font_path), 13) if font_path.is_file() else ImageFont.load_default()
        labels_cn = {"arm_upper":"大臂", "arm_forearm":"小臂", "arm_hand":"手", "leg_thigh":"大腿", "leg_shin":"小腿", "leg_foot":"脚", "cloth_inner":"内层布料", "cloth_outer":"外层布料", "animal_body":"躯干", "animal_head":"头", "animal_tail":"尾巴"}
        for i, p in enumerate(featured):
            x1, y1, x2, y2 = _bbox(p.mask, 4)
            im = _on_checker(p.rgba[y1:y2, x1:x2], 200)
            x, y = (i%4)*210, (i//4)*260
            contact.paste(im, (x+(210-im.width)//2, y+8))
            draw = ImageDraw.Draw(contact)
            original_labels = {"hair":"头发","headwear":"头饰","face":"脸部","eyes":"眼睛","ears":"耳朵","nose":"鼻子","mouth":"嘴部","neck":"颈部","neckwear":"颈饰","topwear":"上装","bottomwear":"下装","handwear":"手臂待细分","legwear":"腿部待细分","footwear":"鞋","tail":"尾巴","wings":"翅膀","objects":"物品"}
            label = p.meta.get("label", labels_cn.get(p.role, original_labels.get(p.name, p.role)))
            draw.text((x+5, y+214), label[:15], font=font, fill="black")
            draw.text((x+5, y+237), p.name[:27], font=font, fill="black")
        contact.save(output / "部件总览.png")
    if not export_motion:
        return
    union = np.logical_or.reduce([p.mask for p in layers if p.source != "objects"] or [p.mask for p in layers])
    x1, y1, x2, y2 = _bbox(union, 28)
    scale = min(1.0, 640/max(y2-y1, x2-x1))
    size = (max(1, round((x2-x1)*scale)), max(1, round((y2-y1)*scale)))
    small: list[Layer] = []
    for layer in layers:
        premult = layer.rgba[y1:y2, x1:x2].astype(np.float32)/255
        premult[..., :3] *= premult[..., 3:4]
        premult = cv2.resize(premult, size, interpolation=cv2.INTER_AREA)
        premult[..., :3] /= np.maximum(premult[..., 3:4], 1e-8)
        rgba = np.round(premult*255).clip(0, 255).astype(np.uint8)
        depth = cv2.resize(layer.depth[y1:y2, x1:x2], size, interpolation=cv2.INTER_NEAREST)
        pivot = ((np.array(layer.pivot)-[x1, y1])*scale).tolist() if layer.pivot else None
        small.append(Layer(layer.name, rgba, depth, layer.source, pivot, layer.parent, layer.role))
    layers = small
    cloth = next((p for p in layers if p.role == "cloth_outer" and p.source == "bottomwear"), None)
    moving = [p for p in layers if p.role in {"arm_forearm", "arm_hand"}]
    moving_legs = [p for p in layers if p.role in {"leg_shin", "leg_foot"}]
    kinds = []
    if cloth is not None:
        kinds.append("裙子抬起")
    if moving:
        kinds.append("手臂屈伸")
    if moving_legs:
        kinds.append("腿部屈伸")
    for kind in kinds:
        frames: list[Image.Image] = []
        for t in range(13):
            amount = (1-math.cos(t/12*2*math.pi))/2
            transformed: list[Layer] = []
            for p in layers:
                matrix = None
                if kind == "裙子抬起" and p is cloth:
                    pivot = p.pivot
                    matrix = np.array([[1, 0, 0], [0, 1-amount*0.58, pivot[1]*amount*0.58]], dtype=float)
                if kind == "手臂屈伸" and p in moving:
                    base = p if p.role == "arm_forearm" else next((v for v in moving if v.name == p.parent), None)
                    if base is not None:
                        direction = 1 if "left" in base.name else -1
                        matrix = cv2.getRotationMatrix2D(tuple(base.pivot), direction*22*amount, 1)
                if kind == "腿部屈伸" and p in moving_legs:
                    base = p if p.role == "leg_shin" else next((v for v in moving_legs if v.name == p.parent), None)
                    if base is not None:
                        direction = 1 if "left" in base.name else -1
                        matrix = cv2.getRotationMatrix2D(tuple(base.pivot), direction*20*amount, 1)
                transformed.append(_warp(p, matrix) if matrix is not None else p)
            rgba = composite(transformed)
            image = _on_checker(rgba, 640)
            panel = Image.new("RGB", (image.width, image.height+28), "white")
            panel.paste(image)
            draw = ImageDraw.Draw(panel)
            labels = {"裙子抬起": "Outer cloth lift 0-58%", "手臂屈伸": "Elbow rotation 0-22 deg", "腿部屈伸": "Knee rotation 0-20 deg"}
            draw.text((8, image.height+5), labels[kind]+" | motion check", fill="black")
            frames.append(panel)
            if t == 6:
                panel.save(output / f"{kind}_最大幅度.png")
        frames[0].save(output / f"{kind}.gif", save_all=True, append_images=frames[1:], duration=90, loop=0, disposal=2)


def export_layers(layers: list[Layer], output: Path, metadata: dict) -> None:
    from psd_tools import PSDImage
    output.mkdir(parents=True, exist_ok=True)
    previous_names = {p["name"] for p in metadata.get("layers", [])}
    h, w = layers[0].mask.shape
    psd = PSDImage.new(mode="RGBA", size=(w, h))
    entries: list[dict] = []
    # The manifest preserves compositing order: equal local depths are resolved
    # by the later entry. Median sorting here changes those ties after reload.
    for layer in layers:
        if not layer.mask.any():
            continue
        Image.fromarray(layer.rgba).save(output / f"{layer.name}.png")
        Image.fromarray(layer.depth).save(output / f"{layer.name}_depth.png")
        if layer.completed is not None and layer.completed.any():
            Image.fromarray(layer.completed.astype(np.uint8)*255).save(output / f"{layer.name}_补全区域.png")
        x1, y1, x2, y2 = _bbox(layer.mask)
        entries.append({"name": layer.name, "file": f"{layer.name}.png", "source": layer.source,
                        "role": layer.role, "pivot_xy": layer.pivot, "parent": layer.parent,
                        "bbox_xyxy": [x1, y1, x2, y2], "median_depth": layer.z,
                        "inferred_pixels": int(layer.completed.sum()) if layer.completed is not None else 0,
                        **layer.meta})
    for layer in sorted(layers, key=lambda p: p.z, reverse=True):
        if not layer.mask.any():
            continue
        x1, y1, x2, y2 = _bbox(layer.mask)
        psd.create_pixel_layer(Image.fromarray(layer.rgba[y1:y2, x1:x2]), name=layer.name, top=y1, left=x1)
    psd.save(output / "动画拆层.psd")
    _cleanup_stale(output, {e["name"] for e in entries}, previous_names)
    metadata.update({"canvas_wh": [w, h], "layers": entries,
                     "coordinates": "top_left_xy_pixels_full_canvas",
                     "composition": "smaller_depth_is_closer_per_pixel; equal_depth_uses_later_manifest_entry; PSD uses median ordering",
                     "limitations": ["Estimated 2D pose and depth; inferred cloth and joint pixels require visual review.",
                                     "Previews test only the displayed motion range; no arbitrary-motion guarantee.",
                                     "PNG/PSD and pivot hierarchy are provided; no Spine mesh weights or project."]})
    (output / "关节与图层.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _cleanup_stale(output: Path, names: set[str], previous: set[str] | None = None) -> None:
    canonical = {"handwear", "legwear", "footwear", "bottomwear", "topwear"}
    candidates = set(previous or set()) | canonical
    for path in output.glob("*_unsplit.png"):
        candidates.add(path.stem)
    for stale in candidates-names:
        for suffix in (".png", "_depth.png", "_补全区域.png"):
            path = output / f"{stale}{suffix}"
            if path.is_file():
                path.unlink()


def refine_for_animation(source: str | Path, output: str | Path, pose_model: str | Path | None = None,
                         use_lama: bool = True, previews: bool = True) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source:
        raise ValueError("Use the animation subfolder; do not overwrite original semantic PNGs")
    src, originals = load_layers(source)
    cached = load_refined(output)
    if cached and cached[1].get("refinement_version", 0) >= 5:
        from utils.animation_integrity import REVISION, ensure_complete, foreground_mask, measure_completeness, read_hints
        import hashlib
        integrity_hints, hints_digest = read_hints(source.name)
        integrity = cached[1].get("integrity", {})
        target, _ = foreground_mask(src, originals, integrity_hints)
        current_coverage = measure_completeness(target, cached[0])
        source_digest = hashlib.sha256(src.tobytes()).hexdigest()
        if (integrity.get("revision", 0) < REVISION or integrity.get("hints_digest") != hints_digest
                or integrity.get("source_sha256") != source_digest or current_coverage["missing_pixels"]):
            repaired, info = ensure_complete(src, originals, cached[0], cached[1], source, use_lama)
            export_layers(repaired, output, info)
            if previews:
                save_previews(repaired, output)
            return {**info, "reused": True}
        if cached[1].get("completion_revision", 0) < 2:
            for layer in cached[0]:
                if layer.completed is not None and layer.completed.any() and layer.role != "background_completion":
                    known = layer.mask & ~layer.completed
                    if known.any():
                        layer.rgba, _ = _complete_rgb(layer.rgba, known, layer.mask, use_lama)
            cached[1]["completion_revision"] = 2
            export_layers(cached[0], output, cached[1])
            if previews:
                save_previews(cached[0], output)
        _cleanup_stale(output, {p["name"] for p in cached[1]["layers"]})
        return {**cached[1], "reused": True}
    layers, metadata = cached if cached else (originals, {"input": str(source), "method": "Ours Full + optional motion refinement", "cloth_splits": [], "arms": [], "ownership_cleanup": [], "warnings": []})
    metadata.update({"completion_method": "author_LaMa" if use_lama else "nearest_material_debug", "legs": [], "instances": [], "refinement_version": 5, "completion_revision": 2})
    metadata["warnings"] = []
    if not layers:
        raise ValueError(f"No nonempty layers: {source}")
    pose_path = Path(pose_model) if pose_model is not None else PROJECT / "workspace/models/pose/yolo11s-pose.pt"
    if not pose_path.is_file():
        raise FileNotFoundError(f"Pose weights must be downloaded into this project: {pose_path}")
    poses = detect_poses(src, pose_path)
    metadata["detected_people"] = len(poses)
    hint_path = PROJECT / "workspace/animation_joint_hints.json"
    hints = json.loads(hint_path.read_text(encoding="utf-8")).get(source.name, {}) if hint_path.is_file() else {}
    metadata["reviewed_joint_hints"] = bool(hints)
    is_character = bool(poses) or hints.get("kind") == "biped"
    garment = next((p for p in layers if p.name == "bottomwear"), None)
    if garment is not None and is_character and not cached:
        # Inspect ownership on copies; a rejected cloth split leaves inputs intact.
        original_top = next((p for p in layers if p.name == "topwear"), None)
        trial_top = _cut(original_top, original_top.mask, original_top.name) if original_top is not None else None
        trial_garment = _cut(garment, garment.mask, garment.name)
        cleanup = _remove_duplicate_cloth([trial_top] if trial_top is not None else [], trial_garment)
        result = split_cloth(trial_garment, use_lama)
        if result is not None:
            split, evidence = result
            if cleanup and original_top is not None:
                original_top.rgba = trial_top.rgba
                metadata["ownership_cleanup"].append(cleanup)
            layers = [p for p in layers if p is not garment]+split
            metadata["cloth_splits"].append(evidence)
        else:
            metadata["warnings"].append("No reliable inner/outer cloth separation; bottomwear retained.")
    arm = next((p for p in originals if p.name == "handwear"), None)
    if arm is not None and hints.get("arms"):
        split, events = split_limb_chains(arm, hints["arms"], "arm", use_lama)
        layers = [p for p in layers if p.source != "handwear"]+split
        metadata["arms"] = events
    elif arm is not None and is_character:
        if not cached:
            split, events = split_arms(arm, poses, use_lama)
            layers = [p for p in layers if p is not arm]+split
            metadata["arms"] = events
        # Retry each unresolved completed limb using its own shape.
        for pending in [p for p in layers if p.source == "handwear" and (p.role.startswith("unresolved") or p.role == "original")]:
            chains = infer_limb_chains(pending, "arm", poses)
            existing = {p.name for p in layers if p is not pending}
            for chain in chains:
                if any(name.startswith(chain["name"]+"_") for name in existing):
                    chain["name"] = pending.name+"_"+chain["name"]
            split, events = split_limb_chains(pending, chains, "arm", use_lama)
            if events:
                layers = [p for p in layers if p is not pending]+split
                metadata["arms"].extend(events)
    leg = next((p for p in originals if p.name == "legwear"), None)
    if leg is not None and is_character:
        chains = hints.get("legs", [])
        if not chains:
            for pi, pose in enumerate(poses):
                for side, ids in (("left", [11, 13, 15]), ("right", [12, 14, 16])):
                    joints = pose[ids]
                    if joints[:, 2].min() < 0.35:
                        continue
                    distance = ndi.distance_transform_edt(~leg.mask)
                    xy = np.clip(np.round(joints[:, :2]).astype(int), [0, 0], [leg.mask.shape[1]-1, leg.mask.shape[0]-1])
                    limb_length = np.linalg.norm(np.diff(joints[:, :2], axis=0), axis=1).sum()
                    if distance[xy[1:, 1], xy[1:, 0]].min() > max(24, limb_length*0.2):
                        continue
                    points_xy = joints[:, :2].copy()
                    shoe = next((p for p in originals if p.name == "footwear"), None)
                    if shoe is not None:
                        close = shoe.mask & (np.linalg.norm(np.stack(np.indices(shoe.mask.shape)[::-1], axis=-1)-points_xy[-1], axis=-1) < limb_length*0.25)
                        if close.sum() >= 20:
                            _, sy, _, _ = _bbox(close)
                            ax = np.median(np.where(close & (np.indices(close.shape)[0] < sy+8))[1])
                            points_xy[-1] = [ax, sy+3]
                    chains.append({"name":f"person{pi+1}_leg_{side}","points":points_xy.tolist(),"provenance":"human_pose_with_mask_alignment","complete_missing":True})
            if not chains:
                chains = infer_limb_chains(leg, "leg", poses)
        split, events = split_limb_chains(leg, chains, "leg", use_lama)
        if events:
            layers = [p for p in layers if p.source != "legwear"]+split
            metadata["legs"] = events
            for shoe in [p for p in layers if p.source == "footwear"]:
                # The author's legwear often repeats the shoes; moving feet need one owner.
                shoe.rgba[leg.mask] = 0
    if hints.get("instances"):
        names = [p["name"] for p in hints["instances"]]
        layers = [p for p in layers if not any(p.name.startswith(name+"_") for name in names)]
        layers, metadata["instances"] = extract_scene_instances(src, layers, hints["instances"], use_lama)
    unresolved = [p.name for p in layers if p.role.startswith("unresolved")]
    if unresolved:
        metadata["warnings"].append("Unresolved remnants retained: "+", ".join(unresolved))
    estimated = [p.name for p in layers if p.meta.get("needs_joint_review")]
    if estimated:
        metadata["warnings"].append("Skeleton-estimated joints need review: "+", ".join(estimated))
    if not metadata["cloth_splits"] and not metadata["arms"] and not metadata["legs"] and not metadata["instances"]:
        metadata["unchanged"] = True
    else:
        metadata.pop("unchanged", None)
    layers = [p for p in layers if p.mask.any()]
    from utils.animation_integrity import ensure_complete
    layers, metadata = ensure_complete(src, originals, layers, metadata, source, use_lama)
    export_layers(layers, output, metadata)
    if previews:
        save_previews(layers, output)
    return metadata

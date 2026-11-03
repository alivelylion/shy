"""Regression checks for source bodies omitted by the author/cached export."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "common")]

from utils.animation_parts import Layer, composite, export_layers, load_refined, refine_for_animation
from utils.animation_integrity import ensure_complete, foreground_mask


def sprite_fixture() -> tuple[np.ndarray, Layer]:
    rgba = np.zeros((64, 64, 4), np.uint8)
    yy, xx = np.indices((64, 64))
    head = (xx-32)**2+(yy-17)**2 <= 12**2
    body = (xx >= 22) & (xx < 43) & (yy >= 25) & (yy < 51)
    feet = (yy >= 48) & (yy < 61) & (((xx >= 18) & (xx < 29)) | ((xx >= 36) & (xx < 47)))
    mask = head | body | feet
    rgba[mask, 0] = xx[mask]*3
    rgba[mask, 1] = yy[mask]*3
    rgba[mask, 2] = 140
    rgba[mask, 3] = 255
    face = rgba.copy(); face[~head] = 0
    return rgba, Layer("face", face, np.full((64, 64), 90, np.uint8), "face")


class CompletenessRegressionTests(unittest.TestCase):
    def test_face_only_prediction_preserves_missing_source_body(self) -> None:
        src, face = sprite_fixture()
        layers, info = ensure_complete(src, [face], [face], {"warnings":[]}, Path("unlisted_sprite_fixture"), use_lama=False)
        self.assertLess(info["integrity"]["before"]["coverage"], 0.6)
        self.assertEqual(info["integrity"]["after"]["missing_pixels"], 0)
        np.testing.assert_array_equal(composite(layers), src)
        self.assertFalse(info["integrity"]["joint_segmentation_verified"])
        self.assertTrue(any(p.role == "source_unclassified" for p in layers))

    def test_existing_version5_cache_is_checked_before_reuse(self) -> None:
        cache = ROOT / "workspace/.cache/tmp"
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache, prefix="integrity_test_") as temp:
            directory = Path(temp).resolve()
            self.assertTrue(directory.is_relative_to(cache.resolve()))
            source, output = directory / "sprite", directory / "result"
            source.mkdir()
            src, face = sprite_fixture()
            Image.fromarray(src).save(source / "src_img.png")
            Image.fromarray(face.rgba).save(source / "face.png")
            Image.fromarray(face.depth).save(source / "face_depth.png")
            (source / "info.json").write_text(json.dumps({"parts":{"face":{}}}), encoding="utf-8")
            info = {"refinement_version":5,"completion_revision":2,"warnings":[],"arms":[],"legs":[],"instances":[],"cloth_splits":[]}
            export_layers([face], output, info)
            repaired = refine_for_animation(source, output, use_lama=False, previews=False)
            self.assertEqual(repaired["integrity"]["after"]["coverage"], 1.0)
            stored = json.loads((output / "关节与图层.json").read_text(encoding="utf-8"))
            self.assertGreater(len(stored["layers"]), 1)
            self.assertEqual(stored["integrity"]["after"]["missing_pixels"], 0)
            from psd_tools import PSDImage
            psd = PSDImage.open(output / "动画拆层.psd")
            rendered = np.asarray(psd.composite(force=True).convert("RGBA"))
            np.testing.assert_array_equal(rendered[..., 3] > 0, src[..., 3] > 0)
            # A valid manifest must not hide a later incomplete PNG edit.
            remainder = output / "source_visible_remainder.png"
            damaged = np.array(Image.open(remainder))
            damaged[40:] = 0
            Image.fromarray(damaged).save(remainder)
            recovered = refine_for_animation(source, output, use_lama=False, previews=False)
            self.assertEqual(recovered["integrity"]["after"]["missing_pixels"], 0)
            names = [part["name"] for part in recovered["layers"]]
            self.assertEqual(len(names), len(set(names)))

    def test_repeated_foreground_audits_use_the_same_boundary(self) -> None:
        src = np.full((64,64,4), 255, np.uint8)
        src[..., :3] = [20,145,210]
        cv2.circle(src, (32,31), 21, (245,245,245,255), -1)
        hints = {"foreground_polygon":[[11,11],[52,11],[52,53],[11,53]],"refine_boundary":True}
        a, _ = foreground_mask(src, [], hints)
        cv2.setRNGSeed(12345)
        b, _ = foreground_mask(src, [], hints)
        np.testing.assert_array_equal(a, b)

    def test_export_reload_preserves_local_depth_ties(self) -> None:
        src, _ = sprite_fixture()
        first = src.copy()
        first[..., :3] = [230, 20, 20]
        second = src.copy()
        second[..., :3] = [20, 20, 230]
        front_depth = np.full(src.shape[:2], 30, np.uint8)
        back_depth = np.full(src.shape[:2], 150, np.uint8)
        back_depth[:23] = 30
        layers = [Layer("front", first, front_depth, "fixture"),
                  Layer("back", second, back_depth, "fixture")]
        layers, info = ensure_complete(src, layers, layers, {"warnings":[]},
                                       Path("depth_tie_fixture"), use_lama=False)
        cache = ROOT / "workspace/.cache/tmp"
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache, prefix="integrity_test_") as temp:
            output = Path(temp)
            export_layers(layers, output, info)
            reloaded, _ = load_refined(output)
            self.assertEqual([p.name for p in reloaded], [p.name for p in layers])
            np.testing.assert_array_equal(composite(reloaded), composite(layers))
            np.testing.assert_array_equal(composite(reloaded), src)


if __name__ == "__main__":
    unittest.main()

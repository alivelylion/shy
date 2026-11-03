"""Download only the four models used by the Windows animation entry point."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time
from urllib.request import Request, urlopen

import download_paper_assets as assets


POSE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt"
POSE_SHA256 = "1060bda4a27012060eca246f9b2adeea22eabb045a1e58f8d229be29b7ebc2ba"
POSE_SIZE = 20373161


def pose_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_pose() -> None:
    target = assets.WORKSPACE / "models/pose/yolo11s-pose.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and pose_hash(target) == POSE_SHA256:
        print("VERIFIED pose/yolo11s-pose.pt", flush=True)
        return
    partial = target.with_suffix(".pt.part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > POSE_SIZE:
        raise ValueError(f"Oversized partial file; inspect before retrying: {partial}")
    if offset < POSE_SIZE:
        headers = {"User-Agent": "SeeThrough-animation", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with urlopen(Request(POSE_URL, headers=headers), timeout=60) as response:
            if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                raise ValueError("Pose model server did not honor the resume offset")
            with partial.open("ab" if offset else "wb") as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
    if partial.stat().st_size != POSE_SIZE or pose_hash(partial) != POSE_SHA256:
        raise ValueError(f"Pose model checksum mismatch: {partial}")
    partial.replace(target)
    print("VERIFIED pose/yolo11s-pose.pt", flush=True)


def download_record(record: dict) -> None:
    for attempt in range(3):
        try:
            path = assets.transfer_file(record)
            assets.verify_file(path, record)
            print(f"VERIFIED {record['repo_id']}/{record['filename']}", flush=True)
            return
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Fetch metadata and show size without downloading model files")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    assets.ASSETS[:] = [asset for asset in assets.ASSETS if asset["repo_type"] == "model"]
    manifest = assets.make_manifest()
    print(f"MODEL_FILES {manifest['file_count'] + 1}; TOTAL_GB {(manifest['total_bytes'] + POSE_SIZE) / 1e9:.3f}", flush=True)
    for asset in assets.ASSETS:
        print(f"SOURCE https://huggingface.co/{asset['repo_id']}/tree/{asset['revision']}", flush=True)
    print(f"SOURCE {POSE_URL}", flush=True)
    if args.list:
        return
    assets.write_json(assets.DOWNLOADS / "inference_models_manifest.json", manifest)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for _ in executor.map(download_record, manifest["files"]):
            pass
    download_pose()
    print("MODELS_OK: weights downloaded and verified; no test dataset was downloaded.", flush=True)


if __name__ == "__main__":
    main()

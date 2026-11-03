"""Download pinned public reproduction assets using an existing Python environment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "workspace"
DOWNLOADS = WORKSPACE / "downloads"
CACHE = WORKSPACE / ".cache"
for key, value in {
    "HF_HOME": CACHE / "huggingface",
    "HF_HUB_CACHE": CACHE / "huggingface" / "hub",
    "HF_XET_CACHE": CACHE / "huggingface" / "xet",
    "TORCH_HOME": CACHE / "torch",
}.items():
    os.environ[key] = str(value)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

ASSETS = [
    {
        "repo_id": "layerdifforg/seethroughv0.0.1_layerdiff3d",
        "repo_type": "model",
        "revision": "6e0aa05f6160a40a53137ed3f1a318280367c8ed",
        "local_dir": "models/layerdiff_v0.0.1",
        "role": "paper_19_class_pretrained_model",
    },
    {
        "repo_id": "layerdifforg/seethroughv0.0.1_marigold",
        "repo_type": "model",
        "revision": "4f4ffc46050b6feb764b628859968a26e76e6b6a",
        "local_dir": "models/marigold_v0.0.1",
        "role": "paper_pseudo_depth_model",
    },
    {
        "repo_id": "dreMaz/AnimeMangaInpainting",
        "repo_type": "model",
        "revision": "2953a4e935bf01ad1471f6cbfd26ab81abeeb92d",
        "local_dir": "models/lama",
        "include": ["lama_large_512px.ckpt"],
        "role": "lama_inpainting_baseline_weight",
    },
    {
        "repo_id": "layerdifforg/layerdiff_eval",
        "repo_type": "dataset",
        "revision": "b8ec75e84674fa49d764704ced5e7bd4e0adc469",
        "local_dir": "datasets/layerdiff_eval",
        "role": "public_inputs_and_reference_predictions_NOT_confirmed_test_ground_truth",
    },
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def make_manifest() -> dict:
    records = []
    for asset in ASSETS:
        kind = "datasets" if asset["repo_type"] == "dataset" else "models"
        url = f"https://huggingface.co/api/{kind}/{asset['repo_id']}/revision/{asset['revision']}?blobs=true"
        with urlopen(Request(url, headers={"User-Agent": "See-through-reproduction"}), timeout=60) as response:
            info = json.load(response)
        if info["sha"] != asset["revision"]:
            raise ValueError("Pinned repository revision did not match")
        for sibling in info["siblings"]:
            if asset.get("include") and sibling["rfilename"] not in asset["include"]:
                continue
            records.append({
                "repo_id": asset["repo_id"],
                "repo_type": asset["repo_type"],
                "revision": asset["revision"],
                "filename": sibling["rfilename"],
                "size": sibling["size"],
                "sha256": sibling.get("lfs", {}).get("sha256"),
                "git_blob_sha1": sibling["blobId"],
                "path": str(WORKSPACE / asset["local_dir"] / sibling["rfilename"]),
                "local_dir": str(WORKSPACE / asset["local_dir"]),
                "role": asset["role"],
            })
    return {
        "created_at": now(),
        "files": records,
        "total_bytes": sum(record["size"] for record in records),
        "file_count": len(records),
        "official_847_test_ground_truth_available": False,
        "ground_truth_status": "not_located_or_verified_in_public_sources_checked",
        "without_consistency_checkpoint_available": False,
        "without_consistency_status": "not_located_in_public_sources_checked",
    }


def verify_file(path: Path, record: dict) -> str:
    if path.stat().st_size != record["size"]:
        raise ValueError("File size does not match pinned manifest")
    if record["sha256"]:
        digest = hashlib.sha256()
    else:
        digest = hashlib.sha1()
        digest.update(f"blob {record['size']}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    expected = record["sha256"] or record["git_blob_sha1"]
    if digest.hexdigest() != expected:
        raise ValueError("File hash does not match pinned manifest")
    return digest.hexdigest()


def legacy_partials(record: dict) -> list[Path]:
    directory = Path(record["local_dir"]) / ".cache" / "huggingface" / "download" / Path(record["filename"]).parent
    if not directory.exists():
        return []
    digest = record["sha256"] or record["git_blob_sha1"]
    return list(directory.glob(f"*.{digest}.incomplete"))


def transfer_file(record: dict) -> Path:
    path = Path(record["path"])
    if not path.resolve().is_relative_to(WORKSPACE.resolve()):
        raise ValueError("Download destination is outside the project workspace")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == record["size"]:
        return path
    partial = path.with_name(path.name + ".part")
    # Preserve work from the earlier Hugging Face downloader.
    if not partial.exists():
        candidates = legacy_partials(record)
        if candidates:
            source = max(candidates, key=lambda item: item.stat().st_size)
            if not source.resolve().is_relative_to(WORKSPACE.resolve()):
                raise ValueError("Partial download is outside the project workspace")
            source.replace(partial)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > record["size"]:
        raise ValueError("Partial download is larger than its pinned file size")
    if offset < record["size"]:
        prefix = "datasets/" if record["repo_type"] == "dataset" else ""
        url = f"https://huggingface.co/{prefix}{record['repo_id']}/resolve/{record['revision']}/{quote(record['filename'], safe='/')}"
        headers = {"User-Agent": "See-through-reproduction", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with urlopen(Request(url, headers=headers), timeout=60) as response:
            if offset:
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or not content_range.startswith(f"bytes {offset}-"):
                    raise ValueError("Server did not honor the resume offset")
            with partial.open("ab" if offset else "wb") as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
    verify_file(partial, record)
    partial.replace(path)
    return path


def main() -> None:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    manifest_path = DOWNLOADS / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = make_manifest()
        write_json(manifest_path, manifest)
    records = manifest["files"]
    state = {
        "started_at": now(), "updated_at": now(), "status": "downloading",
        "total_files": len(records), "total_bytes": manifest["total_bytes"],
        "verified_files": 0, "verified_bytes": 0, "downloaded_bytes": 0,
        "files": {record["path"]: {"status": "pending"} for record in records},
    }
    lock = threading.Lock()

    def save() -> None:
        state["updated_at"] = now()
        write_json(DOWNLOADS / "status.json", state)

    def download(record: dict) -> None:
        path = Path(record["path"])
        with lock:
            state["files"][str(path)] = {"status": "downloading", "size": record["size"]}
            save()
        for attempt in range(1, 4):
            try:
                transfer_file(record)
                digest = verify_file(path, record)
                with lock:
                    state["files"][str(path)] = {
                        "status": "verified", "size": record["size"],
                        "hash": digest, "verified_at": now(),
                    }
                    state["verified_files"] += 1
                    state["verified_bytes"] += record["size"]
                    save()
                    print(f"VERIFIED {state['verified_files']}/{len(records)} {record['repo_id']}/{record['filename']}", flush=True)
                return
            except Exception as error:
                with lock:
                    state["files"][str(path)] = {
                        "status": "retrying" if attempt < 3 else "failed",
                        "attempt": attempt, "error_type": type(error).__name__,
                    }
                    save()
                if attempt == 3:
                    raise
                time.sleep(attempt * 2)

    save()
    print(f"MANIFEST {len(records)} files {manifest['total_bytes'] / 1e9:.3f} GB", flush=True)
    errors = []
    next_progress = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download, record): record for record in records}
        pending = set(futures)
        while pending:
            finished, pending = wait(pending, timeout=10)
            for future in finished:
                try:
                    future.result()
                except Exception as error:
                    record = futures[future]
                    errors.append({"file": record["filename"], "error_type": type(error).__name__})
            if time.monotonic() >= next_progress or not pending:
                downloaded = 0
                for record in records:
                    final, partial = Path(record["path"]), Path(record["path"] + ".part")
                    if final.exists():
                        downloaded += final.stat().st_size
                    elif partial.exists():
                        downloaded += partial.stat().st_size
                    else:
                        downloaded += max((item.stat().st_size for item in legacy_partials(record)), default=0)
                with lock:
                    state["downloaded_bytes"] = downloaded
                    save()
                    print(f"PROGRESS {downloaded / 1e9:.3f}/{manifest['total_bytes'] / 1e9:.3f} GB; verified {state['verified_files']}/{len(records)}", flush=True)
                next_progress = time.monotonic() + 30
    with lock:
        state["status"] = "failed" if errors else "complete"
        state["completed_at"] = now()
        state["errors"] = errors
        save()
    print(json.dumps({key: state[key] for key in ("status", "verified_files", "verified_bytes", "errors")}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

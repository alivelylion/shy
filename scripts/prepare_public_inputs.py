"""Extract verified public demo inputs, keeping them distinct from test ground truth."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "workspace" / "datasets" / "layerdiff_eval"
ARCHIVE = DATA / "miniset.zip"
DESTINATION = DATA / "inputs"


def main() -> None:
    status = json.loads((ROOT / "workspace/downloads/status.json").read_text(encoding="utf-8"))
    if status["files"].get(str(ARCHIVE), {}).get("status") != "verified":
        raise RuntimeError("The input archive has not passed its pinned hash check")
    inventory = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        if sum(item.file_size for item in archive.infolist()) > 10 * 1024**3:
            raise ValueError("Unexpected archive expansion size")
        for item in archive.infolist():
            destination = (DESTINATION / item.filename).resolve()
            if not destination.is_relative_to(DESTINATION.resolve()):
                raise ValueError("Archive member would escape the extraction directory")
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not accepted")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".extracting")
            with archive.open(item) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if temporary.stat().st_size != item.file_size:
                raise ValueError("Extracted file size mismatch")
            temporary.replace(destination)
            inventory.append({
                "archive_path": item.filename, "path": str(destination),
                "size": item.file_size, "zip_crc32": f"{item.CRC:08x}",
                "category": item.filename.split("/", 1)[0],
            })
    result = {
        "source_archive": str(ARCHIVE), "file_count": len(inventory),
        "total_uncompressed_bytes": sum(item["size"] for item in inventory),
        "categories": dict(Counter(item["category"] for item in inventory)),
        "dataset_role": "public_demo_inputs",
        "official_847_test_set": "not_verified",
        "ground_truth_available": False,
        "files": inventory,
    }
    path = ROOT / "workspace/downloads/public_inputs_inventory.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("file_count", "categories", "total_uncompressed_bytes", "ground_truth_available")}))


if __name__ == "__main__":
    main()

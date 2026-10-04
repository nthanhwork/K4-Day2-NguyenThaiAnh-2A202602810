"""Download the author's unmodified fold-0 CSVs, with reproducible provenance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

BASE = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"


def download_labels(labels_dir: Path, manifest_path: Path) -> dict:
    labels_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("labels", "train_subset0", "val_subset0", "test_subset0"):
        path = labels_dir / f"{name}.csv"
        url = f"{BASE}/{name}.csv"
        # Fetch even existing files to verify that local CSVs are unmodified.
        with urlopen(url, timeout=60) as response:
            content = response.read()
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
        required = {"Filename", "Label", "Species"} if name == "labels" else {"Filename", "Label"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Invalid CSV header downloaded from {url}")
        rows = list(reader)
        if not rows:
            raise ValueError(f"Empty CSV downloaded from {url}")
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"{path} differs from the author's CSV; not overwriting it")
        if not path.exists():
            tmp = path.with_suffix(".csv.tmp")
            tmp.write_bytes(content)
            tmp.replace(path)
        files[path.name] = {
            "url": url, "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content), "rows": len(rows),
        }
        print(f"{path.name}: {len(rows)} rows, SHA256 {files[path.name]['sha256']}")
    manifest = {"downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "files": files}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=Path("data/labels"))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("data_sources.json"))
    args = parser.parse_args()
    download_labels(args.labels_dir, args.manifest)

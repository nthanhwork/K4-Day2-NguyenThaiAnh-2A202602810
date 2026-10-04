"""Offline setup audit (CSV integrity, split checks, optional full image decode)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

import dataset
from train import Config, _verify_provenance, environment_info, write_json


def validate_setup(images_dir: str, labels_dir: str, out_dir: Path, decode_images=False):
    frames = dataset.load_split(labels_dir)
    reference = pd.read_csv(Path(labels_dir) / "labels.csv")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in Path(labels_dir).glob("*.csv") if p.name in {
                  "labels.csv", "train_subset0.csv", "val_subset0.csv", "test_subset0.csv"}}
    _verify_provenance(hashes, 0)
    report = dataset.check_split(*frames, images_dir, reference_df=reference,
                                 verify_images=decode_images)
    write_json(out_dir / "split_check.json", report)
    write_json(out_dir / "environment.json", environment_info(torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    cfg = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", default=cfg.images_dir)
    parser.add_argument("--labels-dir", default=cfg.labels_dir)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("validation"))
    parser.add_argument("--decode-images", action="store_true")
    args = parser.parse_args()
    validate_setup(args.images_dir, args.labels_dir, args.out_dir, args.decode_images)

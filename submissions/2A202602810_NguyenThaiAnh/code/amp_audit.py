"""Audit AMP on one continuation TRAIN epoch in memory; preserve pilot checkpoints."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

import dataset
import losses
import model
import train


class LoggedScaler:
    def __init__(self, scaler, optimizer):
        self.scaler = scaler
        self.rows = []
        self.current = None
        self.optimizer_steps = 0
        self.handle = optimizer.register_step_post_hook(self._optimizer_updated)

    def _optimizer_updated(self, *args):
        self.optimizer_steps += 1

    def is_enabled(self):
        return self.scaler.is_enabled()

    def get_scale(self):
        return self.scaler.get_scale()

    def scale(self, loss):
        self.current = {"batch": len(self.rows) + 1, "loss": float(loss.detach()),
                        "scale_before": self.get_scale(), "optimizer_steps_before": self.optimizer_steps}
        return self.scaler.scale(loss)

    def unscale_(self, optimizer):
        return self.scaler.unscale_(optimizer)

    def step(self, optimizer):
        return self.scaler.step(optimizer)

    def update(self):
        self.scaler.update()
        self.current["scale_after"] = self.get_scale()
        self.current["optimizer_updated"] = self.optimizer_steps > self.current["optimizer_steps_before"]
        self.rows.append(self.current)


def audit_one(cfg, out_dir):
    path = train.run_dir(cfg)
    checkpoint_path = path / "last.pt"
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_epoch = source["epoch"]
    if cfg.save_test_predictions or cfg.ema_decay is not None or cfg.loss != "ce":
        raise ValueError("This audit supports the CE/no-EMA/no-test T00 pilots")
    train.set_seed(cfg.seed, cfg.deterministic)
    device = torch.device(cfg.device)
    train_df, val_df, _ = dataset.load_split(cfg.labels_dir, cfg.fold)
    metadata = json.loads((path / "model.json").read_text())
    kw = {key: metadata["preprocessing"][key] for key in ("mean", "std", "interpolation", "resize_size")}
    train_loader = dataset.make_loader(train_df, cfg.images_dir,
        dataset.build_transforms(True, cfg.img_size, cfg.aug, **kw), cfg.batch_size, True,
        cfg.sampler, cfg.num_workers, seed=cfg.seed)
    # No val images are loaded; this loader exists only to restore its RNG state.
    val_loader = dataset.make_loader(val_df, cfg.images_dir,
        dataset.build_transforms(False, cfg.img_size, **kw), cfg.batch_size, False,
        num_workers=cfg.num_workers, seed=cfg.seed + 1)
    network = model.build_model(cfg.backbone, init="scratch").to(device)
    network.load_state_dict(source["model"])
    optimizer = train.build_optimizer(network, cfg)
    scheduler = train.build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    optimizer.load_state_dict(source["optimizer"])
    scheduler.load_state_dict(source["scheduler"])
    scaler.load_state_dict(source["scaler"])
    train._restore_rng(source["rng"], train_loader, val_loader)
    initial_scale = scaler.get_scale()
    scheduler_before = scheduler.last_epoch
    previous = source["history"][-1]
    source_growth_tracker = source["scaler"]["_growth_tracker"]
    del source
    logged = LoggedScaler(scaler, optimizer)
    actual_clip = torch.nn.utils.clip_grad_norm_

    def record_clip(parameters, *args, **kwargs):
        norm = float(actual_clip(parameters, *args, **kwargs))
        logged.current["gradient_norm_finite"] = math.isfinite(norm)
        logged.current["gradient_norm"] = norm if math.isfinite(norm) else None
        return torch.tensor(norm, device=device)

    print(f"Audit {cfg.backbone}: continuation epoch {source_epoch + 1}, scale={initial_scale:g}", flush=True)
    try:
        with patch("train.torch.nn.utils.clip_grad_norm_", side_effect=record_clip):
            stats = train.train_one_epoch(network, train_loader, losses.build_criterion("ce"),
                optimizer, scheduler, logged, cfg, device)
    finally:
        logged.handle.remove()
    rows = logged.rows
    skipped = [r["batch"] for r in rows if not r["optimizer_updated"]]
    protected = all(not r["optimizer_updated"] and r["scale_after"] < r["scale_before"]
                    for r in rows if not r["gradient_norm_finite"])
    weights_finite = all(bool(torch.isfinite(v).all()) for v in network.state_dict().values()
                         if v.is_floating_point())
    passed = (all(math.isfinite(r["loss"]) for r in rows) and protected and weights_finite
              and logged.optimizer_steps == stats["optimizer_updates"]
              and scheduler.last_epoch - scheduler_before == stats["optimizer_updates"])
    pd.DataFrame(rows).to_csv(out_dir / f"{cfg.backbone}_steps.csv", index=False)
    result = {
        "backbone": cfg.backbone, "source_exp_id": cfg.exp_id,
        "source_checkpoint_sha256": checkpoint_hash, "source_epoch": source_epoch,
        "source_skipped_updates": previous["skipped_updates"],
        "source_last_skip_batch_inferred": len(train_loader) - source_growth_tracker,
        "source_successful_steps_since_last_skip": source_growth_tracker,
        "audited_epoch": source_epoch + 1, "train_batches": len(rows),
        "amp_scale_before": initial_scale, "amp_scale_after": scaler.get_scale(),
        "optimizer_updates": stats["optimizer_updates"], "skipped_updates": len(skipped),
        "skipped_batches": skipped, "all_losses_finite": all(math.isfinite(r["loss"]) for r in rows),
        "all_nonfinite_gradient_norms_protected": protected,
        "all_model_state_finite": weights_finite,
        "scheduler_step_delta": scheduler.last_epoch - scheduler_before,
        "optimizer_step_hook_count": logged.optimizer_steps,
        "train_loss": stats["train_loss"], "passed": passed,
        "original_checkpoint_unchanged": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_hash,
    }
    result["passed"] = passed and result["original_checkpoint_unchanged"]
    print(f"{cfg.backbone}: updates={stats['optimizer_updates']}, skips={skipped}, "
          f"scale={scaler.get_scale():g}, passed={result['passed']}", flush=True)
    del network, optimizer, scheduler, scaler, logged, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="P62_session01")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.exit(2, "Run from a host terminal with GPU access.\n")
    if Path(args.session).name != args.session or args.session in {".", ".."}:
        parser.error("session must be a basename")
    torch.set_num_threads(4)
    folder = Path(__file__).parent / "validation/amp_audit" / args.session
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    paths = sorted((train.ROOT / "runs/pilot62").glob(f"{args.session}_*/seed0/config.json"))
    if len(paths) != 5:
        parser.error("Expected 5 pilot configs")
    for path in paths:
        cfg = train.Config(**json.loads(path.read_text()))
        results.append(audit_one(cfg, folder))
        train.write_json(folder / "summary.json", {
            "source_session": args.session, "scope": "continuation TRAIN epoch in memory, no val/test forward or checkpoint writes",
            "results": results, "all_passed": len(results) == 5 and all(r["passed"] for r in results),
        })
    pd.DataFrame(results).to_csv(folder / "summary.csv", index=False)
    if not all(r["passed"] for r in results):
        raise SystemExit("AMP audit failed; inspect the step logs.")


if __name__ == "__main__":
    main()

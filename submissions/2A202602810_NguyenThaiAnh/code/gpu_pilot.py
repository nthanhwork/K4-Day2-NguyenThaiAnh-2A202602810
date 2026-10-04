"""Short throughput pilot: estimates full-epoch runtime, never reads test images."""
from __future__ import annotations

import argparse
import gc
import itertools
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import dataset
import model
from train import Config, build_optimizer, build_scheduler, set_seed, train_one_epoch, write_json

BACKBONES = ["resnet50", "resnext50_32x4d", "convnext_tiny",
             "deit_small_patch16_224", "efficientnet_b0"]


def pilot(name, cfg, train_steps, val_steps, warmup):
    set_seed(cfg.seed, cfg.deterministic)
    train_df, val_df, _ = dataset.load_split(cfg.labels_dir)
    train_loader = dataset.make_loader(train_df, cfg.images_dir,
        dataset.build_transforms(True, cfg.img_size), cfg.batch_size, True,
        num_workers=cfg.num_workers, seed=cfg.seed)
    val_loader = dataset.make_loader(val_df, cfg.images_dir,
        dataset.build_transforms(False, cfg.img_size), cfg.batch_size, False,
        num_workers=cfg.num_workers, seed=cfg.seed + 1)
    # Scratch weights: same architecture/optimizer workload as finetuning, no download.
    network = model.build_model(name, init="scratch", cache_dir=cfg.cache_dir).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = build_optimizer(network, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    iterator = iter(train_loader)
    train_one_epoch(network, itertools.islice(iterator, warmup), criterion,
                     optimizer, scheduler, scaler, cfg, "cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    stats = train_one_epoch(network, itertools.islice(iterator, train_steps), criterion,
                            optimizer, scheduler, scaler, cfg, "cuda")
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    network.eval()
    val_iterator = iter(val_loader)
    # Match train.evaluate(): validation forward in FP32, not autocast.
    with torch.inference_mode():
        for x, y, _ in itertools.islice(val_iterator, warmup):
            criterion(network(x.cuda(non_blocking=True)).float(), y.cuda(non_blocking=True))
        torch.cuda.synchronize()
        start = time.perf_counter()
        val_images = 0
        for x, y, _ in itertools.islice(val_iterator, val_steps):
            logits = network(x.cuda(non_blocking=True)).float()
            criterion(logits, y.cuda(non_blocking=True))
            logits.cpu().numpy()  # same device-to-host logits transfer as evaluate()
            y.numpy()
            val_images += len(x)
        torch.cuda.synchronize()
        val_seconds = time.perf_counter() - start
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    measured_steps = stats["train_images"] / cfg.batch_size
    estimate = train_seconds / measured_steps * train_batches + val_seconds / val_steps * val_batches
    result = {
        "backbone": name, "batch_size": cfg.batch_size, "img_size": cfg.img_size,
        "train_steps_measured": measured_steps, "val_steps_measured": val_steps,
        "warmup_steps_per_phase": warmup, "num_workers": cfg.num_workers,
        "train_amp": cfg.amp, "validation_dtype": "FP32",
        "deterministic": cfg.deterministic, "pretrained_loaded": False,
        "train_seconds_measured": train_seconds, "val_seconds_measured": val_seconds,
        "train_images_per_second": stats["train_images"] / train_seconds,
        "val_images_per_second": val_images / val_seconds,
        "train_batches_full_epoch": train_batches, "val_batches_full_epoch": val_batches,
        "epoch_seconds_estimated": estimate,
        "minutes_for_12_epochs_estimated": estimate * 12 / 60,
        "peak_allocated_gib": peak_allocated, "peak_reserved_gib": peak_reserved,
        "optimizer_updates": stats["optimizer_updates"], "skipped_updates": stats["skipped_updates"],
        "scope": "measured short train/val pilot; extrapolation excludes downloads/checkpoint/profiling/plots",
    }
    del iterator, val_iterator, network, optimizer, scheduler, scaler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=BACKBONES)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=30)
    parser.add_argument("--val-steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("validation") / "gpu_pilot.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.exit(2, "CUDA unavailable in this process; run from a host terminal with GPU access.\n")
    if min(args.train_steps, args.val_steps, args.warmup, args.batch_size) < 1:
        parser.error("step counts, warmup and batch_size must be positive")
    torch.set_num_threads(4)
    cfg = Config(batch_size=args.batch_size, num_workers=args.num_workers,
                 device="cuda", init="scratch")
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0), "device_count": torch.cuda.device_count(),
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "kind": "runtime estimate from a short real-data pilot, NOT model-quality results",
        "results": [],
    }
    for name in args.models:
        print(f"Pilot {name}: train {args.train_steps} steps + val {args.val_steps} steps", flush=True)
        try:
            result = pilot(name, cfg, args.train_steps, args.val_steps, args.warmup)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            report["results"].append({"backbone": name, "status": "OOM",
                                      "batch_size": args.batch_size})
            write_json(args.out, report)
            raise
        report["results"].append(result)
        write_json(args.out, report)
        print(f"{name}: ~{result['epoch_seconds_estimated']:.1f}s/epoch, "
              f"~{result['minutes_for_12_epochs_estimated']:.1f}min/12 epochs, "
              f"peak allocated {result['peak_allocated_gib']:.2f}GiB", flush=True)

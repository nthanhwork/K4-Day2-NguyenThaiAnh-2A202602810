"""One training entry point for B/T/F runs, selected exclusively by validation F1."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
SUBMISSION = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import dataset
import losses
import model as model_utils
from eval import compute_metrics, save_predictions


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = str(ROOT / "data")
    labels_dir: str = str(ROOT / "data/labels")
    out_dir: str = str(ROOT / "runs")
    pred_dir: str = str(SUBMISSION / "predictions")
    save_test_predictions: bool = False
    # Extensions to the starter interface.
    curves_dir: str = str(SUBMISSION / "curves")
    cache_dir: str = str(ROOT / ".cache/models")
    device: str = "auto"
    deterministic: bool = True
    resume: bool = False
    grad_clip: float | None = 1.0


def validate_config(cfg: Config) -> None:
    if not cfg.exp_id or Path(cfg.exp_id).name != cfg.exp_id or cfg.exp_id in {".", ".."}:
        raise ValueError("exp_id must be a nonempty basename")
    for key in ("epochs", "batch_size", "img_size"):
        if type(getattr(cfg, key)) is not int or getattr(cfg, key) <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg.seed) is not int or not 0 <= cfg.seed < 2**32 or cfg.num_workers < 0:
        raise ValueError("Invalid seed/num_workers")
    if type(cfg.fold) is not int or not 0 <= cfg.fold <= 4:
        raise ValueError("fold must be 0..4; use fold 0 for the required lab")
    if cfg.init not in {"scratch", "frozen", "finetune"} or cfg.aug not in {"basic", "color", "trivial", "randaug"}:
        raise ValueError("Invalid initialization or augmentation")
    if cfg.loss not in {"ce", "ls", "focal", "ce_weighted"} or cfg.mix not in {None, "mixup", "cutmix"}:
        raise ValueError("Invalid loss/mix")
    if cfg.sampler not in {None, "balanced"}:
        raise ValueError("Invalid sampler")
    for key in ("drop_rate", "mix_alpha", "label_smoothing", "focal_gamma", "lr_backbone",
                "lr_head", "weight_decay", "warmup_epochs"):
        if not math.isfinite(getattr(cfg, key)):
            raise ValueError(f"{key} must be finite")
    if not 0 <= cfg.drop_rate < 1 or not 0 <= cfg.label_smoothing < 1:
        raise ValueError("drop_rate/label_smoothing must be in [0,1)")
    if min(cfg.lr_backbone, cfg.lr_head, cfg.mix_alpha) <= 0 or min(cfg.weight_decay, cfg.focal_gamma) < 0:
        raise ValueError("Invalid LR, mix_alpha, weight_decay or focal_gamma")
    if not 0 <= cfg.warmup_epochs < cfg.epochs:
        raise ValueError("warmup_epochs must be >=0 and <epochs")
    if cfg.ema_decay is not None and (not math.isfinite(cfg.ema_decay) or not 0 <= cfg.ema_decay < 1):
        raise ValueError("ema_decay must be in [0,1)")
    if cfg.class_weight_beta is not None and (
            not math.isfinite(cfg.class_weight_beta) or not 0 <= cfg.class_weight_beta < 1):
        raise ValueError("class_weight_beta must be in [0,1)")
    if cfg.grad_clip is not None and (not math.isfinite(cfg.grad_clip) or cfg.grad_clip <= 0):
        raise ValueError("grad_clip must be finite and positive")
    if cfg.device != "auto" and torch.device(cfg.device).type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices: auto, cpu, cuda[:index]")


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    if split not in {"val", "test"}:
        raise ValueError("split must be val/test")
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def set_seed(seed: int, deterministic: bool = True) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def build_optimizer(model, cfg: Config):
    return torch.optim.AdamW(model_utils.param_groups(
        model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Initialize warmup LR for step 0; advance only after successful optimizer steps."""
    if steps_per_epoch < 1:
        raise ValueError("Training loader has no batches")
    total = cfg.epochs * steps_per_epoch
    warmup = round(cfg.warmup_epochs * steps_per_epoch)

    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """EMA parameters, copied BN buffers (not averaged running statistics)."""
    def __init__(self, model, decay: float):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0,1)")
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        source = dict(model.named_parameters())
        for name, p in self.model.named_parameters():
            p.lerp_(source[name].detach(), 1 - self.decay)
        buffers = dict(model.named_buffers())
        for name, buffer in self.model.named_buffers():
            buffer.copy_(buffers[name])
        self.model.eval()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    model_utils.set_train_mode(model)
    device = torch.device(device)
    total_loss = 0.0
    total = 0
    updates = skipped = 0
    amp_enabled = cfg.amp and device.type == "cuda"
    start = time.perf_counter()
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        targets = None
        if cfg.mix is not None:
            x, targets = losses.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y) if targets is None else losses.mixed_loss(criterion, logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; stop and inspect inputs/config")
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        if cfg.grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        did_step = not scaler.is_enabled() or scaler.get_scale() >= scale_before
        if did_step:
            scheduler.step()
            if ema is not None:
                ema.update(model)
            updates += 1
        else:
            skipped += 1
        total_loss += float(loss.detach()) * len(x)
        total += len(x)
    if not total:
        raise ValueError("Empty training loader")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "train_loss": total_loss / total, "train_seconds": time.perf_counter() - start,
        "train_images": total, "optimizer_updates": updates, "skipped_updates": skipped,
        "lr": float(optimizer.param_groups[0]["lr"]),
        "group_lrs": {g["group_name"]: float(g["lr"]) for g in optimizer.param_groups},
    }


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    device = torch.device(device)
    filenames, labels, all_logits = [], [], []
    total_loss = 0.0
    for x, y, names in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x).float()
        if logits.shape != (len(x), 9) or not torch.isfinite(logits).all():
            raise ValueError("Expected finite N×9 logits")
        total_loss += float(criterion(logits, y)) * len(x)
        filenames.extend(names)
        labels.append(y.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
    if not filenames:
        raise ValueError("Empty evaluation loader")
    return filenames, np.concatenate(labels), np.concatenate(all_logits), total_loss / len(filenames)


def probabilities(logits):
    return torch.from_numpy(np.asarray(logits)).softmax(1).numpy()


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="val")
    axes[0].set_ylabel("Loss (configured criterion)")
    axes[1].plot(frame["epoch"], frame["val_macro_f1"], label="val macro-F1")
    axes[1].plot(frame["epoch"], frame["val_top1"], label="val top-1")
    axes[1].set_ylabel("Score (0–1)")
    axes[2].plot(frame["epoch"], frame["lr"], label="LR after epoch")
    axes[2].set_ylabel("Learning rate")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def environment_info(device) -> dict:
    names = ("torch", "torchvision", "timm", "numpy", "pandas", "Pillow", "matplotlib",
             "scikit-learn", "openpyxl", "fvcore")
    return {
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in names},
        "torch_build": torch.__version__, "cuda_build": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "ema_buffers": "copy from live model each successful optimizer step",
    }


def _rng_state(train_loader, val_loader):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_loader": train_loader.generator.get_state(),
        "val_loader": val_loader.generator.get_state(),
    }


def _restore_rng(state, train_loader, val_loader):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    train_loader.generator.set_state(state["train_loader"].cpu())
    val_loader.generator.set_state(state["val_loader"].cpu())


def _save_checkpoint(path, state):
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _verify_provenance(hashes, fold):
    provenance = Path(__file__).with_name("data_sources.json")
    if fold == 0 and provenance.exists():
        original = json.loads(provenance.read_text())["files"]
        if any(hashes[name] != metadata["sha256"] for name, metadata in original.items()):
            raise ValueError("CSV hashes differ from the downloaded originals")


def run(cfg: Config, *, stop_after_epoch: int | None = None) -> dict:
    """Train full train split; select by val F1; test only when explicitly enabled.

    Current test branch is I00 (single-view FP32). Later inference methods must be
    integrated before enabling test for final TTA/ensemble/calibration configs.
    """
    validate_config(cfg)
    # A pilot keeps the full scheduler horizon/config for exact epoch-boundary resume.
    if stop_after_epoch is not None:
        if type(stop_after_epoch) is not int or not 1 <= stop_after_epoch <= cfg.epochs:
            raise ValueError("stop_after_epoch must be an integer in 1..epochs")
        if cfg.save_test_predictions:
            raise ValueError("A pilot/early-stop run must disable test predictions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(cfg.seed, cfg.deterministic)
    path = run_dir(cfg)
    config_path = path / "config.json"
    if cfg.save_test_predictions and ((path / "test_started.json").exists() or pred_path(cfg, "test").exists()):
        raise FileExistsError("Test was already started/exported; do not rerun or overwrite test")
    if config_path.exists():
        if not cfg.resume:
            raise FileExistsError(f"{path} already exists; use resume=true for the same configuration")
        previous = json.loads(config_path.read_text())
        current = asdict(cfg)
        previous.pop("resume", None)
        current.pop("resume", None)
        if previous != current:
            raise ValueError("Resume config differs from original run")
    elif cfg.resume:
        raise FileNotFoundError("No original config to resume")
    else:
        if pred_path(cfg, "val").exists():
            raise FileExistsError("Val prediction ID already exists; choose a fresh exp_id/out_dir")
        write_json(config_path, asdict(cfg))
    write_json(path / "environment.json", environment_info(device))
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    reference = pd.read_csv(Path(cfg.labels_dir) / "labels.csv")
    report = dataset.check_split(train_df, val_df, test_df, cfg.images_dir, reference_df=reference)
    write_json(path / "split_check.json", report)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [Path(cfg.labels_dir) / "labels.csv", *[
                  Path(cfg.labels_dir) / f"{split}_subset{cfg.fold}.csv"
                  for split in ("train", "val", "test")]]}
    _verify_provenance(hashes, cfg.fold)
    write_json(path / "split_hashes.json", hashes)
    model = model_utils.build_model(cfg.backbone, num_classes=9, drop_rate=cfg.drop_rate,
                                    init=cfg.init, cache_dir=cfg.cache_dir).to(device)
    pretrained_cfg = dict(getattr(model, "pretrained_cfg", {}))
    preprocessing = {
        "img_size": cfg.img_size, "resize_size": round(cfg.img_size * 256 / 224),
        "mean": pretrained_cfg.get("mean", dataset.IMAGENET_MEAN),
        "std": pretrained_cfg.get("std", dataset.IMAGENET_STD),
        "interpolation": pretrained_cfg.get("interpolation", "bicubic"),
        "aug": cfg.aug, "validation": "resize then center crop, no random augmentation",
    }
    write_json(path / "model.json", {
        "requested_name": cfg.backbone, "resolved_name": model.lab_model_name,
        "pretrained_loaded": model.lab_pretrained, "pretrained_cfg": pretrained_cfg,
        "preprocessing": preprocessing,
        "optimizer_groups": "backbone/head LR × decay/no_decay; all bias/norm exempt",
    })
    transform_kw = {key: preprocessing[key] for key in ("mean", "std", "interpolation", "resize_size")}
    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, dataset.build_transforms(True, cfg.img_size, cfg.aug, **transform_kw),
        cfg.batch_size, True, cfg.sampler, cfg.num_workers, seed=cfg.seed)
    val_transform = dataset.build_transforms(False, cfg.img_size, **transform_kw)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, val_transform,
                                     cfg.batch_size, False, num_workers=cfg.num_workers, seed=cfg.seed + 1)
    weights = None
    if cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().reindex(range(9), fill_value=0).to_numpy(copy=True)
        weights = losses.class_weights(counts, cfg.class_weight_beta or 0.0)
    criterion = losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing,
                                       gamma=cfg.focal_gamma, weight=weights).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None
    params = model_utils.count_params(model)
    gmacs = model_utils.count_gmacs(model, cfg.img_size)
    write_json(path / "mac_profile.json", model.lab_mac_profile)
    best_f1, best_epoch, history, first_epoch = -1.0, 0, [], 1
    if cfg.resume:
        # Only read our own local training checkpoint (contains Python/NumPy RNG).
        state = torch.load(path / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        if ema is not None:
            ema.model.load_state_dict(state["ema"])
        best_f1, best_epoch = state["best_f1"], state["best_epoch"]
        history, first_epoch = state["history"], state["epoch"] + 1
        _restore_rng(state["rng"], train_loader, val_loader)
    last_epoch = cfg.epochs if stop_after_epoch is None else stop_after_epoch
    if device.type == "cuda":
        torch.cuda.empty_cache()  # Release the profiler's unused cached allocations once.
    for epoch in range(first_epoch, last_epoch + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        selected = ema.model if ema is not None else model
        val_started = time.perf_counter()
        _, val_y, val_logits, val_loss = evaluate(selected, val_loader, criterion, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        val_seconds = time.perf_counter() - val_started
        probs = probabilities(val_logits)
        metrics = compute_metrics(val_y, probs.argmax(1), probs)
        row = {"epoch": epoch, **{k: v for k, v in stats.items() if k != "group_lrs"},
               "val_seconds": val_seconds, "epoch_train_val_seconds": time.perf_counter() - epoch_started,
               "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
               "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else None,
               "group_lrs_json": json.dumps(stats["group_lrs"]), "val_loss": val_loss,
               **{f"val_{key}": float(metrics[key]) for key in ("macro_f1", "top1", "balanced_acc", "ece", "nll")}}
        history.append(row)
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = float(metrics["macro_f1"]), epoch
            _save_checkpoint(path / "best.pt", {
                "model": selected.state_dict(), "epoch": epoch, "val_macro_f1": best_f1,
                "weights": "EMA" if ema is not None else "live", "config": asdict(cfg),
            })
        pd.DataFrame(history).to_csv(path / "history.csv", index=False)
        _save_checkpoint(path / "last.pt", {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "ema": ema.model.state_dict() if ema is not None else None,
            "epoch": epoch, "best_f1": best_f1, "best_epoch": best_epoch,
            "history": history, "rng": _rng_state(train_loader, val_loader),
        })
        plot_curves(history, Path(cfg.curves_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{cfg.backbone}.png",
                    f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed}")
        print(f"{cfg.exp_id} seed={cfg.seed} epoch={epoch}/{cfg.epochs} "
              f"train_loss={stats['train_loss']:.4f} val_macro_f1={metrics['macro_f1']:.4f}", flush=True)
    best = torch.load(path / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model"])

    def export(split, loader):
        names, labels, logits, loss = evaluate(model, loader, criterion, device)
        probs = probabilities(logits)
        np.save(path / f"{split}_logits.npy", logits)
        np.save(path / f"{split}_filenames.npy", np.asarray(names, dtype=str))
        np.save(path / f"{split}_labels.npy", labels)
        save_predictions(pred_path(cfg, split), names, labels, probs)
        m = compute_metrics(labels, probs.argmax(1), probs)
        result = {key: float(m[key]) for key in ("macro_f1", "top1", "balanced_acc", "ece", "nll")}
        result["loss"] = loss
        result["n"] = len(names)
        return result

    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "best_epoch": best_epoch,
        "completed_epochs": len(history), "scheduled_epochs": cfg.epochs,
        "training_complete": len(history) == cfg.epochs, "stop_after_epoch": stop_after_epoch,
        "params_m": params, "gmacs": gmacs, "mac_profile": model.lab_mac_profile,
        "train_seconds_per_epoch": float(np.mean([r["train_seconds"] for r in history])),
        "weights": best["weights"], "val": export("val", val_loader),
    }
    if cfg.save_test_predictions:
        marker = path / "test_started.json"
        with marker.open("x") as f:
            json.dump({"seed": cfg.seed, "inference": "I00 FP32 single-view",
                       "note": "Configuration fixed; never use test to select"}, f)
        # This is the only creation of a test loader in run().
        test_loader = dataset.make_loader(test_df, cfg.images_dir, val_transform,
                                          cfg.batch_size, False, num_workers=cfg.num_workers, seed=cfg.seed + 2)
        summary["test"] = export("test", test_loader)
        write_json(path / "test_completed.json", {"prediction": str(pred_path(cfg, "test"))})
    write_json(path / "summary.json", summary)
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    hints = get_type_hints(Config)
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE, got {pair!r}")
        key, raw = pair.split("=", 1)
        if key not in hints:
            raise ValueError(f"Unknown Config field: {key}")
        if key in overrides:
            raise ValueError(f"Duplicate Config field: {key}")
        kind = hints[key]
        optional = get_origin(kind) is types.UnionType
        candidates = get_args(kind) if optional else (kind,)
        if type(None) in candidates and raw.lower() in {"none", "null"}:
            overrides[key] = None
            continue
        kind = next(k for k in candidates if k is not type(None))
        if kind is bool:
            if raw.lower() not in {"true", "false", "1", "0"}:
                raise ValueError(f"{key}: bool must be true/false/1/0")
            value = raw.lower() in {"true", "1"}
        else:
            try:
                value = kind(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key}: invalid {kind.__name__} value {raw!r}") from exc
        overrides[key] = value
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    try:
        cfg = Config(**parse_overrides(args.set))
        result = run(cfg)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Shared notebook runner: isolated training processes, per-run locks and output audit."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")

import numpy as np
import pandas as pd
import torch
import train
from eval import read_pred

BACKBONES = {
    "B01": ("resnet50", 3),
    "B02": ("resnext50_32x4d", 2),
    "B03": ("convnext_tiny", 1),
    "B04": ("deit_small_patch16_224", 1),
    "B05": ("efficientnet_b0", 2),
}


def t00_config(exp_id, backbone, *, seed=0, epochs=12, batch_size=64, num_workers=2):
    return train.Config(
        exp_id=exp_id, backbone=backbone, seed=seed, epochs=epochs, batch_size=batch_size,
        num_workers=num_workers, init="finetune", img_size=224, aug="basic", loss="ce",
        lr_backbone=1e-4, lr_head=1e-3, weight_decay=.05, warmup_epochs=1,
        mix=None, sampler=None, label_smoothing=0, ema_decay=None,
        amp=True, device="cuda:0", deterministic=True, save_test_predictions=False)


def training_status(cfg):
    train.validate_config(cfg)
    if cfg.save_test_predictions:
        raise ValueError("Backbone notebooks must keep test predictions disabled")
    path = train.run_dir(cfg)
    if not (path / "config.json").exists():
        return "new"
    old = json.loads((path / "config.json").read_text())
    current = asdict(cfg)
    old.pop("resume", None)
    current.pop("resume", None)
    if old != current:
        raise ValueError(f"{path}: config differs; retain the original config or choose a fresh EXP_ID")
    if (path / "summary.json").is_file():
        summary = json.loads((path / "summary.json").read_text())
        if summary.get("training_complete") and summary.get("completed_epochs") == cfg.epochs:
            return "complete"
    if not (path / "last.pt").exists():
        raise FileExistsError(f"{path}: no completed epoch checkpoint; choose a fresh EXP_ID")
    return "resume"


def verify_outputs(cfg):
    path = train.run_dir(cfg)
    required = ("config.json", "environment.json", "model.json", "split_check.json", "split_hashes.json",
                "history.csv", "best.pt", "last.pt", "summary.json", "mac_profile.json",
                "val_logits.npy", "val_labels.npy", "val_filenames.npy")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path}: missing {missing}")
    if training_status(cfg) != "complete":
        raise ValueError("Training has not completed the configured number of epochs")
    summary = json.loads((path / "summary.json").read_text())
    history = pd.read_csv(path / "history.csv")
    if history["epoch"].tolist() != list(range(1, cfg.epochs + 1)):
        raise ValueError("History does not cover every configured epoch")
    values = history[["train_loss", "val_loss", "val_macro_f1", "val_top1"]].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Non-finite loss or metric in history")
    selected_epoch = int(history.loc[history["val_macro_f1"].idxmax(), "epoch"])
    if selected_epoch != summary["best_epoch"]:
        raise ValueError("Checkpoint selection disagrees with earliest highest val macro-F1")
    reference = pd.read_csv(Path(cfg.labels_dir) / f"val_subset{cfg.fold}.csv")
    prediction = read_pred(str(train.pred_path(cfg, "val")))
    if prediction.filenames.tolist() != reference["Filename"].tolist():
        raise ValueError("Validation prediction filenames differ from original split/order")
    np.testing.assert_array_equal(prediction.y_true, reference["Label"].to_numpy())
    if summary["val"]["n"] != len(reference):
        raise ValueError("Validation count mismatch")
    np.testing.assert_array_equal(np.load(path / "val_filenames.npy"), prediction.filenames)
    np.testing.assert_array_equal(np.load(path / "val_labels.npy"), prediction.y_true)
    logits = np.load(path / "val_logits.npy")
    if logits.shape != (len(reference), 9) or not np.isfinite(logits).all():
        raise ValueError("Invalid validation logits")
    z = logits.astype(np.float64) - logits.max(1, keepdims=True)
    probs = np.exp(z)
    probs /= probs.sum(1, keepdims=True)
    np.testing.assert_allclose(probs, prediction.probs, atol=1e-6, rtol=1e-5)
    if "test" in summary or (path / "test_started.json").exists() or train.pred_path(cfg, "test").exists():
        raise ValueError("Unexpected test evaluation in a backbone run")
    curve = Path(cfg.curves_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{cfg.backbone}.png"
    if not curve.is_file():
        raise FileNotFoundError(curve)
    metadata = json.loads((path / "model.json").read_text())
    if cfg.init != "scratch" and not metadata["pretrained_loaded"]:
        raise ValueError("Expected pretrained backbone")
    return summary, history, metadata


def execute_training(cfg, planned_group):
    """Called in the child; the OS releases this lock even after a process crash."""
    train.validate_config(cfg)
    path = train.run_dir(cfg)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "notebook_train.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"{cfg.exp_id}/seed{cfg.seed} is already training in another process") from exc
        status = training_status(cfg)
        if status == "complete":
            result, _, _ = verify_outputs(cfg)
            print(f"{cfg.exp_id}: read completed {cfg.epochs}-epoch run; no retraining", flush=True)
            return result
        actual = replace(cfg, resume=status == "resume")
        torch.set_num_threads(4)
        started = datetime.now(timezone.utc).isoformat()
        wall_start = time.perf_counter()
        print(f"{cfg.exp_id}: {status}, {cfg.backbone}, seed={cfg.seed}, epochs={cfg.epochs}, "
              f"batch={cfg.batch_size}, planned group={planned_group}, pid={os.getpid()}", flush=True)
        result = train.run(actual)
        verify_outputs(cfg)
        record = {
            "mode": status, "pid": os.getpid(), "parallel_group_planned": planned_group,
            "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds_in_this_invocation": time.perf_counter() - wall_start,
            "timing_scope": "user-scheduled training; timings are not an isolated backbone benchmark",
        }
        with (path / "notebook_invocations.jsonl").open("a") as output:
            output.write(json.dumps(record) + "\n")
        return result


def run_in_subprocess(cfg, planned_group):
    """Stream the child's output; use a dedicated notebook kernel for each simultaneous job."""
    train.validate_config(cfg)
    env = dict(os.environ)
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    env["HF_HOME"] = str(ROOT / ".cache/huggingface")
    env["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib" / cfg.exp_id / f"seed{cfg.seed}")
    env["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, str(Path(__file__).resolve()), "--config-json", json.dumps(asdict(cfg)),
               "--planned-group", str(planned_group)]
    log_path = train.ROOT / "runs/notebook_logs" / f"{cfg.exp_id}_seed{cfg.seed}_{time.time_ns()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Log:", log_path, flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            raise
        finally:
            process.stdout.close()
    if code:
        raise RuntimeError(f"Training exited with status {code}; inspect {log_path}. "
                           "If OOM, run this model alone with the same config/checkpoint.")
    return verify_outputs(cfg)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--planned-group", type=int, required=True)
    args = parser.parse_args()
    execute_training(train.Config(**json.loads(args.config_json)), args.planned_group)


if __name__ == "__main__":
    main()

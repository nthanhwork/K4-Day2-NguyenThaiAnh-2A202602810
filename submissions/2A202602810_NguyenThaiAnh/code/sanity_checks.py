"""Sanity checks on real TRAIN images; diagnostic outputs, never test evaluation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import time
from pathlib import Path
from unittest.mock import patch

# This check uses public timm weights; keep downloads local and avoid stale account tokens.
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HOME"] = str(Path(__file__).resolve().parents[3] / ".cache/huggingface")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import dataset
import losses
import model
import train
from eval import read_pred, save_predictions


def state_digest(network):
    digest = hashlib.sha256()
    for name, value in network.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "validation/sanity")
    args = parser.parse_args()
    if args.max_steps < 20:
        parser.error("max-steps must be at least 20")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.exit(2, "Run in a host terminal with GPU access, or pass --device cpu.\n")
    torch.set_num_threads(4)
    train.set_seed(42)
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = train.Config(seed=42, backbone="resnet50", batch_size=18, epochs=1,
                       warmup_epochs=0, num_workers=0, device=str(device))
    # Reading the train CSV directly avoids creating any val/test image loader.
    frame = pd.read_csv(Path(cfg.labels_dir) / "train_subset0.csv")
    fixed = pd.concat([frame[frame.Label == label].sample(2, random_state=42 + label)
                       for label in range(9)], ignore_index=True)
    fixed.to_csv(args.out / "train_batch.csv", index=False)
    print("Loading ImageNet pretrained ResNet-50; cache:", cfg.cache_dir, flush=True)
    network = model.build_model(cfg.backbone, cache_dir=cfg.cache_dir).to(device)
    pretrained = copy.deepcopy(network).cpu()
    pretrained_cfg = network.pretrained_cfg
    mean = pretrained_cfg.get("mean", dataset.IMAGENET_MEAN)
    std = pretrained_cfg.get("std", dataset.IMAGENET_STD)
    interpolation = pretrained_cfg.get("interpolation", "bilinear")
    transform = dataset.build_transforms(False, 224, mean=mean, std=std,
                                         interpolation=interpolation)
    loader = dataset.make_loader(fixed, cfg.images_dir, transform, 18, False,
                                 num_workers=0, seed=42)
    x_cpu, y_cpu, names = next(iter(loader))
    x, y = x_cpu.to(device), y_cpu.to(device)
    assert x.shape == (18, 3, 224, 224) and y.dtype == torch.int64
    assert torch.isfinite(x).all() and y.min() == 0 and y.max() == 8
    assert list(names) == fixed.Filename.tolist()
    assert y_cpu.tolist() == fixed.Label.tolist()
    # Exercise the actual shuffled, augmented train loader as well.
    augmented = dataset.make_loader(frame, cfg.images_dir,
        dataset.build_transforms(True, 224, mean=mean, std=std, interpolation=interpolation),
        64, True, num_workers=0, seed=42)
    ax, ay, an = next(iter(augmented))
    label_map = frame.set_index("Filename").Label
    assert ax.shape == (64, 3, 224, 224) and torch.isfinite(ax).all()
    assert ay.tolist() == [int(label_map[n]) for n in an]
    report = {"scope": "sanity on TRAIN images only, not a model-quality experiment",
              "seed": 42, "backbone": "resnet50", "init": "ImageNet pretrained + new 9-class head",
              "environment": train.environment_info(device),
              "pretrained_cfg": pretrained_cfg,
              "batch_contract": {"passed": True, "fixed_shape": list(x.shape),
                                  "augmented_shape": list(ax.shape), "labels": y_cpu.tolist()},
              "test_evaluated": False, "val_evaluated": False}
    network.eval()
    with torch.inference_mode():
        logits = network(x).float()
        initial_loss = float(F.cross_entropy(logits, y))
    assert logits.shape == (18, 9) and torch.isfinite(logits).all()
    assert math.isfinite(initial_loss) and initial_loss > 0
    report["initial_ce"] = {"value": initial_loss, "uniform_reference_ln9": math.log(9),
                             "passed": True, "note": "Finite CE; closeness to ln(9) is not a hard assertion."}
    print(f"Initial CE: {initial_loss:.6f}; ln(9): {math.log(9):.6f}", flush=True)

    groups = model.param_groups(network, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    head_ids = {id(p) for p in model.classifier(network).parameters()}
    norm_ids = {id(p) for m in network.modules()
                if isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm,
                                  nn.modules.instancenorm._InstanceNorm))
                for p in m.parameters(recurse=False)}
    parameter_names = {id(p): n for n, p in network.named_parameters()}
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in network.parameters() if p.requires_grad}
    for g in groups:
        for p in g["params"]:
            assert g["lr"] == (cfg.lr_head if id(p) in head_ids else cfg.lr_backbone)
            if id(p) in norm_ids or parameter_names[id(p)].endswith(".bias") or p.ndim <= 1:
                assert g["weight_decay"] == 0
    report["param_groups"] = {"passed": True, "unique_tensors": len(ids), "groups": [
        {"name": g["group_name"], "lr": g["lr"], "weight_decay": g["weight_decay"],
         "tensors": len(g["params"])} for g in groups]}

    z = logits.detach().clone().requires_grad_(True)
    ce = F.cross_entropy(z, y)
    ce_grad = torch.autograd.grad(ce, z, retain_graph=True)[0]
    for criterion in (losses.FocalLoss(0), losses.LabelSmoothingCE(0)):
        value = criterion(z, y)
        torch.testing.assert_close(value, ce, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(torch.autograd.grad(value, z, retain_graph=True)[0],
                                   ce_grad, atol=1e-6, rtol=1e-6)
    report["loss_equivalence"] = {"passed": True, "focal_gamma0_and_ls_eps0": "CE values and gradients match"}
    permutation = torch.arange(len(x), device=device).roll(1)
    original = x.clone()
    with patch("losses.torch.randperm", return_value=permutation), \
         patch("losses.np.random.beta", return_value=0.36), \
         patch("losses.np.random.randint", return_value=223):
        mixed, targets = losses.mix_batch(x, y, mode="cutmix")
    # sqrt(1-.36)*224 -> cut width 179; corner (223,223) clips to [134:224].
    expected = x.clone()
    expected[:, :, 134:224, 134:224] = x[permutation, :, 134:224, 134:224]
    torch.testing.assert_close(mixed, expected)
    assert torch.equal(x, original) and torch.equal(targets[0], y)
    assert torch.equal(targets[1], y[permutation])
    assert targets[2] == 1 - 90 * 90 / (224 * 224)
    soft_targets = targets[2] * F.one_hot(y, 9) + (1 - targets[2]) * F.one_hot(y[permutation], 9)
    torch.testing.assert_close(losses.mixed_loss(nn.CrossEntropyLoss(), z, targets),
                               F.cross_entropy(z, soft_targets.float()))
    with patch("losses.torch.randperm", return_value=permutation), \
         patch("losses.np.random.beta", return_value=0.36):
        mixed_up, up_targets = losses.mix_batch(x, y, mode="mixup")
    torch.testing.assert_close(mixed_up, .36 * x + .64 * x[permutation])
    assert torch.equal(up_targets[1], y[permutation])
    report["mix_checks"] = {"passed": True, "mixup_lam": .36, "cutmix_actual_lam": targets[2],
                            "clipped_rectangle_xyxy": [134, 134, 224, 224]}

    # Frozen mode: compare all state tensors, not just one representative layer.
    frozen = pretrained.to(device)
    model.freeze_backbone(frozen)
    snapshot = {n: v.detach().cpu().clone() for n, v in frozen.state_dict().items()}
    frozen_head_names = {n for n, p in frozen.named_parameters() if p.requires_grad}
    frozen_optimizer = train.build_optimizer(frozen, cfg)
    frozen_optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(frozen(x), y).backward()
    assert all(p.grad is None for p in frozen.parameters() if not p.requires_grad)
    assert all(not m.training for m in frozen.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    frozen_optimizer.step()
    changed = [n for n, v in frozen.state_dict().items() if not torch.equal(v.cpu(), snapshot[n])]
    assert changed and set(changed) <= frozen_head_names
    report["frozen"] = {"passed": True, "changed_state_tensors": changed,
                        "backbone_weights_and_bn_buffers_unchanged": True}
    del frozen, pretrained, frozen_optimizer, snapshot

    # Evaluate the fixed TRAIN batch through the real evaluator, including a short last batch.
    eval_loader = dataset.make_loader(fixed, cfg.images_dir, transform, 7, False, num_workers=0, seed=42)
    before = state_digest(network)
    network.zero_grad(set_to_none=True)
    grad_enabled_in_forward = []
    handle = network.register_forward_pre_hook(lambda m, inputs: grad_enabled_in_forward.append(torch.is_grad_enabled()))
    try:
        filenames, labels, eval_logits, eval_loss = train.evaluate(network, eval_loader, nn.CrossEntropyLoss(), device)
    finally:
        handle.remove()
    assert before == state_digest(network)
    assert not any(grad_enabled_in_forward) and all(p.grad is None for p in network.parameters())
    assert filenames == fixed.Filename.tolist() and labels.tolist() == fixed.Label.tolist()
    probs = train.probabilities(eval_logits)
    np.testing.assert_allclose(probs.sum(1), 1, atol=1e-6)
    path = save_predictions(args.out / "diagnostic_train_seed42.csv", filenames, labels, probs)
    restored = read_pred(str(path))
    assert restored.filenames.tolist() == filenames
    np.testing.assert_array_equal(restored.y_true, labels)
    np.testing.assert_allclose(restored.probs, probs, atol=1e-7)
    report["evaluate_and_csv"] = {"passed": True, "images": len(filenames),
                                  "batch_sizes": [7, 7, 4], "state_unchanged": True,
                                  "grad_disabled": True, "filename_order_preserved": True,
                                  "csv_roundtrip": str(path.relative_to(train.ROOT))}
    train.write_json(args.out / "summary.json", report)

    # Fixed images, no stochastic augmentation, no mixing/smoothing. Constant LR isolates memorization.
    optimizer = train.build_optimizer(network, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = [{"step": 0, "train_ce": None, "eval_ce": initial_loss,
                "eval_top1": float((logits.argmax(1) == y).float().mean())}]
    skipped = 0
    gradient_check = None
    head_before = model.classifier(network).weight.detach().clone()
    backbone_before = network.conv1.weight.detach().clone()
    for step in range(1, args.max_steps + 1):
        stats = train.train_one_epoch(network, [(x_cpu, y_cpu, names)], nn.CrossEntropyLoss(),
                                      optimizer, scheduler, scaler, cfg, device)
        skipped += stats["skipped_updates"]
        if step == 1:
            gradients = [p.grad for p in network.parameters() if p.requires_grad]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert network.conv1.weight.grad.abs().sum() > 0
            assert model.classifier(network).weight.grad.abs().sum() > 0
            assert not torch.equal(head_before, model.classifier(network).weight)
            assert not torch.equal(backbone_before, network.conv1.weight)
            gradient_check = {"passed": True, "tensors": len(gradients),
                              "head_and_backbone_gradients_nonzero": True,
                              "head_and_backbone_weights_changed": True}
        row = {"step": step, "train_ce": stats["train_loss"], "eval_ce": None, "eval_top1": None}
        if step % 10 == 0:
            network.eval()
            with torch.inference_mode():
                output = network(x).float()
                row["eval_ce"] = float(F.cross_entropy(output, y))
                row["eval_top1"] = float((output.argmax(1) == y).float().mean())
            print(f"Overfit step {step}: train CE={row['train_ce']:.6f}, "
                  f"fixed-batch eval CE={row['eval_ce']:.6f}, top1={row['eval_top1']:.1%}", flush=True)
        history.append(row)
        if step >= 20 and row["eval_ce"] is not None and row["eval_ce"] < .05 and row["eval_top1"] == 1:
            break
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(args.out / "overfit_history.csv", index=False)
    measured = history_frame.dropna(subset=["eval_ce"]).iloc[-1]
    passed = bool(measured.eval_ce < .05 and measured.eval_top1 == 1 and skipped == 0)
    report["gradients_and_updates"] = gradient_check
    report["overfit"] = {"passed": passed, "images": 18, "images_per_class": 2,
                         "steps": step, "evaluation_step": int(measured.step), "initial_ce": initial_loss,
                         "final_train_ce": stats["train_loss"], "final_eval_ce": float(measured.eval_ce),
                         "final_top1": float(measured.eval_top1), "skipped_optimizer_updates": skipped,
                         "pass_rule": "fixed TRAIN batch eval CE < 0.05, top1=100%, no skipped update",
                         "settings": "224px, deterministic center crop, CE, no mix/smoothing, AdamW, constant baseline LR, AMP on CUDA"}
    report["seconds_total"] = time.perf_counter() - started
    report["all_passed"] = passed
    train.write_json(args.out / "summary.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history_frame.step, history_frame.train_ce, label="Train mode CE")
    sample = history_frame.dropna(subset=["eval_ce"])
    axes[0].plot(sample.step, sample.eval_ce, "o-", label="Eval mode CE on SAME train batch")
    axes[0].axhline(.05, ls="--", color="gray", label="Pass threshold 0.05")
    axes[0].set(xlabel="Optimizer steps", ylabel="Cross entropy")
    axes[1].plot(sample.step, sample.eval_top1, "o-")
    axes[1].set(xlabel="Optimizer steps", ylabel="Top-1 on SAME train batch", ylim=(0, 1.05))
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle("SANITY ONLY: ResNet-50 memorizing 18 TRAIN images (2 per class)")
    fig.tight_layout()
    figure_path = train.SUBMISSION / "curves/sanity/overfit_train_batch.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    lines = ["# Sanity checks — real train images", "",
             f"- Status: {'PASS' if passed else 'FAIL'}; ResNet-50 ImageNet pretrained, seed 42.",
             "- Batch contract, optimizer groups, Focal/LS equivalence, Mixup/CutMix: PASS.",
             "- Frozen: only head changes; all backbone weights and BN buffers stay identical.",
             "- Evaluate: no gradients/state changes; filename order and prediction CSV round-trip: PASS.",
             f"- Initial CE: {initial_loss:.6f}; ln(9): {math.log(9):.6f} (reference only).",
             f"- Overfit: {step} updates on 18 train images; eval-mode CE {measured.eval_ce:.6f}, top-1 {measured.eval_top1:.1%}.",
             f"- Skipped updates: {skipped}; final training CE {stats['train_loss']:.6f}.",
             "- No val/test images evaluated; these numbers do not measure generalization.",
             "- See summary.json, train_batch.csv, overfit_history.csv and curves/sanity/overfit_train_batch.png.",
             "- Next: one full pretrained train/val epoch to verify run outputs before B01–B05."]
    (args.out / "sanity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Report:", args.out / "sanity_report.md", flush=True)
    if not passed:
        raise SystemExit("Overfit sanity failed; inspect the saved diagnostics before full experiments.")


if __name__ == "__main__":
    main()

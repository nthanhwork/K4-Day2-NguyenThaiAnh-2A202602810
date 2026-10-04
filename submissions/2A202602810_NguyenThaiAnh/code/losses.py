"""Cross-entropy variants and batch-level Mixup/CutMix."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0,1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        self.gamma = gamma
        alpha = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)
        if alpha is not None and (alpha.ndim != 1 or not torch.isfinite(alpha).all() or (alpha < 0).any()):
            raise ValueError("alpha must be a finite nonnegative class vector")
        self.register_buffer("alpha", alpha)

    def forward(self, logits, target):
        log_pt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -(1.0 - log_pt.exp()).pow(self.gamma) * log_pt
        if self.alpha is not None:
            if len(self.alpha) != logits.shape[1]:
                raise ValueError("alpha length differs from number of classes")
            loss = loss * self.alpha[target]
        return loss.mean()


def build_criterion(kind: str = "ce", **kw):
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None:
            raise ValueError("ce_weighted requires train-derived weights")
        weight = torch.as_tensor(kw["weight"], dtype=torch.float32)
        if weight.shape != (9,) or not torch.isfinite(weight).all() or (weight <= 0).any():
            raise ValueError("weight must be a positive finite nine-class vector")
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"Unknown loss: {kind}")


def class_weights(counts, beta: float = 0.0):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.shape != (9,) or not torch.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("counts must contain nine positive finite train class counts")
    if not math.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("beta must be in [0,1)")
    if beta == 0:
        weights = counts.reciprocal()
    else:
        # -expm1(log(beta)*n) is stable for beta close to one.
        weights = (1 - beta) / (-torch.expm1(math.log(beta) * counts))
    return (weights / weights.mean()).float()


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    if mode not in {"mixup", "cutmix"} or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("mode must be mixup/cutmix and alpha must be positive")
    if x.ndim != 4 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        raise ValueError("Expected nonempty images NCHW and labels N")
    perm = torch.randperm(len(x), device=x.device)
    y_b = y[perm]
    lam = float(np.random.beta(alpha, alpha))
    if mode == "mixup":
        mixed = lam * x + (1 - lam) * x[perm]
    else:
        height, width = x.shape[-2:]
        ratio = math.sqrt(1 - lam)
        cut_w, cut_h = int(width * ratio), int(height * ratio)
        cx, cy = np.random.randint(width), np.random.randint(height)
        left, right = max(0, cx - cut_w // 2), min(width, cx + (cut_w + 1) // 2)
        top, bottom = max(0, cy - cut_h // 2), min(height, cy + (cut_h + 1) // 2)
        mixed = x.clone()
        mixed[:, :, top:bottom, left:right] = x[perm, :, top:bottom, left:right]
        lam = 1.0 - (right - left) * (bottom - top) / (height * width)
    return mixed, (y, y_b, lam)


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    if not 0 <= lam <= 1:
        raise ValueError("lam must be in [0,1]")
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

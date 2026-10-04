"""timm backbones, frozen feature extraction, optimizer groups and MAC profiling."""
from __future__ import annotations

import copy

import torch
from torch import nn
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50", "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny", "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224", "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def classifier(model):
    head = model.get_classifier()
    if isinstance(head, str):
        head = model.get_submodule(head)
    if not isinstance(head, nn.Module):
        raise ValueError("Expected a single module classifier from timm")
    return head


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune", *, cache_dir=None):
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError(f"Invalid init: {init}")
    if num_classes != 9:
        raise ValueError("This lab requires exactly nine classes")
    name = SUGGESTED_BACKBONES.get(name, name)
    if init == "frozen" and not pretrained:
        raise ValueError("Frozen initialization requires pretrained=True")
    model = timm.create_model(name, pretrained=pretrained and init != "scratch",
                              num_classes=num_classes, drop_rate=drop_rate,
                              cache_dir=str(cache_dir) if cache_dir is not None else None)
    model.lab_model_name = name
    model.lab_pretrained = pretrained and init != "scratch"
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model) -> None:
    head_ids = {id(p) for p in classifier(model).parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in head_ids)
    model._backbone_frozen = True
    set_train_mode(model)


def set_train_mode(model) -> None:
    """Keep the entire frozen feature extractor (including BN/dropout) in eval."""
    if getattr(model, "_backbone_frozen", False):
        model.eval()
        classifier(model).train()
    else:
        model.train()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Split backbone/head LR and exempt ALL norm/bias parameters from decay.

    Four groups when nonempty; the return contract remains list[dict].
    """
    if min(lr_backbone, lr_head) <= 0 or weight_decay < 0:
        raise ValueError("Invalid optimizer learning rates/weight decay")
    head_ids = {id(p) for p in classifier(model).parameters()}
    norm_types = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm,
                  nn.modules.instancenorm._InstanceNorm)
    norm_ids = {id(p) for module in model.modules() if isinstance(module, norm_types)
                for p in module.parameters(recurse=False)}
    exempt_names = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    groups = {}
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            raise ValueError(f"Duplicate parameter: {name}")
        seen.add(id(p))
        is_head = id(p) in head_ids
        no_decay = p.ndim <= 1 or name.endswith(".bias") or id(p) in norm_ids or name in exempt_names
        key = ("head" if is_head else "backbone", "no_decay" if no_decay else "decay")
        group = groups.setdefault(key, {
            "params": [], "lr": lr_head if is_head else lr_backbone,
            "weight_decay": 0.0 if no_decay else weight_decay,
            "group_name": "_".join(key),
        })
        group["params"].append(p)
    if not groups:
        raise ValueError("No trainable parameters")
    return list(groups.values())


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """fvcore MAC-style count; record unsupported operators instead of hiding them."""
    from fvcore.nn import FlopCountAnalysis
    if img_size < 1:
        raise ValueError("img_size must be positive")
    # Profile a copy so tracing cannot mutate BN buffers or model training modes.
    replica = copy.deepcopy(model).eval()
    # Expose QK/AV matmuls to fvcore rather than an unsupported fused SDPA op.
    for module in replica.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = False
    p = next(replica.parameters())
    x = torch.zeros(1, 3, img_size, img_size, device=p.device, dtype=p.dtype)
    with torch.inference_mode():
        analysis = FlopCountAnalysis(replica, x).unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        total = analysis.total()
    model.lab_mac_profile = {
        "tool": "fvcore.nn.FlopCountAnalysis (one multiply-add = one operation)",
        "unsupported_ops": dict(analysis.unsupported_ops()),
        "uncalled_modules": sorted(analysis.uncalled_modules()),
    }
    return float(total / 1e9)

"""Validation-selected inference, TTA, scalar calibration and safe BN fusion."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _logits(value):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 9 or not len(array) or not np.isfinite(array).all():
        raise ValueError('Expected finite nonempty N x 9 logits')
    return array


def apply_temperature(logits, T: float):
    if not math.isfinite(T) or T <= 0:
        raise ValueError('Temperature must be finite and positive')
    z = _logits(logits) / T
    z -= z.max(axis=1, keepdims=True)
    probs = np.exp(z)
    return probs / probs.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Minimize validation NLL over log(T) in [-6,6]; include T=1 as fallback."""
    logits = _logits(val_logits)
    labels = np.asarray(val_labels)
    if labels.shape != (len(logits),) or not np.isfinite(labels).all() or \
            not np.equal(labels, np.floor(labels)).all() or (labels < 0).any() or (labels > 8).any():
        raise ValueError('Expected N integer validation labels in 0..8')
    labels = labels.astype(np.int64)
    def objective(log_t):
        z = logits / math.exp(log_t)
        z -= z.max(axis=1, keepdims=True)
        return float((np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(z)), labels]).mean())
    left, right = -6., 6.
    ratio = (math.sqrt(5) - 1) / 2
    c, d = right - ratio * (right-left), left + ratio * (right-left)
    fc, fd = objective(c), objective(d)
    for _ in range(90):
        if fc <= fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right-left)
            fc = objective(c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right-left)
            fd = objective(d)
    choices = [0., -6., 6., (left+right)/2]
    return math.exp(min(choices, key=objective))


def view_identity(x):
    return x


def _images(x):
    if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != 3 or not x.shape[0]:
        raise ValueError('Expected nonempty NCHW RGB tensor')


def view_hflip(x):
    _images(x)
    return x.flip(-1)


def center_crop(x, crop):
    _images(x)
    h, w = x.shape[-2:]
    if type(crop) is not int or crop <= 0 or min(h,w) < crop:
        raise ValueError('Crop must be a positive integer within the image')
    top, left = round((h-crop)/2), round((w-crop)/2)
    return x[..., top:top+crop, left:left+crop]


def views_multicrop(x, crop: int):
    """Four corners plus center; source must be larger than crop in both axes."""
    center = center_crop(x, crop)
    h, w = x.shape[-2:]
    if min(h,w) <= crop:
        raise ValueError('Five distinct crop positions require source H,W > crop')
    return [x[..., :crop, :crop], x[..., :crop, w-crop:],
            x[..., h-crop:, :crop], x[..., h-crop:, w-crop:], center]


def views_multiscale(x, sizes):
    _images(x)
    sizes = list(sizes)
    if not sizes or any(type(s) is not int or s <= 0 for s in sizes):
        raise ValueError('Expected positive integer sizes; verify model compatibility separately')
    return [F.interpolate(x, size=(s,s), mode='bicubic', align_corners=False, antialias=True) for s in sizes]


def aggregate_views(logits_per_view, space: str = 'prob'):
    values = [_logits(value) for value in logits_per_view]
    if not values or any(v.shape != values[0].shape for v in values):
        raise ValueError('Views must have matching N x 9 shapes and filename/label order')
    if space == 'logit':
        return apply_temperature(np.mean(values, axis=0), 1.)
    if space != 'prob':
        raise ValueError('space must be prob or logit')
    return ensemble_probs([apply_temperature(v, 1.) for v in values])


def ensemble_probs(list_of_probs):
    """Array-level mean; file-level callers must verify filenames/labels first."""
    values = [_logits(value) for value in list_of_probs]
    if not values or any(v.shape != values[0].shape for v in values):
        raise ValueError('Probability arrays must have matching shapes')
    if any((v < 0).any() or (v > 1+1e-6).any() or not np.allclose(v.sum(1), 1., atol=1e-5) for v in values):
        raise ValueError('Expected normalized probabilities')
    probs = np.mean(values, axis=0)
    return probs / probs.sum(1, keepdims=True)


def ensemble_prediction_files(paths):
    from eval import read_pred
    predictions = [read_pred(str(p)) for p in paths]
    if not predictions:
        raise ValueError('No prediction files')
    reference = predictions[0]
    for pred in predictions[1:]:
        if not np.array_equal(reference.filenames, pred.filenames) or not np.array_equal(reference.y_true, pred.y_true):
            raise ValueError('Ensemble filename/label order mismatch')
    return reference.filenames, reference.y_true, ensemble_probs([p.probs for p in predictions])


def _forward(model, x, *, amp=False):
    if amp and x.device.type != 'cuda':
        raise ValueError('FP16 AMP requires CUDA for this lab protocol')
    with torch.amp.autocast(x.device.type, dtype=torch.float16 if x.device.type == 'cuda' else torch.bfloat16,
                            enabled=amp):
        logits = model(x)
    if logits.shape != (len(x),9) or not torch.isfinite(logits).all():
        raise ValueError('Model must return finite N x 9 logits')
    return logits.float()


@torch.inference_mode()
def predict_logits(model, loader, device, view=None, *, amp=False):
    model.eval()
    names, labels, values = [], [], []
    for x,y,filenames in loader:
        x = x.to(device, non_blocking=True)
        logits = _forward(model, x if view is None else view(x), amp=amp)
        names.extend(filenames)
        labels.append(torch.as_tensor(y).cpu().numpy())
        values.append(logits.cpu().numpy())
    if not names or len(set(names)) != len(names):
        raise ValueError('Empty loader or duplicate filenames')
    return names, np.concatenate(labels), np.concatenate(values)


@torch.inference_mode()
def method_logits(model, full_x, method, *, img_size=224, temperature=1.):
    """Actual pipeline: GPU crop/view forwards/aggregation; TTA final logits=log(p)."""
    model.eval()
    x = center_crop(full_x, img_size)
    if method in {'I00','I07','I08'}:
        logits = _forward(model, x, amp=method == 'I08')
        if method == 'I07':
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError('Temperature must be positive and finite')
            logits = logits / temperature
        return logits
    if method == 'I01':
        views = [x, view_hflip(x)]
    elif method == 'I02':
        views = views_multicrop(full_x, img_size)
    else:
        raise ValueError(f'Unknown inference method: {method}')
    probs = sum(_forward(model, view).softmax(1) for view in views) / len(views)
    return probs.clamp_min(1e-12).log()


@torch.inference_mode()
def predict_method(model, loader, device, method, *, img_size=224, temperature=1.):
    names, labels, values = [], [], []
    for x,y,filenames in loader:
        logits = method_logits(model, x.to(device, non_blocking=True), method,
                               img_size=img_size, temperature=temperature)
        names.extend(filenames)
        labels.append(torch.as_tensor(y).cpu().numpy())
        values.append(logits.cpu().numpy())
    if not names or len(set(names)) != len(names):
        raise ValueError('Empty loader or duplicate filenames')
    return names, np.concatenate(labels), np.concatenate(values)


def fuse_conv_bn(model):
    """Fuse proven adjacent pairs in plain nn.Sequential on an eval COPY only.

    Arbitrary attributes named conv/bn do not establish forward topology and are skipped.
    lab_fused_pairs records applicability; zero pairs is not a measured fusion method.
    """
    replica = copy.deepcopy(model).eval()
    pairs = []
    for name, module in replica.named_modules():
        if type(module) is not nn.Sequential:
            continue
        children = list(module._modules.items())
        for (conv_name, conv), (bn_name, bn) in zip(children, children[1:]):
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                module._modules[conv_name] = torch.nn.utils.fuse_conv_bn_eval(conv, bn)
                module._modules[bn_name] = nn.Identity()
                pairs.append(f'{name}.{conv_name}+{bn_name}')
    replica.lab_fused_pairs = pairs
    return replica

"""Synchronized latency measurements with declared pipeline scope and raw samples."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    if type(warmup) is not int or warmup < 10 or type(iters) is not int or iters < 50:
        raise ValueError('Protocol requires >=10 warmup and >=50 measured iterations')
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples.append((time.perf_counter()-start)*1000)
    return {'p50':float(np.percentile(samples,50)), 'p95':float(np.percentile(samples,95)),
            'p99':float(np.percentile(samples,99)), 'mean':float(np.mean(samples)),
            'n':iters, 'warmup':warmup, 'samples_ms':samples}


def device_metadata(device):
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    return {'device':str(device), 'gpu':torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'torch':str(torch.__version__), 'cuda':torch.version.cuda}


def pipeline_latency(fn, *, device, batch_size, img_size, dtype, k_views,
                     input_size=None, warmup=10, iters=100, scope):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    device = torch.device(device)
    sync = (lambda:torch.cuda.synchronize(device)) if device.type == 'cuda' else None
    with torch.inference_mode():
        result = bench(fn, warmup, iters, sync)
    result.update(device_metadata(device))
    result.update({'dtype':dtype,'batch':batch_size, 'img_size':img_size,
                   'input_size':input_size or img_size, 'k_views':k_views, 'bn_fused':False,
                   'scope':scope, 'images_per_s':batch_size/(result['p50']/1000)})
    return result


def latency_report(model, batch_size: int, img_size: int, dtype: str = 'fp32', device: str = 'cuda',
                   warmup: int = 10, iters: int = 100) -> dict:
    if dtype not in {'fp32','amp','fp16'} or type(img_size) is not int or img_size <= 0:
        raise ValueError('Invalid dtype/img_size')
    device = torch.device(device)
    if dtype != 'fp32' and device.type != 'cuda':
        raise ValueError('FP16/AMP measurements require CUDA in this protocol')
    replica = copy.deepcopy(model).to(device).eval()
    replica.half() if dtype == 'fp16' else replica.float()
    x = torch.randn(batch_size,3,img_size,img_size,device=device,
                    dtype=torch.float16 if dtype == 'fp16' else torch.float32)
    def fn():
        with torch.amp.autocast(device.type, enabled=dtype == 'amp'):
            return replica(x)
    return pipeline_latency(fn, device=device, batch_size=batch_size, img_size=img_size, dtype=dtype,
                            k_views=1, warmup=warmup, iters=iters,
                            scope='forward only; resident input; excludes preprocessing/H2D/softmax')


def tta_latency(model, k_views: int, **kw) -> dict:
    """Measure real center/hflip/5-crop forwards and probability aggregation."""
    import inference
    if k_views not in {1,2,5}:
        raise ValueError('Supported measured TTA policies: 1,2,5 views')
    img_size = kw.pop('img_size',224)
    batch = kw.pop('batch_size',1)
    device = torch.device(kw.pop('device','cuda'))
    dtype = kw.pop('dtype','fp32')
    if dtype != 'fp32':
        raise ValueError('This TTA helper measures FP32; AMP is a separate method')
    warmup, iters = kw.pop('warmup',10), kw.pop('iters',100)
    input_size = kw.pop('input_size',round(img_size*256/224))
    if kw:
        raise ValueError(f'Unsupported benchmark arguments: {sorted(kw)}')
    replica = copy.deepcopy(model).to(device).float().eval()
    x = torch.randn(batch,3,input_size,input_size,device=device)
    method = {1:'I00',2:'I01',5:'I02'}[k_views]
    return pipeline_latency(lambda:inference.method_logits(replica,x,method,img_size=img_size).softmax(1),
                            device=device,batch_size=batch,img_size=img_size,input_size=input_size,
                            dtype='fp32',k_views=k_views,warmup=warmup,iters=iters,
                            scope='resident normalized input -> crops/views -> model -> aggregation -> softmax; excludes CPU preprocessing/H2D')

"""Measure backbone latency for the report without training or reading test images."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch

import backbone_notebook as backbones
import benchmark
import inference
import inference_notebook as inference_runner
import model as model_utils
import train

OUTPUT = train.SUBMISSION / "code/validation/backbone_latency.json"
SCOPE = inference_runner.SCOPE


def measure(output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Measurement already saved: {output}; use another --output to measure again")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    train.set_seed(0)
    device = torch.device("cuda:0")
    rows = []
    before = {}
    for exp_id, (architecture, _) in backbones.BACKBONES.items():
        cfg = backbones.t00_config(exp_id, architecture)
        backbones.verify_outputs(cfg)
        checkpoint_path = train.run_dir(cfg) / "best.pt"
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        before[str(checkpoint_path.relative_to(train.ROOT))] = digest
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        network = model_utils.build_model(architecture, pretrained=False, num_classes=9)
        network.load_state_dict(checkpoint["model"], strict=True)
        network.to(device).float().eval()
        for batch in (1, 32):
            others = inference_runner.gpu_processes()
            if others:
                raise RuntimeError(f"Other GPU compute jobs: {others}")
            x = torch.randn(batch, 3, 256, 256, device=device)
            result = benchmark.pipeline_latency(
                lambda: inference.method_logits(network, x, "I00", img_size=224).softmax(1),
                device=device, batch_size=batch, img_size=224, input_size=256,
                dtype="fp32", k_views=1, warmup=10, iters=100, scope=SCOPE,
            )
            rows.append({"exp_id": exp_id, "seed": 0, "backbone": architecture,
                         "checkpoint_sha256": digest, "input": "synthetic resident normalized tensor",
                         **result})
            print(f"{exp_id} batch={batch}: p50={result['p50']:.4f}, p95={result['p95']:.4f} ms", flush=True)
            del x
        del network, checkpoint
        torch.cuda.empty_cache()
    if inference_runner.gpu_processes():
        raise RuntimeError("Another compute job appeared during measurement")
    for name, digest in before.items():
        if hashlib.sha256((train.ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checkpoint changed: {name}")
    train.write_json(output, {"schema": 1, "seed": 0, "pid": os.getpid(),
                              "test_evaluated": False, "training_performed": False,
                              "checkpoint_hashes": before, "measurements": rows})
    pd.DataFrame([{k: v for k, v in row.items() if k != "samples_ms"} for row in rows]).to_csv(
        output.with_suffix(".csv"), index=False)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(measure(args.output))

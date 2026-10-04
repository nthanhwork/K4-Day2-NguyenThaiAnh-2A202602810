"""Run and summarize the ConvNeXt-Tiny training ablation matrix.

The completed B03 run is the T00 baseline when its configuration matches the
shared T00 recipe. All ablations use validation only; test predictions remain
disabled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

import backbone_notebook
import train
from eval import CLASS_NAMES, compute_metrics, read_pred


BASELINE_EXP_ID = "B03"
ABLATION_OUTPUT = train.ROOT / "runs" / "ablation_convnext_tiny"


@dataclass(frozen=True)
class AblationSpec:
    exp_id: str
    axis: str
    description: str
    changes: dict[str, Any]


def default_plan() -> list[AblationSpec]:
    """Return the controlled T00/T01-T07 plan from the lab instructions."""
    return [
        AblationSpec("T00", "baseline", "B03 alias: finetune + basic + CE", {}),
        AblationSpec("T01", "init", "scratch initialization", {"init": "scratch"}),
        AblationSpec("T02", "init", "frozen pretrained backbone", {"init": "frozen"}),
        AblationSpec("T03", "augmentation", "color augmentation", {"aug": "color"}),
        AblationSpec("T04", "augmentation", "CutMix alpha=1.0",
                     {"mix": "cutmix", "mix_alpha": 1.0}),
        AblationSpec("T05", "loss", "label smoothing=0.1",
                     {"loss": "ls", "label_smoothing": 0.1}),
        AblationSpec("T06", "loss", "focal loss gamma=2.0",
                     {"loss": "focal", "focal_gamma": 2.0}),
        AblationSpec("T07", "loss", "weighted CE from train counts",
                     {"loss": "ce_weighted", "class_weight_beta": 0.0}),
    ]


def _baseline_config() -> train.Config:
    config_path = train.run_dir(train.Config(exp_id=BASELINE_EXP_ID)) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing completed baseline config: {config_path}. "
            "Finish B03 before running the ablation matrix."
        )
    config = train.Config(**json.loads(config_path.read_text()))
    if config.backbone != "convnext_tiny":
        raise ValueError(f"{config_path}: expected convnext_tiny, got {config.backbone!r}")
    if config.save_test_predictions:
        raise ValueError("The ablation baseline must not contain test predictions")
    status = backbone_notebook.training_status(config)
    if status != "complete":
        raise ValueError(f"B03 baseline status is {status!r}; complete it before ablation")
    backbone_notebook.verify_outputs(config)
    expected = backbone_notebook.t00_config(BASELINE_EXP_ID, "convnext_tiny")
    actual_dict, expected_dict = asdict(config), asdict(expected)
    for key in ("resume", "device", "num_workers"):
        actual_dict.pop(key)
        expected_dict.pop(key)
    if actual_dict != expected_dict:
        raise ValueError("B03 does not match the declared T00 recipe")
    _check_current_splits(config)
    return replace(config, resume=False, save_test_predictions=False)


def _check_current_splits(config):
    recorded = json.loads((train.run_dir(config) / "split_hashes.json").read_text())
    current = {name: hashlib.sha256((Path(config.labels_dir) / name).read_bytes()).hexdigest()
               for name in recorded}
    if recorded != current:
        raise ValueError("CSV hashes changed since this run")
    return recorded


def active_plan(output_dir=None):
    """Include a previously locked T08 so later collection cannot drop its row."""
    plan = default_plan()
    output_dir = ABLATION_OUTPUT if output_dir is None else output_dir
    path = Path(output_dir) / "t08_selection.json"
    if path.is_file():
        decision = json.loads(path.read_text())
        plan.append(make_combination(decision["changes"], description=decision["description"]))
    return plan


def configs_for(plan: list[AblationSpec] | None = None) -> dict[str, train.Config]:
    """Build configs while preserving every B03 setting not under ablation."""
    base = _baseline_config()
    configs = {}
    selected = default_plan() if plan is None else plan
    if sum(spec.exp_id == "T00" for spec in selected) != 1:
        raise ValueError("Plan must contain exactly one T00 baseline")
    if len({spec.exp_id for spec in selected}) != len(selected):
        raise ValueError("Duplicate ablation IDs")
    canonical = {spec.exp_id: spec for spec in default_plan()}
    for spec in selected:
        if spec.exp_id == "T08":
            make_combination(spec.changes)
        elif spec.exp_id not in canonical or spec.changes != canonical[spec.exp_id].changes:
            raise ValueError(f"{spec.exp_id}: use the declared controlled ablation changes")
        if spec.exp_id == "T00":
            configs[spec.exp_id] = base
            continue
        configs[spec.exp_id] = replace(base, exp_id=spec.exp_id, **spec.changes)
        train.validate_config(configs[spec.exp_id])
    return configs


def verify_ablation(config, base):
    """Audit complete outputs, identical folds/preprocessing and evaluator metrics."""
    summary, history, metadata = backbone_notebook.verify_outputs(config)
    if _check_current_splits(config) != _check_current_splits(base):
        raise ValueError("Ablation split differs from B03")
    baseline_metadata = json.loads((train.run_dir(base) / "model.json").read_text())
    for key in ("img_size", "resize_size", "mean", "std", "interpolation", "validation"):
        if metadata["preprocessing"][key] != baseline_metadata["preprocessing"][key]:
            raise ValueError(f"Ablation preprocessing differs from B03: {key}")
    for key in ("architecture", "tag"):
        if metadata["pretrained_cfg"].get(key) != baseline_metadata["pretrained_cfg"].get(key):
            raise ValueError(f"Ablation model tag differs from B03: {key}")
    if metadata["preprocessing"]["aug"] != config.aug:
        raise ValueError("Recorded augmentation disagrees with config")
    if metadata["pretrained_loaded"] != (config.init != "scratch"):
        raise ValueError("Pretrained flag disagrees with initialization")
    prediction = read_pred(str(train.pred_path(config, "val")))
    metrics = compute_metrics(prediction.y_true, prediction.y_pred, prediction.probs)
    for key in ("macro_f1", "top1", "balanced_acc", "ece", "nll"):
        if not math.isclose(metrics[key], summary["val"][key], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"Summary metric differs from validation CSV: {key}")
    if not math.isclose(metrics["macro_f1"], history["val_macro_f1"].max(), abs_tol=1e-7):
        raise ValueError("Exported best checkpoint differs from history")
    return summary, history, metrics


def _summary_row(spec: AblationSpec, config: train.Config, summary: dict) -> dict:
    val = summary["val"]
    return {
        "exp_id": spec.exp_id,
        "baseline_id": BASELINE_EXP_ID,
        "backbone": config.backbone,
        "seed": config.seed,
        "axis": spec.axis,
        "description": spec.description,
        "changes_json": json.dumps(spec.changes, sort_keys=True),
        "config_json": json.dumps(asdict(config), sort_keys=True),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "img_size": config.img_size,
        "pretrained_tag": json.loads((train.run_dir(config) / "model.json").read_text())["pretrained_cfg"].get("tag"),
        "macro_f1_val": float(val["macro_f1"]),
        "top1_val": float(val["top1"]),
        "balanced_acc_val": float(val["balanced_acc"]),
        "ece_val": float(val["ece"]),
        "nll_val": float(val["nll"]),
        "delta_macro_f1": None,
        "delta_macro_f1_pp": None,
        "train_seconds_per_epoch_observed": summary["train_seconds_per_epoch"],
        "timing_scope": "observed training, not an isolated latency benchmark",
        "best_epoch": int(summary["best_epoch"]),
        "status": "complete",
        "run_path": str(train.run_dir(config)),
    }


def collect_results(
    plan: list[AblationSpec] | None = None,
    *,
    output_dir: str | Path | None = None,
    require_complete: bool = True,
    write_outputs: bool = True,
) -> pd.DataFrame:
    """Read summaries and write a machine-readable validation comparison."""
    output_dir = ABLATION_OUTPUT if output_dir is None else output_dir
    selected_plan = active_plan(output_dir) if plan is None else plan
    configs = configs_for(selected_plan)
    rows = []
    class_rows = []
    base = configs["T00"]
    for spec in selected_plan:
        config = configs[spec.exp_id]
        status = backbone_notebook.training_status(config)
        if status != "complete":
            if require_complete:
                raise ValueError(f"{spec.exp_id}: incomplete run ({status})")
            rows.append({"exp_id": spec.exp_id, "baseline_id": BASELINE_EXP_ID,
                         "backbone": config.backbone, "seed": config.seed,
                         "axis": spec.axis, "description": spec.description,
                         "changes_json": json.dumps(spec.changes, sort_keys=True),
                         "config_json": json.dumps(asdict(config), sort_keys=True),
                         "status": status, "run_path": str(train.run_dir(config))})
            continue
        summary, history, metrics = verify_ablation(config, base)
        row = _summary_row(spec, config, summary)
        row["total_skipped_updates"] = int(history["skipped_updates"].sum())
        rows.append(row)
        for label, name in enumerate(CLASS_NAMES):
            class_rows.append({"exp_id": spec.exp_id, "baseline_id": BASELINE_EXP_ID,
                               "seed": config.seed, "class_id": label, "class_name": name,
                               **{key: float(metrics[key][label]) for key in ("precision", "recall", "f1")},
                               "support": int(metrics["support"][label])})
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame["exp_id"] == "T00", "macro_f1_val"]
    if len(baseline) != 1:
        raise ValueError("Plan must contain exactly one T00 baseline")
    frame["delta_macro_f1"] = frame["macro_f1_val"] - float(baseline.iloc[0])
    frame["delta_macro_f1_pp"] = frame["delta_macro_f1"] * 100
    if not write_outputs:
        return frame
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "training_ablation.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output / "training_per_class_val.csv", index=False)
    (output / "ablation_plan.json").write_text(json.dumps(
        [asdict(spec) for spec in selected_plan], indent=2, ensure_ascii=False
    ) + "\n")
    train.write_json(output / "baseline_alias.json", {
        "alias": "T00", "source_exp_id": BASELINE_EXP_ID, "seed": base.seed,
        "source_run": str(train.run_dir(base)), "independent_run": False,
        "config": asdict(base), "split_hashes": _check_current_splits(base),
        "selection_basis": "highest macro-F1 among B01-B05 on validation seed 0",
    })
    complete = frame[frame["status"] == "complete"]
    winner = complete.sort_values(["macro_f1_val", "exp_id"], ascending=[False, True]).iloc[0]
    lines = ["# ConvNeXt-Tiny — ablation validation", "",
             "T00 là alias của B03/seed0, không phải một lần train độc lập.",
             "Một seed dùng để sàng lọc; chưa có độ lệch chuẩn hoặc kết luận về ý nghĩa thống kê.", "",
             "| ID | Trục | Trạng thái | Macro-F1 val (%) | Δ so với T00 (điểm %) |",
             "|---|---|---|---:|---:|"]
    for row in frame.to_dict("records"):
        measured = row["status"] == "complete"
        score = f'{row["macro_f1_val"] * 100:.4f}' if measured else "—"
        delta = f'{row["delta_macro_f1_pp"]:+.4f}' if measured else "—"
        lines.append(f'| {row["exp_id"]} | {row["axis"]} | {row["status"]} | {score} | {delta} |')
    lines += ["", f'Mốc tốt nhất đã hoàn tất: {winner["exp_id"]}.',
              "Giữ các kết quả không cải thiện; không mặc định T08 sẽ tốt hơn T00.",
              "Loss train/val của các criterion khác nhau không dùng để so sánh trực tiếp; dùng F1/NLL.",
              "Timing train có thể chia sẻ GPU; đo latency độc lập ở phần 9."]
    decision_path = output / "t08_selection.json"
    if decision_path.is_file() and "T08" in set(frame["exp_id"]):
        decision = json.loads(decision_path.read_text())
        lines += ["", f'T08: {decision["mode"]}; {decision["description"]}.',
                  f'Các thành phần: {", ".join(decision["source_ids"])}.']
        combo = frame[frame["exp_id"] == "T08"].iloc[0]
        parts = complete[complete["exp_id"].isin(decision["source_ids"])]
        if combo["status"] == "complete" and len(parts):
            gain = (combo["macro_f1_val"] - parts["macro_f1_val"].max()) * 100
            lines.append(f'T08 so với thành phần tốt nhất: {gain:+.4f} điểm phần trăm macro-F1.')
    (output / "ablation_report.md").write_text("\n".join(lines) + "\n")
    train.write_json(output / "screening_selection.json", {
        "candidate_id": winner["exp_id"], "source_run": winner["run_path"],
        "macro_f1_val": winner["macro_f1_val"], "seed": base.seed,
        "selection_split": "val", "single_seed_only": True,
        "final_configuration_locked": False,
        "section8_complete": {f"T{i:02}" for i in range(9)} <= set(complete["exp_id"]),
    })
    return frame


def run_ablation(
    plan: list[AblationSpec] | None = None,
    *,
    planned_group: int = 4,
    run_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Run T01-T07 through the existing isolated/resumable notebook runner."""
    selected_plan = active_plan() if plan is None else plan
    configs = configs_for(selected_plan)
    selected_ids = set(run_ids) if run_ids is not None else {f"T{i:02}" for i in range(1, 8)}
    if selected_ids - (set(configs) - {"T00"}):
        raise ValueError("Unknown run IDs or attempt to retrain the T00 alias")
    for spec in selected_plan:
        if spec.exp_id not in selected_ids:
            continue
        print(f"Running {spec.exp_id}: {spec.description}", flush=True)
        backbone_notebook.run_in_subprocess(configs[spec.exp_id], planned_group)
        verify_ablation(configs[spec.exp_id], configs["T00"])
    return collect_results(selected_plan, require_complete=False)


def make_combination(
    changes: dict[str, Any],
    *,
    description: str = "selected combination",
) -> AblationSpec:
    """Validate a T08 recipe combining at least two effective ablation factors."""
    allowed = {"init", "aug", "mix", "mix_alpha", "loss", "label_smoothing",
               "focal_gamma", "class_weight_beta"}
    unknown = set(changes) - allowed
    if unknown:
        raise ValueError(f"T08 contains unsupported changes: {sorted(unknown)}")
    factors = []
    for key, default in (("init", "finetune"), ("aug", "basic"), ("mix", None), ("loss", "ce")):
        if key in changes and changes[key] != default:
            factors.append(key)
    if len(factors) < 2:
        raise ValueError("T08 must combine at least two effective factors; dependent parameters do not count")
    dependencies = {"mix_alpha": ("mix", {"cutmix", "mixup"}),
                    "label_smoothing": ("loss", {"ls"}),
                    "focal_gamma": ("loss", {"focal"}),
                    "class_weight_beta": ("loss", {"ce_weighted"})}
    for key, (parent, values) in dependencies.items():
        if key in changes and changes.get(parent) not in values:
            raise ValueError(f"{key} requires an active {parent} in {sorted(values)}")
    train.validate_config(replace(backbone_notebook.t00_config("T08", "convnext_tiny"), **changes))
    return AblationSpec("T08", "combination", description, dict(changes))


def propose_combination(frame, *, min_delta=0.002):
    """Select the best two distinct factors; explicitly label exploratory choices."""
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and nonnegative")
    expected = {f"T{i:02}" for i in range(8)}
    complete = frame[frame["status"] == "complete"]
    if not expected <= set(complete["exp_id"]):
        raise ValueError("Complete T00-T07 before selecting T08")
    candidates = complete[complete["exp_id"].isin(expected - {"T00"})].copy()
    # Frozen/scratch are mutually exclusive; likewise LS/focal/weighted CE.
    factors = {"T01": "init", "T02": "init", "T03": "aug", "T04": "mix",
               "T05": "loss", "T06": "loss", "T07": "loss"}
    candidates["factor"] = candidates["exp_id"].map(factors)
    chosen = candidates.sort_values(["delta_macro_f1", "exp_id"], ascending=[False, True]) \
        .drop_duplicates("factor").head(2)
    changes = {}
    specs = {spec.exp_id: spec for spec in default_plan()}
    for exp_id in chosen["exp_id"]:
        changes.update(specs[exp_id].changes)
    improved = bool((chosen["delta_macro_f1"] > min_delta).all())
    description = ("combine validation improvements" if improved else
                   "exploratory combination: fewer than two factors exceed the practical threshold")
    spec = make_combination(changes, description=description)
    return {"exp_id": spec.exp_id, "changes": spec.changes, "description": description,
            "source_ids": chosen["exp_id"].tolist(), "mode": "validation_improvements" if improved else "exploratory",
            "min_delta": min_delta, "threshold_note": "practical screening threshold, not statistical significance",
            "source_deltas": dict(zip(chosen["exp_id"], chosen["delta_macro_f1"]))}


def lock_combination(decision, *, output_dir=None):
    """Save selection before train; retain it unchanged on resume/completed reuse."""
    output_dir = ABLATION_OUTPUT if output_dir is None else output_dir
    spec = make_combination(decision["changes"], description=decision["description"])
    configs = configs_for()
    collect_results(default_plan(), output_dir=output_dir, write_outputs=False)
    sources = {}
    for exp_id, config in configs.items():
        paths = [train.run_dir(config) / "config.json", train.run_dir(config) / "summary.json",
                 train.pred_path(config, "val")]
        sources[exp_id] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    payload = {**decision, "changes": spec.changes, "source_hashes": sources,
               "baseline_id": BASELINE_EXP_ID, "seed": configs["T00"].seed,
               "selection_split": "val", "test_evaluated": False}
    path = Path(output_dir) / "t08_selection.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["changes"] != payload["changes"] or saved["source_hashes"] != sources:
            raise ValueError("T08 selection already locked; retain its recipe and source results")
        return saved
    train.write_json(path, payload)
    return payload


def plot_results(frame, *, output_dir=None):
    """Plot measured scores/deltas only, and save a publication-ready PNG."""
    import matplotlib.pyplot as plt
    completed = frame[frame["status"] == "complete"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(completed["exp_id"], completed["macro_f1_val"] * 100)
    axes[0].set(ylabel="Validation macro-F1 (%)", ylim=(0, 100))
    axes[1].bar(completed["exp_id"], completed["delta_macro_f1_pp"])
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set(ylabel="Delta macro-F1 vs T00 (percentage points)")
    fig.suptitle("ConvNeXt-Tiny | fold 0 | seed 0 | T00 = B03")
    fig.tight_layout()
    output = Path(output_dir) if output_dir is not None else Path(_baseline_config().curves_dir) / "ablation_convnext_tiny"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "training_ablation_val.png", dpi=180, bbox_inches="tight")
    return fig


def _parse_changes(pairs: list[str]) -> dict[str, Any]:
    base = _baseline_config()
    parsed = train.parse_overrides(pairs)
    # Restrict the CLI to fields that are meaningful ablation axes.
    spec = make_combination(parsed)
    replace(base, exp_id=spec.exp_id, **spec.changes)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="run T01-T07, resuming existing runs")
    parser.add_argument("--ids", nargs="+", help="subset of T01-T07 to run sequentially")
    parser.add_argument("--auto-t08", action="store_true", help="select and run T08 after T01-T07")
    parser.add_argument("--min-delta", type=float, default=.002,
                        help="practical screening threshold (0.002 = 0.2 percentage points)")
    parser.add_argument("--t08", nargs="*", default=None, metavar="KEY=VALUE",
                        help="run T08 with at least two validated changes")
    args = parser.parse_args()
    if args.t08 is not None and args.auto_t08:
        parser.error("Choose --t08 or --auto-t08")
    if args.ids and not args.run:
        parser.error("--ids requires --run")
    try:
        if args.run:
            frame = run_ablation(run_ids=args.ids)
        else:
            frame = collect_results(require_complete=False)
        if args.t08 is not None or args.auto_t08:
            if args.auto_t08:
                path = ABLATION_OUTPUT / "t08_selection.json"
                decision = json.loads(path.read_text()) if path.exists() else propose_combination(frame, min_delta=args.min_delta)
            else:
                changes = _parse_changes(args.t08)
                decision = {"changes": changes, "description": "manual validation-based exploratory combination",
                            "source_ids": [s.exp_id for s in default_plan()[1:]
                                           if all(changes.get(k) == v for k, v in s.changes.items())],
                            "mode": "manual_exploratory"}
            decision = lock_combination(decision)
            plan = active_plan()
            configs = configs_for(plan)
            backbone_notebook.run_in_subprocess(configs["T08"], planned_group=4)
            frame = collect_results(plan)
        plot_results(frame)
        print(frame.to_string(index=False))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

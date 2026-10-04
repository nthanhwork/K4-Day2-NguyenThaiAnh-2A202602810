"""Offline audit of the six deliverables, workbook values and prediction provenance."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re

import nbformat
import numpy as np
from openpyxl import load_workbook
import pandas as pd

import export_submission as exporter


def check():
    sub = exporter.SUB
    exporter.verify_snapshot()
    for name in ("results.xlsx", "report.md", "report.pdf", "README.md", "code", "curves", "predictions"):
        if not (sub / name).exists():
            raise ValueError(f"Missing deliverable: {name}")
    frames = exporter.tables()
    wb = load_workbook(sub / "results.xlsx", data_only=False)
    if set(wb.sheetnames) != set(frames):
        raise ValueError("Workbook must contain exactly the seven required sheets")
    for name, frame in frames.items():
        ws = wb[name]
        if ws.max_row != len(frame) + 1 or ws.max_column != len(frame.columns):
            raise ValueError(f"Workbook shape differs from source table: {name}")
        if ws.freeze_panes != "A2" or not ws.auto_filter.ref:
            raise ValueError(f"Workbook formatting incomplete: {name}")
        if [c.value for c in ws[1]] != list(frame.columns):
            raise ValueError(f"Workbook columns differ: {name}")
        for i, row in enumerate(frame.itertuples(index=False, name=None), start=2):
            for j, expected in enumerate(row, start=1):
                actual = ws.cell(i, j).value
                if ws.cell(i, j).data_type == "f":
                    raise ValueError(f"Unexpected formula: {name}!{ws.cell(i, j).coordinate}")
                if pd.isna(expected):
                    if actual is not None:
                        raise ValueError(f"Missing value changed: {name} row{i},col{j}")
                elif isinstance(expected, (float, np.floating)):
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                elif actual != expected:
                    raise ValueError(f"Workbook value differs: {name} row{i},col{j}")
    count = 0
    for path in sorted((sub / "predictions").rglob("*.csv")):
        split = "test" if path.name.endswith("_test.csv") else "val" if path.name.endswith("_val.csv") else None
        if split is None:
            raise ValueError(f"Unexpected prediction file: {path}")
        pred, _ = exporter.prediction(str(path.relative_to(sub / "predictions")), split)
        if len(pred.y_true) != (3507 if split == "test" else 3501):
            raise ValueError(f"Incomplete split: {path}")
        count += 1
    notebooks = list((sub / "code").rglob("*.ipynb"))
    for path in notebooks:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                if cell.outputs or cell.execution_count is not None:
                    raise ValueError(f"Embedded execution output remains: {path}")
                # Legacy setup notebook contains IPython magics; validate regular Python notebooks.
                if not any(line.lstrip().startswith(("%", "!")) for line in cell.source.splitlines()):
                    ast.parse(cell.source)
    links = 0
    for document in (sub / "README.md", sub / "report.md", sub / "code/notebooks/README.md"):
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            if target.startswith(("https://", "http://", "#")):
                continue
            if not (document.parent / target.split("#")[0]).exists():
                raise ValueError(f"Broken local link in {document.name}: {target}")
            links += 1
    for sheet in ("Backbones", "Training", "Final"):
        for curve in frames[sheet].curve:
            if curve.startswith("curves/") and not (sub / curve).exists():
                raise ValueError(f"Missing experiment curve: {curve}")
    calibrated = frames["Final"]
    aggregate = calibrated[(calibrated.exp_id == "F01") & (calibrated.seed == "mean")].iloc[0]
    result = {"status": "PASS", "sheets": {name: len(frame) for name, frame in frames.items()},
              "prediction_files_checked": count, "notebooks_checked": len(notebooks), "local_links_checked": links,
              "n_seeds": 3, "test_images_per_seed": 3507, "sample_std_ddof": 1,
              "final_test_macro_f1_mean": aggregate.macro_f1_test,
              "final_test_macro_f1_std": aggregate.macro_f1_test_std,
              "training_performed": False, "test_forward_performed": False,
              "colab_status": "GitHub-backed link requires the commit to be pushed to main",
              "results_sha256": exporter.sha(sub / "results.xlsx"), "report_sha256": exporter.sha(sub / "report.md"),
              "pdf_sha256": exporter.sha(sub / "report.pdf")}
    exporter.write_json(sub / "code/validation/submission_check.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    check()

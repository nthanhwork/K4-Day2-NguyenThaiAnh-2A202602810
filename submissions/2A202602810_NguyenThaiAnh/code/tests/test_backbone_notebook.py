"""Notebook orchestration tests use synthetic images, never DeepWeeds training."""
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backbone_notebook as notebooks
import dataset
import train
from test_implementation import TinyModel, create_fixture


class TestBackboneNotebooks(unittest.TestCase):
    def test_all_models_share_t00_and_two_two_one_groups(self):
        configs = [notebooks.t00_config(exp_id, name) for exp_id, (name, group) in notebooks.BACKBONES.items()]
        common = []
        for cfg in configs:
            value = asdict(cfg)
            value.pop("exp_id")
            value.pop("backbone")
            common.append(value)
        self.assertTrue(all(c == common[0] for c in common))
        self.assertEqual(configs[0].epochs, 12)
        self.assertEqual(configs[0].batch_size, 64)
        self.assertEqual(configs[0].warmup_epochs, 1)
        self.assertTrue(all(not c.save_test_predictions for c in configs))
        self.assertEqual([sum(group == i for _, group in notebooks.BACKBONES.values()) for i in (1, 2, 3)], [2, 2, 1])

    def test_lock_rejects_duplicate_job_and_test_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = train.Config(exp_id="lock_fixture", out_dir=tmp)
            path = train.run_dir(cfg)
            path.mkdir(parents=True)
            with (path / "notebook_train.lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "already training"):
                    notebooks.execute_training(cfg, 1)
            self.assertFalse((path / "config.json").exists())
            with self.assertRaisesRegex(ValueError, "test predictions disabled"):
                notebooks.training_status(replace(cfg, save_test_predictions=True))

    def test_resume_completed_reuse_and_output_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, labels = create_fixture(root)
            cfg = train.Config(exp_id="notebook_fixture", init="scratch", device="cpu", amp=False,
                epochs=2, warmup_epochs=0, batch_size=9, img_size=16, num_workers=0,
                images_dir=str(images), labels_dir=str(labels), out_dir=str(root / "runs"),
                pred_dir=str(root / "predictions"), curves_dir=str(root / "curves"))
            check_split = dataset.check_split
            with patch.object(train.dataset, "check_split", side_effect=lambda *a, **k: check_split(*a, expected_total=45, **k)), \
                 patch.object(train.model_utils, "build_model", side_effect=lambda *a, **k: TinyModel()), \
                 patch.object(train, "_verify_provenance", return_value=None):
                self.assertEqual(notebooks.training_status(cfg), "new")
                train.run(cfg, stop_after_epoch=1)
                self.assertEqual(notebooks.training_status(cfg), "resume")
                notebooks.execute_training(cfg, 1)
            self.assertEqual(notebooks.training_status(cfg), "complete")
            summary, history, metadata = notebooks.verify_outputs(cfg)
            self.assertEqual(len(history), 2)
            self.assertEqual(summary["val"]["n"], 9)
            invocation = json.loads((train.run_dir(cfg) / "notebook_invocations.jsonl").read_text())
            self.assertEqual(invocation["mode"], "resume")
            with patch.object(train, "run", side_effect=AssertionError("A completed run must not retrain")):
                result = notebooks.execute_training(cfg, 1)
            self.assertEqual(result, summary)
            # Exercise the real subprocess path using a completed CPU fixture: no GPU/train.
            with patch.object(notebooks, "ROOT", root), patch.object(train, "ROOT", root):
                from_child = notebooks.run_in_subprocess(cfg, 1)
            self.assertEqual(from_child, summary)
            with self.assertRaisesRegex(ValueError, "config differs"):
                notebooks.training_status(replace(cfg, batch_size=8))
            prediction_path = train.pred_path(cfg, "val")
            import pandas as pd
            corrupted = pd.read_csv(prediction_path)
            corrupted.loc[0, "Filename"] = "incorrect.jpg"
            corrupted.to_csv(prediction_path, index=False)
            with self.assertRaisesRegex(ValueError, "filenames differ"):
                notebooks.verify_outputs(cfg)
            self.assertFalse(train.pred_path(cfg, "test").exists())


if __name__ == "__main__":
    unittest.main()

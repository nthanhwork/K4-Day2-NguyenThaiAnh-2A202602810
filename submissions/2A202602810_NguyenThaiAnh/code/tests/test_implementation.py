"""CPU checks for the submission; run separately from the original starter tests."""
from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import sys

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

import dataset
import losses
import model
import train
from eval import read_pred

torch.set_num_threads(1)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4),
                                      nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(4, 9)
        self.pretrained_cfg = {}
        self.lab_model_name = "test_fixture"
        self.lab_pretrained = False

    def get_classifier(self):
        return self.head

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def create_fixture(path):
    """45 synthetic images, disjoint 27/9/9 CSVs; NOT lab results."""
    images, labels = path / "images", path / "labels"
    images.mkdir()
    labels.mkdir()
    frames = []
    index = 0
    for split, copies in (("train", 3), ("val", 1), ("test", 1)):
        rows = []
        for label in range(9):
            for _ in range(copies):
                name = f"fixture_{index}.jpg"
                color = (label * 25, 128, 255 - label * 25)
                Image.new("RGB", (16, 16), color).save(images / name)
                rows.append({"Filename": name, "Label": label, "Species": dataset.CLASS_NAMES[label]})
                index += 1
        df = pd.DataFrame(rows)
        # Match original author split headers exactly.
        df[["Filename", "Label"]].to_csv(labels / f"{split}_subset0.csv", index=False)
        frames.append(df)
    pd.concat(frames).to_csv(labels / "labels.csv", index=False)
    return images, labels


class TestData(unittest.TestCase):
    def test_original_two_column_headers_enriched_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            images, labels = create_fixture(path)
            original = (labels / "train_subset0.csv").read_bytes()
            frames = dataset.load_split(labels)
            report = dataset.check_split(*frames, images, expected_total=45,
                                         reference_df=pd.read_csv(labels / "labels.csv"),
                                         verify_images=True)
            self.assertEqual(report["n"], {"train": 27, "val": 9, "test": 9})
            self.assertTrue(all(v == 0 for v in report["overlap"].values()))
            self.assertEqual(original, (labels / "train_subset0.csv").read_bytes())
            self.assertEqual(frames[0]["Species"].iloc[0], dataset.CLASS_NAMES[0])

    def test_overlap_missing_and_wrong_label_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels = create_fixture(Path(tmp))
            a, b, c = dataset.load_split(labels)
            bad = b.copy()
            bad.loc[0, "Filename"] = a.loc[0, "Filename"]
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                dataset.check_split(a, bad, c, images, expected_total=45)
            (images / c.loc[0, "Filename"]).unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                dataset.check_split(a, b, c, images, expected_total=45)
            a["Label"] = a["Label"].astype(float)
            a.loc[0, "Label"] = 0.5
            with self.assertRaisesRegex(ValueError, "integers"):
                dataset.DeepWeedsDataset(a, images)

    def test_eval_order_and_seeded_train_sampler(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels = create_fixture(Path(tmp))
            a, b, _ = dataset.load_split(labels)
            transform = dataset.build_transforms(False, 16)
            loader = dataset.make_loader(b, images, transform, 4, False, num_workers=0)
            filenames = [n for _, _, names in loader for n in names]
            self.assertEqual(filenames, b["Filename"].tolist())
            batch = next(iter(loader))
            self.assertEqual(tuple(batch[0].shape), (4, 3, 16, 16))
            for sampler in (None, "balanced"):
                def sample():
                    loader = dataset.make_loader(a, images, transform, 4, True, sampler, 0, seed=17)
                    return [n for _, _, names in loader for n in names]
                self.assertEqual(sample(), sample())
            with self.assertRaises(ValueError):
                dataset.make_loader(b, images, transform, 4, False, "balanced", 0)

    def test_all_augmentations_and_deterministic_eval(self):
        image = Image.new("RGB", (256, 256), (70, 130, 210))
        for aug in ("basic", "color", "trivial", "randaug"):
            x = dataset.build_transforms(True, 32, aug)(image)
            self.assertEqual(x.shape, (3, 32, 32))
            self.assertTrue(torch.isfinite(x).all())
        transform = dataset.build_transforms(False, 32)
        self.assertTrue(torch.equal(transform(image), transform(image)))


class TestLosses(unittest.TestCase):
    def test_focal_zero_and_smoothing_zero_equal_ce(self):
        torch.manual_seed(2)
        z, y = torch.randn(12, 9, requires_grad=True), torch.arange(12) % 9
        expected = F.cross_entropy(z, y)
        for criterion in (losses.FocalLoss(0), losses.LabelSmoothingCE(0)):
            actual = criterion(z, y)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
            grad = torch.autograd.grad(actual, z, retain_graph=True)[0]
            grad_ce = torch.autograd.grad(expected, z, retain_graph=True)[0]
            torch.testing.assert_close(grad, grad_ce, atol=1e-6, rtol=0)

    def test_extreme_logits_and_alpha_buffer(self):
        z = torch.tensor([[1000., -1000.] + [0.] * 7], requires_grad=True)
        criterion = losses.FocalLoss(2, torch.ones(9))
        loss = criterion(z, torch.tensor([1]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertIn("alpha", dict(criterion.named_buffers()))

    def test_weights_are_normalized_and_effective_number_is_stable(self):
        counts = np.arange(1, 10) * 100
        inverse = losses.class_weights(counts)
        expected = torch.tensor(1 / counts, dtype=torch.float32)
        torch.testing.assert_close(inverse, expected / expected.mean())
        for beta in (0., .99, .999999999):
            w = losses.class_weights(counts, beta)
            self.assertTrue(torch.isfinite(w).all())
            self.assertAlmostEqual(w.mean().item(), 1., places=6)
        with self.assertRaises(ValueError):
            losses.class_weights([0] + [1] * 8)
        with self.assertRaises(ValueError):
            losses.class_weights(counts, 1.)

    def test_mixup_and_mixed_loss_match_soft_targets(self):
        x, y = torch.randn(4, 3, 8, 8), torch.arange(4)
        permutation = torch.tensor([3, 2, 1, 0])
        with patch("losses.np.random.beta", return_value=.3), patch("losses.torch.randperm", return_value=permutation):
            mixed, targets = losses.mix_batch(x, y, mode="mixup")
        torch.testing.assert_close(mixed, .3 * x + .7 * x[permutation])
        self.assertTrue(torch.equal(targets[1], y[permutation]))
        z = torch.randn(4, 9)
        soft = .3 * F.one_hot(y, 9) + .7 * F.one_hot(y[permutation], 9)
        torch.testing.assert_close(losses.mixed_loss(nn.CrossEntropyLoss(), z, targets),
                                   F.cross_entropy(z, soft.float()))

    def test_cutmix_clipped_area_and_labels(self):
        x = torch.stack([torch.full((3, 8, 8), float(i)) for i in range(4)])
        y = torch.arange(4)
        permutation = torch.tensor([3, 2, 1, 0])
        with patch("losses.np.random.beta", return_value=0.), \
             patch("losses.np.random.randint", return_value=0), \
             patch("losses.torch.randperm", return_value=permutation):
            mixed, (ya, yb, lam) = losses.mix_batch(x, y)
        changed = (mixed[0, 0] != x[0, 0]).sum().item()
        self.assertEqual(changed, 16)  # 8x8 rectangle clipped to 4x4 at the corner.
        self.assertEqual(lam, 1 - changed / 64)
        self.assertTrue(torch.equal(ya, y))
        self.assertTrue(torch.equal(yb, y[permutation]))
        self.assertTrue(torch.equal(x[0], torch.zeros_like(x[0])))

    def test_invalid_loss_inputs(self):
        with self.assertRaises(ValueError):
            losses.build_criterion("invalid")
        with self.assertRaises(ValueError):
            losses.build_criterion("ce_weighted")
        with self.assertRaises(ValueError):
            losses.mix_batch(torch.zeros(2, 3, 8, 8), torch.zeros(2, dtype=torch.long), alpha=0)


class TestTraining(unittest.TestCase):
    def setUp(self):
        train.set_seed(42)
        self.m = TinyModel()
        self.cfg = train.Config(epochs=3, warmup_epochs=1, batch_size=4, img_size=16,
                                num_workers=0, amp=False, device="cpu", grad_clip=None)

    def test_groups_cover_trainable_params_and_exempt_bias_norm(self):
        groups = model.param_groups(self.m, 1e-4, 1e-3, .05)
        ids = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in self.m.parameters()})
        for g in groups:
            for p in g["params"]:
                if p.ndim <= 1:
                    self.assertEqual(g["weight_decay"], 0)
        for g in groups:
            if g["group_name"].startswith("head"):
                self.assertEqual(g["lr"], 1e-3)

    def test_frozen_backbone_stays_eval_and_unchanged(self):
        model.freeze_backbone(self.m)
        bn = self.m.features[1]
        before = copy.deepcopy(self.m.state_dict())
        optimizer = train.build_optimizer(self.m, self.cfg)
        scheduler = train.build_scheduler(optimizer, self.cfg, 1)
        batch = [(torch.randn(4, 3, 16, 16), torch.arange(4), list("abcd"))]
        train.train_one_epoch(self.m, batch, nn.CrossEntropyLoss(), optimizer, scheduler,
                              torch.amp.GradScaler("cuda", enabled=False), self.cfg, "cpu")
        self.assertFalse(bn.training)
        self.assertTrue(self.m.head.training)
        for key in before:
            if key.startswith("features"):
                torch.testing.assert_close(before[key], self.m.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(before["head.weight"], self.m.head.weight))

    def test_ema_parameters_and_buffers(self):
        ema = train.EMA(self.m, .5)
        before = self.m.head.weight.detach().clone()
        with torch.no_grad():
            self.m.head.weight.add_(2)
            self.m.features[1].running_mean.fill_(3)
            self.m.features[1].num_batches_tracked.fill_(4)
        ema.update(self.m)
        torch.testing.assert_close(ema.model.head.weight, before + 1)
        torch.testing.assert_close(ema.model.features[1].running_mean, torch.full((4,), 3.))
        self.assertEqual(ema.model.features[1].num_batches_tracked.item(), 4)
        self.assertFalse(ema.model.training)
        self.assertTrue(all(not p.requires_grad for p in ema.model.parameters()))

    def test_scheduler_warmup_cosine_and_skip_ema_on_overflow(self):
        optimizer = train.build_optimizer(self.m, self.cfg)
        scheduler = train.build_scheduler(optimizer, self.cfg, 2)
        lrs = [optimizer.param_groups[0]["lr"]]
        for _ in range(6):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])
        self.assertAlmostEqual(lrs[0], .5e-4)
        self.assertAlmostEqual(lrs[1], 1e-4)
        self.assertEqual(lrs[-1], 0)
        self.assertTrue(all(a >= b for a, b in zip(lrs[2:], lrs[3:])))

        class SkippedScaler:
            def __init__(self):
                self.scale_value = 8.
            def is_enabled(self): return True
            def get_scale(self): return self.scale_value
            def scale(self, loss): return loss
            def step(self, opt): pass
            def update(self): self.scale_value /= 2

        ema = train.EMA(self.m, .5)
        old_epoch = scheduler.last_epoch
        before = self.m.head.weight.clone()
        with patch.object(ema, "update", wraps=ema.update) as update:
            stats = train.train_one_epoch(self.m,
                [(torch.randn(4, 3, 16, 16), torch.arange(4), list("abcd"))],
                nn.CrossEntropyLoss(), optimizer, scheduler, SkippedScaler(),
                self.cfg, "cpu", ema)
        self.assertEqual(stats["skipped_updates"], 1)
        self.assertEqual(scheduler.last_epoch, old_epoch)
        update.assert_not_called()
        torch.testing.assert_close(self.m.head.weight, before)

    def test_evaluate_order_no_grad_no_buffer_updates(self):
        before = copy.deepcopy(self.m.state_dict())
        batches = [(torch.randn(2, 3, 16, 16), torch.tensor([4, 2]), ["z.jpg", "a.jpg"]),
                   (torch.randn(1, 3, 16, 16), torch.tensor([3]), ["b.jpg"])]
        names, y, z, loss = train.evaluate(self.m, batches, nn.CrossEntropyLoss(), "cpu")
        self.assertEqual(names, ["z.jpg", "a.jpg", "b.jpg"])
        np.testing.assert_array_equal(y, [4, 2, 3])
        self.assertEqual(z.shape, (3, 9))
        self.assertTrue(math.isfinite(loss))
        for key in before:
            torch.testing.assert_close(before[key], self.m.state_dict()[key])
        self.assertTrue(all(p.grad is None for p in self.m.parameters()))

    def test_cli_types_and_invalid_config(self):
        self.assertEqual(train.parse_overrides(["seed=1", "amp=false", "ema_decay=none"]),
                         {"seed": 1, "amp": False, "ema_decay": None})
        for bad in (["unknown=1"], ["amp=yes"], ["seed=1.2"], ["seed=1", "seed=2"]):
            with self.assertRaises(ValueError):
                train.parse_overrides(bad)
        with self.assertRaises(ValueError):
            train.validate_config(replace(self.cfg, exp_id="../escape"))
        with self.assertRaises(ValueError):
            train.validate_config(replace(self.cfg, lr_head=float("nan")))


class TestRunContract(unittest.TestCase):
    def test_one_epoch_pilot_keeps_scheduler_and_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            images, labels = create_fixture(path)
            cfg = train.Config(
                exp_id="pilot", init="scratch", epochs=3, warmup_epochs=1, img_size=16,
                batch_size=9, num_workers=0, amp=False, device="cpu", mix="cutmix", ema_decay=.8,
                images_dir=str(images), labels_dir=str(labels), out_dir=str(path / "runs"),
                pred_dir=str(path / "predictions"), curves_dir=str(path / "curves"))
            actual_check = dataset.check_split
            with patch.object(train.dataset, "check_split", side_effect=lambda *a, **k: actual_check(*a, expected_total=45, **k)), \
                 patch.object(train.model_utils, "build_model", side_effect=lambda *a, **k: TinyModel()), \
                 patch.object(train, "_verify_provenance", return_value=None):
                pilot = train.run(cfg, stop_after_epoch=1)
                self.assertEqual(pilot["completed_epochs"], 1)
                self.assertEqual(pilot["scheduled_epochs"], 3)
                self.assertFalse(pilot["training_complete"])
                checkpoint = torch.load(train.run_dir(cfg) / "last.pt", weights_only=False)
                self.assertEqual(checkpoint["epoch"], 1)
                saved_config = json.loads((train.run_dir(cfg) / "config.json").read_text())
                self.assertEqual(saved_config["epochs"], 3)
                history = pd.read_csv(train.run_dir(cfg) / "history.csv")
                self.assertEqual(len(history), 1)
                self.assertGreater(history.val_seconds.iloc[0], 0)
                self.assertGreater(history.epoch_train_val_seconds.iloc[0], history.train_seconds.iloc[0])
                self.assertTrue(pd.isna(history.peak_allocated_gib.iloc[0]))
                resumed = train.run(replace(cfg, resume=True))
                direct_cfg = replace(cfg, exp_id="full")
                direct = train.run(direct_cfg)
            self.assertTrue(resumed["training_complete"])
            self.assertEqual(resumed["val"], direct["val"])
            for checkpoint_name in ("best.pt", "last.pt"):
                a = torch.load(train.run_dir(cfg) / checkpoint_name, weights_only=False)["model"]
                b = torch.load(train.run_dir(direct_cfg) / checkpoint_name, weights_only=False)["model"]
                for name in a:
                    torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
            self.assertFalse(train.pred_path(cfg, "test").exists())

    def test_pilot_rejects_test_export_and_invalid_epoch(self):
        cfg = train.Config(epochs=12)
        for value in (0, 13, True, 1.5):
            with self.assertRaisesRegex(ValueError, "stop_after_epoch"):
                train.run(cfg, stop_after_epoch=value)
        with self.assertRaisesRegex(ValueError, "disable test"):
            train.run(replace(cfg, save_test_predictions=True), stop_after_epoch=1)

    def test_interrupted_resume_matches_uninterrupted_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            images, labels = create_fixture(path)
            cfg = train.Config(
                exp_id="interrupted", init="scratch", epochs=3, warmup_epochs=1, img_size=16,
                batch_size=9, num_workers=0, amp=False, device="cpu", ema_decay=.8,
                mix="cutmix", sampler="balanced",
                images_dir=str(images), labels_dir=str(labels), out_dir=str(path / "runs"),
                pred_dir=str(path / "predictions"), curves_dir=str(path / "curves"))
            actual_check, actual_epoch = dataset.check_split, train.train_one_epoch
            calls = 0

            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated interruption")
                return actual_epoch(*args, **kwargs)

            with patch.object(train.dataset, "check_split", side_effect=lambda *a, **k: actual_check(*a, expected_total=45, **k)), \
                 patch.object(train.model_utils, "build_model", side_effect=lambda *a, **k: TinyModel()), \
                 patch.object(train, "_verify_provenance", return_value=None):
                with patch.object(train, "train_one_epoch", side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        train.run(cfg)
                resumed = train.run(replace(cfg, resume=True))
                direct_cfg = replace(cfg, exp_id="uninterrupted")
                direct = train.run(direct_cfg)
            self.assertEqual(resumed["best_epoch"], direct["best_epoch"])
            self.assertEqual(resumed["val"], direct["val"])
            for checkpoint in ("best.pt", "last.pt"):
                a = torch.load(train.run_dir(cfg) / checkpoint, weights_only=False)["model"]
                b = torch.load(train.run_dir(direct_cfg) / checkpoint, weights_only=False)["model"]
                for name in a:
                    torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)

    def test_test_export_requires_opt_in_and_cannot_be_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            images, labels = create_fixture(path)
            cfg = train.Config(
                exp_id="test_fixture_only", init="scratch", epochs=1, warmup_epochs=0,
                img_size=16, batch_size=9, num_workers=0, amp=False, device="cpu",
                save_test_predictions=True, images_dir=str(images), labels_dir=str(labels),
                out_dir=str(path / "runs"), pred_dir=str(path / "predictions"),
                curves_dir=str(path / "curves"))
            actual_check = dataset.check_split
            with patch.object(train.dataset, "check_split", side_effect=lambda *a, **k: actual_check(*a, expected_total=45, **k)), \
                 patch.object(train.model_utils, "build_model", side_effect=lambda *a, **k: TinyModel()), \
                 patch.object(train, "_verify_provenance", return_value=None):
                result = train.run(cfg)
            self.assertEqual(result["test"]["n"], 9)
            self.assertTrue((train.run_dir(cfg) / "test_started.json").exists())
            self.assertTrue((train.run_dir(cfg) / "test_completed.json").exists())
            pred = read_pred(str(train.pred_path(cfg, "test")))
            self.assertEqual(len(pred.filenames), 9)
            with self.assertRaisesRegex(FileExistsError, "Test was already"):
                train.run(replace(cfg, resume=True))

    def test_run_outputs_and_resume_without_touching_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            images, labels = create_fixture(path)
            cfg = train.Config(
                exp_id="fixture", init="scratch", epochs=2, warmup_epochs=0, img_size=16,
                batch_size=9, num_workers=0, amp=False, device="cpu", ema_decay=.8,
                images_dir=str(images), labels_dir=str(labels), out_dir=str(path / "runs"),
                pred_dir=str(path / "predictions"), curves_dir=str(path / "curves"))
            actual_check = dataset.check_split
            actual_make = dataset.make_loader

            def fixture_check(*args, **kwargs):
                return actual_check(*args, expected_total=45, **kwargs)

            def guarded_loader(df, *args, **kwargs):
                if any("fixture_36" == Path(n).stem for n in df["Filename"]):
                    raise AssertionError("Test DataLoader must not be created")
                return actual_make(df, *args, **kwargs)

            def fake_model(*args, **kwargs): return TinyModel()

            with patch.object(train.dataset, "check_split", side_effect=fixture_check), \
                 patch.object(train.dataset, "make_loader", side_effect=guarded_loader), \
                 patch.object(train.model_utils, "build_model", side_effect=fake_model):
                # Production provenance must not be applied to synthetic fixture CSVs.
                with patch.object(train, "_verify_provenance", return_value=None):
                    result = train.run(cfg)
                    resumed = train.run(replace(cfg, resume=True))
            run_path = train.run_dir(cfg)
            self.assertEqual(result["val"], resumed["val"])
            self.assertNotIn("test", result)
            self.assertFalse(train.pred_path(cfg, "test").exists())
            self.assertEqual(read_pred(str(train.pred_path(cfg, "val"))).probs.shape, (9, 9))
            for filename in ("config.json", "environment.json", "split_check.json", "split_hashes.json",
                             "history.csv", "best.pt", "last.pt", "val_logits.npy", "summary.json"):
                self.assertTrue((run_path / filename).exists(), filename)
            self.assertEqual(len(pd.read_csv(run_path / "history.csv")), 2)
            self.assertEqual(result["best_epoch"], 1)  # First epoch wins this fixture's tie.
            self.assertEqual(len(list((path / "curves").glob("*.png"))), 1)
            with self.assertRaises(FileExistsError):
                train.run(cfg)


if __name__ == "__main__":
    unittest.main()

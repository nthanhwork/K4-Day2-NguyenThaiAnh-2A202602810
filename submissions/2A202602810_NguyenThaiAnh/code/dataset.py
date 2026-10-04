"""DeepWeeds CSV splits, deterministic validation, transforms and loaders."""
from __future__ import annotations

import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _validate_frame(df: pd.DataFrame, name: str) -> None:
    if not {"Filename", "Label"} <= set(df.columns):
        raise ValueError(f"{name}: requires Filename, Label")
    if df.empty:
        raise ValueError(f"{name}: empty split")
    names = df["Filename"]
    if names.isna().any() or not names.map(lambda n: isinstance(n, str) and bool(n.strip())).all():
        raise ValueError(f"{name}: invalid Filename")
    if names.duplicated().any():
        raise ValueError(f"{name}: duplicate Filename")
    if not names.map(lambda n: Path(n).name == n and n not in {".", ".."}).all():
        raise ValueError(f"{name}: filenames must be basenames, not paths")
    labels = pd.to_numeric(df["Label"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError(f"{name}: labels must be finite integers")
    if ((labels < 0) | (labels >= NUM_CLASSES)).any() or ("Species" in df and df["Species"].isna().any()):
        raise ValueError(f"{name}: labels must be 0..8 and Species cannot be null")


def load_split(labels_dir: str | Path, fold: int = 0):
    """Read the author's three CSVs without filtering or resplitting."""
    if type(fold) is not int or not 0 <= fold <= 4:
        raise ValueError("fold must be an integer in 0..4")
    frames = []
    reference = pd.read_csv(Path(labels_dir) / "labels.csv")
    _validate_frame(reference, "labels.csv")
    if "Species" not in reference:
        raise ValueError("labels.csv requires Species")
    mapping = reference[["Label", "Species"]].drop_duplicates()
    if mapping["Label"].duplicated().any():
        raise ValueError("labels.csv has inconsistent species mapping")
    species = mapping.set_index("Label")["Species"]
    for split in ("train", "val", "test"):
        df = pd.read_csv(Path(labels_dir) / f"{split}_subset{fold}.csv")
        _validate_frame(df, split)
        if "Species" not in df:
            # Enrich only in memory: original author split files have two columns.
            df["Species"] = df["Label"].map(species)
            _validate_frame(df, split)
        frames.append(df)
    return tuple(frames)


def check_split(train_df, val_df, test_df, images_dir, *,
                expected_total: int = 17509, reference_df=None, verify_images: bool = False) -> dict:
    """Validate split metadata and file existence; optionally decode all images.

    expected_total is configurable for unit-test fixtures, never changed in run().
    Test images are decoded only when explicitly requested by a separate data audit.
    """
    frames = dict(zip(("train", "val", "test"), (train_df, val_df, test_df)))
    for name, df in frames.items():
        _validate_frame(df, name)
    names = {name: set(df["Filename"]) for name, df in frames.items()}
    overlap = {f"{a}_{b}": len(names[a] & names[b]) for a, b in itertools.combinations(names, 2)}
    if any(overlap.values()):
        raise ValueError(f"Overlapping splits: {overlap}; stop and report to instructor")
    union = set.union(*names.values())
    if len(union) != expected_total:
        raise ValueError(f"Expected {expected_total} distinct images, found {len(union)}")
    counts = {name: len(df) for name, df in frames.items()}
    ratios = {name: n / expected_total for name, n in counts.items()}
    if any(abs(ratios[name] - ratio) > 0.01
           for name, ratio in (("train", 0.6), ("val", 0.2), ("test", 0.2))):
        raise ValueError(f"Split ratios outside 60/20/20 ±1 percentage point: {ratios}")
    combined = pd.concat(frames.values(), ignore_index=True)
    mapping = combined[["Label", "Species"]].drop_duplicates()
    if mapping["Label"].duplicated().any() or mapping["Species"].duplicated().any():
        raise ValueError("Inconsistent Label/Species mapping across splits")
    if set(mapping["Label"].astype(int)) != set(range(NUM_CLASSES)):
        raise ValueError("All nine classes must exist in the combined split")
    reference_discrepancies = []
    if reference_df is not None:
        _validate_frame(reference_df, "labels.csv")
        if union != set(reference_df["Filename"]):
            raise ValueError("Split union differs from labels.csv")
        reference = reference_df.set_index("Filename")[["Label", "Species"]].sort_index()
        observed = combined.set_index("Filename")[["Label", "Species"]].sort_index()
        different = (observed["Label"].to_numpy(dtype=int) != reference["Label"].to_numpy(dtype=int)) | (
            observed["Species"].astype(str).to_numpy() != reference["Species"].astype(str).to_numpy())
        # The original author files contain one such discrepancy in fold 0.
        # S1 and eval.check_against_csv require original SPLIT targets; never repair them.
        for filename in observed.index[different]:
            reference_discrepancies.append({
                "Filename": str(filename), "split_label": int(observed.loc[filename, "Label"]),
                "labels_csv_label": int(reference.loc[filename, "Label"]),
                "split_species": str(observed.loc[filename, "Species"]),
                "labels_csv_species": str(reference.loc[filename, "Species"]),
            })
    images_dir = Path(images_dir)
    missing = sorted(n for n in union if not (images_dir / n).is_file())
    if missing:
        raise ValueError(f"{len(missing)} missing images, e.g. {missing[:5]}")
    if verify_images:
        for name in sorted(union):
            with Image.open(images_dir / name) as image:
                image.load()
    report = {
        "n": counts, "ratios": ratios,
        "per_class": {name: df["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
                      .astype(int).to_dict() for name, df in frames.items()},
        "overlap": overlap, "union": len(union), "missing": 0,
        "images_decoded": verify_images,
        "reference_discrepancies": reference_discrepancies,
        "label_policy": "Use original split CSV targets unchanged; labels.csv supplies class names",
    }
    return report


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic", *,
                     mean=IMAGENET_MEAN, std=IMAGENET_STD, interpolation="bicubic",
                     resize_size: int | None = None):
    """Crop/flip on train; deterministic resize + center crop on val/test."""
    if img_size <= 0 or aug not in {"basic", "color", "trivial", "randaug"}:
        raise ValueError("Invalid image size or augmentation")
    modes = {"bilinear": InterpolationMode.BILINEAR, "bicubic": InterpolationMode.BICUBIC}
    if interpolation not in modes:
        raise ValueError(f"Unsupported interpolation: {interpolation}")
    interpolation_mode = modes[interpolation]
    if train:
        ops = [transforms.RandomResizedCrop(img_size, interpolation=interpolation_mode),
               transforms.RandomHorizontalFlip()]
        if aug == "color":
            ops.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))
        elif aug == "trivial":
            ops.append(transforms.TrivialAugmentWide(interpolation=interpolation_mode))
        elif aug == "randaug":
            ops.append(transforms.RandAugment(interpolation=interpolation_mode))
    else:
        resize_size = resize_size if resize_size is not None else round(img_size * 256 / 224)
        if resize_size < img_size:
            raise ValueError("Evaluation resize_size must be >= img_size")
        ops = [transforms.Resize(resize_size, interpolation=interpolation_mode),
               transforms.CenterCrop(img_size)]
    return transforms.Compose([*ops, transforms.ToTensor(), transforms.Normalize(mean, std)])


class DeepWeedsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        _validate_frame(df, "dataset")
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        with Image.open(self.images_dir / filename) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["Label"]), filename


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df, images_dir, transform, batch_size, train, sampler=None, num_workers=2,
                *, seed: int | None = None):
    """Seed batch order, worker RNG and optional train-only balanced sampling."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("Invalid batch_size/num_workers")
    if sampler not in {None, "balanced"} or (sampler is not None and not train):
        raise ValueError("Only train supports sampler='balanced'")
    generator = torch.Generator().manual_seed(torch.initial_seed() if seed is None else seed)
    dataset = DeepWeedsDataset(df, images_dir, transform)
    weighted_sampler = None
    if sampler == "balanced":
        counts = df["Label"].value_counts()
        weights = torch.as_tensor([1.0 / counts[int(y)] for y in df["Label"]], dtype=torch.double)
        weighted_sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                                  generator=generator)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train and weighted_sampler is None,
        sampler=weighted_sampler, num_workers=num_workers,
        drop_last=train and len(dataset) >= batch_size,
        pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker, generator=generator,
    )

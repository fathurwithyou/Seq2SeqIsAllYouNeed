from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ..utils.image_utils import load_image_batch

INTEL_KAGGLE_DATASET = "puneet6060/intel-image-classification"
INTEL_CLASSES: tuple[str, ...] = (
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class IntelImageSplit:
    paths: list[Path]
    labels: np.ndarray
    class_names: tuple[str, ...] = field(default=INTEL_CLASSES)

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def to_arrays(
        self,
        *,
        target_size: tuple[int, int] = (150, 150),
        normalize: bool = True,
        batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        images: list[np.ndarray] = []
        for start in range(0, len(self.paths), batch_size):
            chunk = self.paths[start : start + batch_size]
            images.append(
                load_image_batch(chunk, target_size=target_size, normalize=normalize)
            )
        return np.concatenate(images, axis=0), self.labels.copy()


def _collect_split(root: Path, class_names: Sequence[str]) -> tuple[list[Path], np.ndarray]:
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    paths: list[Path] = []
    labels: list[int] = []
    for class_name in class_names:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for entry in sorted(class_dir.iterdir()):
            if entry.is_file() and entry.suffix.lower() in _IMAGE_SUFFIXES:
                paths.append(entry)
                labels.append(class_to_index[class_name])
    return paths, np.asarray(labels, dtype=np.int64)


def _resolve_split_dir(root: Path, split: str) -> Path:
    candidates = [
        root / f"seg_{split}" / f"seg_{split}",
        root / f"seg_{split}",
        root / split,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(
            (candidate / name).is_dir() for name in INTEL_CLASSES
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not find '{split}' split under {root}. Expected layout "
        f"'seg_{split}/seg_{split}/<class>/...' or '{split}/<class>/...'."
    )


def load_intel_image_split(
    root: str | Path,
    split: str,
    *,
    class_names: Sequence[str] = INTEL_CLASSES,
) -> IntelImageSplit:
    root_path = Path(root)
    split_dir = _resolve_split_dir(root_path, split)
    paths, labels = _collect_split(split_dir, class_names)
    return IntelImageSplit(paths=paths, labels=labels, class_names=tuple(class_names))


def load_intel_image_dataset(
    root: str | Path,
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
    class_names: Sequence[str] = INTEL_CLASSES,
) -> dict[str, IntelImageSplit]:
    train = load_intel_image_split(root, "train", class_names=class_names)
    test = load_intel_image_split(root, "test", class_names=class_names)

    if validation_fraction <= 0:
        return {"train": train, "validation": IntelImageSplit([], np.asarray([], dtype=np.int64), tuple(class_names)), "test": test}

    rng = np.random.default_rng(seed)
    indices = np.arange(len(train))
    rng.shuffle(indices)
    val_size = int(round(validation_fraction * len(indices)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    validation = IntelImageSplit(
        paths=[train.paths[i] for i in val_indices],
        labels=train.labels[val_indices],
        class_names=tuple(class_names),
    )
    train_split = IntelImageSplit(
        paths=[train.paths[i] for i in train_indices],
        labels=train.labels[train_indices],
        class_names=tuple(class_names),
    )
    return {"train": train_split, "validation": validation, "test": test}


def download_intel_image_dataset() -> Path:
    cached_root = Path.home() / ".cache" / "kagglehub" / "datasets" / "puneet6060" / "intel-image-classification" / "versions"
    if cached_root.is_dir():
        cached_versions = sorted(cached_root.iterdir(), reverse=True)
        for version in cached_versions:
            if (version / "seg_train").is_dir() and (version / "seg_test").is_dir():
                print("Path to dataset files:", version)
                return version

    import kagglehub

    path = Path(kagglehub.dataset_download(INTEL_KAGGLE_DATASET))
    print("Path to dataset files:", path)
    return path


__all__ = [
    "INTEL_CLASSES",
    "INTEL_KAGGLE_DATASET",
    "IntelImageSplit",
    "download_intel_image_dataset",
    "load_intel_image_split",
    "load_intel_image_dataset",
]

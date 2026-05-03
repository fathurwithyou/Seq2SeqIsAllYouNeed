from __future__ import annotations

from pathlib import Path

import numpy as np

FLICKR8K_KAGGLE_DATASET = "adityajn105/flickr8k"


def load_features(features_path: Path, paths_file: Path) -> dict[str, np.ndarray]:
    features = np.load(features_path)
    names = [
        line.strip()
        for line in paths_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(names) != features.shape[0]:
        raise ValueError(
            f"feature count ({features.shape[0]}) != paths count ({len(names)})"
        )
    return {name: features[index] for index, name in enumerate(names)}


def download_flickr8k_dataset() -> Path:
    cached_root = Path.home() / ".cache" / "kagglehub" / "datasets" / "adityajn105" / "flickr8k" / "versions"
    if cached_root.is_dir():
        cached_versions = sorted(cached_root.iterdir(), reverse=True)
        for version in cached_versions:
            if (version / "captions.txt").is_file() and (version / "Images").is_dir():
                print("Path to dataset files:", version)
                return version

    import kagglehub

    path = Path(kagglehub.dataset_download(FLICKR8K_KAGGLE_DATASET))
    print("Path to dataset files:", path)
    return path


__all__ = ["FLICKR8K_KAGGLE_DATASET", "download_flickr8k_dataset", "load_features"]

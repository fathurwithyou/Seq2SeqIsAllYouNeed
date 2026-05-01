from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def split_dataset(
    dataset: Sequence[Any] | tuple[Sequence[Any], ...],
    left_size: float | int | None = None,
    right_size: float | int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
):
    """Splits array-like data into two disjoint subsets.

    This is a small NumPy-oriented counterpart of `keras.utils.split_dataset`.
    It supports a single sequence or a tuple of aligned sequences, which is the
    format used by the local training scripts.
    """
    arrays = tuple(np.asarray(item) for item in dataset) if _is_tuple_data(dataset) else (np.asarray(dataset),)
    size = len(arrays[0])
    if any(len(array) != size for array in arrays):
        raise ValueError("All dataset components must have the same length.")

    left_count, right_count = _resolve_split_sizes(size, left_size, right_size)
    indices = np.arange(size)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    left_indices = indices[:left_count]
    right_indices = indices[left_count : left_count + right_count]
    left = tuple(array[left_indices] for array in arrays)
    right = tuple(array[right_indices] for array in arrays)
    if not _is_tuple_data(dataset):
        return left[0], right[0]
    return left, right


def _is_tuple_data(dataset) -> bool:
    return isinstance(dataset, tuple)


def _resolve_split_sizes(
    size: int,
    left_size: float | int | None,
    right_size: float | int | None,
) -> tuple[int, int]:
    if left_size is None and right_size is None:
        left_size = 0.5
    if left_size is not None:
        left_count = _normalize_size(left_size, size, "left_size")
        right_count = size - left_count if right_size is None else _normalize_size(right_size, size, "right_size")
    else:
        right_count = _normalize_size(right_size, size, "right_size")
        left_count = size - right_count
    if left_count < 0 or right_count < 0 or left_count + right_count > size:
        raise ValueError(
            "`left_size` and `right_size` must describe non-negative splits "
            "whose sum does not exceed the dataset length."
        )
    return left_count, right_count


def _normalize_size(value: float | int | None, total: int, name: str) -> int:
    if value is None:
        raise ValueError(f"`{name}` must not be None here.")
    if isinstance(value, float):
        if value < 0 or value > 1:
            raise ValueError(f"`{name}` as a float must be in the [0, 1] range.")
        return int(round(total * value))
    if isinstance(value, int):
        return value
    raise ValueError(f"`{name}` must be an int, float, or None.")


__all__ = ["split_dataset"]

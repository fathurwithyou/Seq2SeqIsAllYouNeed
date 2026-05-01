from __future__ import annotations

import numpy as np


def normalize(x, axis: int = -1, order: int = 2) -> np.ndarray:
    """Normalizes a NumPy array along an axis."""
    array = np.asarray(x)
    norm = np.atleast_1d(np.linalg.norm(array, order, axis))
    norm[norm == 0] = 1
    axis = axis or -1
    return array / np.expand_dims(norm, axis)


def to_categorical(x, num_classes: int | None = None) -> np.ndarray:
    """Converts integer class labels to a one-hot matrix."""
    labels = np.asarray(x, dtype="int64")
    input_shape = labels.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = input_shape[:-1]
    labels = labels.reshape(-1)
    if num_classes is None:
        num_classes = int(np.max(labels)) + 1 if labels.size else 0
    output = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    if labels.size:
        output[np.arange(labels.shape[0]), labels] = 1.0
    return output.reshape(input_shape + (num_classes,))


__all__ = ["normalize", "to_categorical"]

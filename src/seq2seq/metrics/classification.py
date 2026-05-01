from __future__ import annotations

from typing import Sequence

import numpy as np


def _to_labels(predictions: np.ndarray) -> np.ndarray:
    pred = np.asarray(predictions)
    if pred.ndim >= 2 and pred.shape[-1] > 1:
        return pred.argmax(axis=-1)
    return pred


def accuracy(y_true: Sequence[int] | np.ndarray, y_pred: np.ndarray) -> float:
    pred = np.asarray(y_pred)
    target = np.asarray(y_true)
    if pred.ndim >= 2 and pred.shape == target.shape:
        target = target.argmax(axis=-1)
    target = target.ravel()
    pred = _to_labels(pred).ravel()
    if target.size != pred.size:
        raise ValueError(f"size mismatch: y_true={target.size}, y_pred={pred.size}")
    return float(np.mean(pred == target))


def confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int | None = None,
) -> np.ndarray:
    target = np.asarray(y_true, dtype=np.int64).ravel()
    pred = _to_labels(y_pred).astype(np.int64).ravel()
    if target.size != pred.size:
        raise ValueError(f"size mismatch: y_true={target.size}, y_pred={pred.size}")
    if num_classes is None:
        num_classes = int(max(target.max(initial=-1), pred.max(initial=-1)) + 1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(target, pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            matrix[t, p] += 1
    return matrix


def f1_per_class(
    y_true: Sequence[int] | np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int | None = None,
) -> np.ndarray:
    matrix = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    tp = np.diag(matrix).astype(np.float64)
    predicted_positive = matrix.sum(axis=0).astype(np.float64)
    actual_positive = matrix.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, predicted_positive, out=np.zeros_like(tp), where=predicted_positive > 0)
    recall = np.divide(tp, actual_positive, out=np.zeros_like(tp), where=actual_positive > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return f1


def macro_f1(
    y_true: Sequence[int] | np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int | None = None,
) -> float:
    scores = f1_per_class(y_true, y_pred, num_classes=num_classes)
    if scores.size == 0:
        return 0.0
    return float(scores.mean())


__all__ = ["accuracy", "confusion_matrix", "f1_per_class", "macro_f1"]

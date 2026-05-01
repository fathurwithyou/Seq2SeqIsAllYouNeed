from __future__ import annotations

from typing import Any

import numpy as np

from ..tensor import Tensor


def squeeze_or_expand_to_same_rank(y_true, y_pred):
    true_ndim = getattr(y_true, "ndim", np.asarray(y_true).ndim)
    pred_ndim = getattr(y_pred, "ndim", np.asarray(y_pred).ndim)
    if true_ndim == pred_ndim:
        return y_true, y_pred
    if true_ndim == pred_ndim + 1 and getattr(y_true, "shape")[-1] == 1:
        return _squeeze(y_true, axis=-1), y_pred
    if pred_ndim == true_ndim + 1 and getattr(y_pred, "shape")[-1] == 1:
        return y_true, _squeeze(y_pred, axis=-1)
    return y_true, y_pred


def _squeeze(value, axis: int):
    if isinstance(value, Tensor):
        return value.squeeze(axis=axis)
    return np.squeeze(value, axis=axis)


class Loss:
    def __init__(
        self,
        name: str | None = None,
        reduction: str | None = "sum_over_batch_size",
        dtype: np.dtype | str | None = None,
    ) -> None:
        if reduction == "mean":
            reduction = "sum_over_batch_size"
        if reduction not in {"sum", "sum_over_batch_size", "none", None}:
            raise ValueError(f"Invalid reduction: {reduction!r}")
        self.name = name or self.__class__.__name__
        self.reduction = reduction
        self.dtype = np.dtype(dtype or np.float32)

    def __call__(self, y_true, y_pred, sample_weight=None):
        losses = self.call(y_true, y_pred)
        if sample_weight is not None:
            losses = losses * _as_loss_tensor(sample_weight, losses)
        return reduce_values(losses, self.reduction)

    def call(self, y_true, y_pred):
        raise NotImplementedError(f"{type(self).__name__} does not implement call()")

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reduction": self.reduction,
            "dtype": str(self.dtype),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        return cls(**config)


def reduce_values(values, reduction: str | None):
    if reduction in {None, "none"}:
        return values
    if reduction == "sum":
        return values.sum() if isinstance(values, Tensor) else np.sum(values)
    return values.mean() if isinstance(values, Tensor) else np.mean(values)


def _as_loss_tensor(value, reference):
    if isinstance(reference, Tensor):
        return value if isinstance(value, Tensor) else Tensor(np.asarray(value))
    return np.asarray(value)


__all__ = ["Loss", "reduce_values", "squeeze_or_expand_to_same_rank"]

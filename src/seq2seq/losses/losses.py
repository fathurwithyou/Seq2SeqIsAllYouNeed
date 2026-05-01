from __future__ import annotations

from typing import Callable

import numpy as np

from ..tensor import Tensor
from .loss import Loss, squeeze_or_expand_to_same_rank


def _to_tensor(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(np.asarray(x))


def _to_int_array(x) -> np.ndarray:
    return x.data.astype(np.int64) if isinstance(x, Tensor) else np.asarray(x, dtype=np.int64)


def _one_hot(indices: np.ndarray, num_classes: int, dtype: np.dtype) -> np.ndarray:
    output = np.zeros((*indices.shape, num_classes), dtype=dtype)
    np.put_along_axis(output.reshape(-1, num_classes), indices.reshape(-1, 1), 1.0, axis=1)
    return output


class LossFunctionWrapper(Loss):
    def __init__(
        self,
        fn: Callable,
        reduction: str | None = "sum_over_batch_size",
        name: str | None = None,
        dtype: np.dtype | str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, reduction=reduction, dtype=dtype)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(self._fn_kwargs)
        return config

    def __repr__(self) -> str:
        return f"<LossFunctionWrapper({self.fn}, kwargs={self._fn_kwargs})>"


class CategoricalCrossentropy(LossFunctionWrapper):
    def __init__(
        self,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        axis: int = -1,
        reduction: str | None = "sum_over_batch_size",
        name: str = "categorical_crossentropy",
        dtype: np.dtype | str | None = None,
    ) -> None:
        super().__init__(
            categorical_crossentropy,
            name=name,
            reduction=reduction,
            dtype=dtype,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.from_logits = bool(from_logits)
        self.label_smoothing = float(label_smoothing)
        self.axis = int(axis)

    def get_config(self) -> dict:
        config = Loss.get_config(self)
        config.update(
            {
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
                "axis": self.axis,
            }
        )
        return config


class SparseCategoricalCrossentropy(LossFunctionWrapper):
    def __init__(
        self,
        from_logits: bool = False,
        ignore_class: int | None = None,
        axis: int = -1,
        reduction: str | None = "sum_over_batch_size",
        name: str = "sparse_categorical_crossentropy",
        dtype: np.dtype | str | None = None,
    ) -> None:
        super().__init__(
            sparse_categorical_crossentropy,
            name=name,
            reduction=reduction,
            dtype=dtype,
            from_logits=from_logits,
            ignore_class=ignore_class,
            axis=axis,
        )
        self.from_logits = bool(from_logits)
        self.ignore_class = ignore_class
        self.axis = int(axis)

    def get_config(self) -> dict:
        config = Loss.get_config(self)
        config.update(
            {
                "from_logits": self.from_logits,
                "ignore_class": self.ignore_class,
                "axis": self.axis,
            }
        )
        return config


def categorical_crossentropy(
    y_true,
    y_pred,
    from_logits: bool = False,
    label_smoothing: float = 0.0,
    axis: int = -1,
    eps: float = 1e-7,
):
    if isinstance(axis, bool):
        raise ValueError(f"`axis` must be an integer. Received: axis={axis!r}")
    pred = _to_tensor(y_pred)
    if from_logits:
        pred = pred.softmax(axis=axis)
    target = _to_tensor(y_true)

    if label_smoothing:
        num_classes = pred.shape[axis]
        target = target * (1.0 - label_smoothing) + label_smoothing / num_classes

    return -(target * (pred + eps).log()).sum(axis=axis)


def sparse_categorical_crossentropy(
    y_true,
    y_pred,
    from_logits: bool = False,
    ignore_class: int | None = None,
    axis: int = -1,
):
    if isinstance(axis, bool):
        raise ValueError(f"`axis` must be an integer. Received: axis={axis!r}")
    pred = _to_tensor(y_pred)
    target = _to_int_array(y_true)
    if target.ndim == pred.ndim and target.shape[-1] == 1:
        target = np.squeeze(target, axis=-1)

    if axis != -1 and axis != pred.ndim - 1:
        pred = pred.transpose(*_move_axis_to_end(pred.ndim, axis))

    valid_mask = None
    safe_target = target
    if ignore_class is not None:
        valid_mask = target != ignore_class
        safe_target = np.where(valid_mask, target, 0)

    one_hot = _one_hot(safe_target, pred.shape[-1], pred.data.dtype)
    losses = categorical_crossentropy(
        one_hot,
        pred,
        from_logits=from_logits,
        label_smoothing=0.0,
        axis=-1,
    )
    if valid_mask is not None:
        return losses * Tensor(valid_mask.astype(pred.data.dtype))
    return losses


def _move_axis_to_end(ndim: int, axis: int) -> tuple[int, ...]:
    axis = axis % ndim
    axes = [index for index in range(ndim) if index != axis]
    axes.append(axis)
    return tuple(axes)


__all__ = [
    "LossFunctionWrapper",
    "CategoricalCrossentropy",
    "SparseCategoricalCrossentropy",
    "categorical_crossentropy",
    "sparse_categorical_crossentropy",
]

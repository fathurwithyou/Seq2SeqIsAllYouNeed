from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InputSpec:
    """Lightweight input contract for Keras-style layer shape validation."""

    dtype: str | np.dtype | None = None
    shape: tuple[int | None, ...] | None = None
    ndim: int | None = None
    max_ndim: int | None = None
    min_ndim: int | None = None
    axes: Mapping[int, int | None] | None = None
    allow_last_axis_squeeze: bool = False
    name: str | None = None
    optional: bool = False

    def __post_init__(self) -> None:
        if self.shape is not None:
            object.__setattr__(self, "shape", tuple(self.shape))
            if self.ndim is None:
                object.__setattr__(self, "ndim", len(self.shape))
        if self.axes is not None:
            object.__setattr__(self, "axes", dict(self.axes))

    def assert_compatible(
        self,
        inputs: Any,
        *,
        layer_name: str | None = None,
    ) -> None:
        if inputs is None and self.optional:
            return

        shape = self._shape_of(inputs)
        dtype = self._dtype_of(inputs)
        display_name = layer_name or self.name or "Layer"

        if self.dtype is not None and dtype is not None:
            expected = np.dtype(self.dtype)
            if np.dtype(dtype) != expected:
                raise ValueError(
                    f'{display_name} expected dtype "{expected}", got "{np.dtype(dtype)}"'
                )

        if shape is None:
            return

        ndim = len(shape)
        if self.ndim is not None and ndim != self.ndim:
            raise ValueError(
                f"{display_name} expected ndim={self.ndim}, got shape {shape}"
            )
        if self.min_ndim is not None and ndim < self.min_ndim:
            raise ValueError(
                f"{display_name} expected ndim >= {self.min_ndim}, got shape {shape}"
            )
        if self.max_ndim is not None and ndim > self.max_ndim:
            raise ValueError(
                f"{display_name} expected ndim <= {self.max_ndim}, got shape {shape}"
            )

        if self.shape is not None:
            self._assert_shape_compatible(shape, display_name)

        if self.axes is not None:
            for axis, expected in self.axes.items():
                normalized_axis = axis if axis >= 0 else ndim + axis
                if expected is None or normalized_axis < 0 or normalized_axis >= ndim:
                    continue
                if shape[normalized_axis] != expected:
                    raise ValueError(
                        f"{display_name} expected axis {axis} to have value "
                        f"{expected}, got shape {shape}"
                    )

    def _assert_shape_compatible(
        self, actual_shape: tuple[int, ...], layer_name: str
    ) -> None:
        expected_shape = self.shape
        if expected_shape is None:
            return
        if self.allow_last_axis_squeeze:
            if actual_shape[-1:] == (1,):
                actual_shape = actual_shape[:-1]
            if expected_shape[-1:] == (1,):
                expected_shape = expected_shape[:-1]
        if len(actual_shape) != len(expected_shape):
            raise ValueError(
                f"{layer_name} expected shape {expected_shape}, got {actual_shape}"
            )
        for actual, expected in zip(actual_shape, expected_shape):
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{layer_name} expected shape {expected_shape}, got {actual_shape}"
                )

    @staticmethod
    def _shape_of(inputs: Any) -> tuple[int, ...] | None:
        if hasattr(inputs, "shape"):
            return tuple(inputs.shape)
        return None

    @staticmethod
    def _dtype_of(inputs: Any) -> np.dtype | None:
        if hasattr(inputs, "dtype"):
            return np.dtype(inputs.dtype)
        if hasattr(inputs, "data") and hasattr(inputs.data, "dtype"):
            return np.dtype(inputs.data.dtype)
        return None


__all__ = ["InputSpec"]

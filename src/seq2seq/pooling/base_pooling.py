from __future__ import annotations

import numpy as np

from ...tensor import Tensor
from ..layer import Layer


def to_array(x):
    return x.data if isinstance(x, Tensor) else np.asarray(x)


def wrap_like(ref, value: np.ndarray):
    return Tensor(value) if isinstance(ref, Tensor) else value


class BasePooling2D(Layer):
    def __init__(
        self,
        pool_size: int | tuple[int, int] = (2, 2),
        *,
        strides: int | tuple[int, int] | None = None,
        padding: str = "valid",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        self.pool_size = tuple(pool_size)
        self.strides = strides
        self.padding = padding

    def extra_repr(self) -> str:
        return (
            f"pool_size={self.pool_size}, strides={self.strides}, padding={self.padding!r}"
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "pool_size": self.pool_size,
                "strides": self.strides,
                "padding": self.padding,
            }
        )
        return config


__all__ = ["BasePooling2D", "to_array", "wrap_like"]

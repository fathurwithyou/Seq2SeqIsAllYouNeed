from __future__ import annotations

import numpy as np

from .initializer import Initializer


class Constant(Initializer):
    def __init__(self, value=0.0) -> None:
        self.value = value

    def __call__(self, shape, dtype=None):
        return np.full(shape, self.value, dtype=np.dtype(dtype or np.float32))

    def get_config(self) -> dict:
        return {"value": self.value}


class Zeros(Initializer):
    def __call__(self, shape, dtype=None):
        return np.zeros(shape, dtype=np.dtype(dtype or np.float32))


class Ones(Initializer):
    def __call__(self, shape, dtype=None):
        return np.ones(shape, dtype=np.dtype(dtype or np.float32))


__all__ = ["Constant", "Zeros", "Ones"]

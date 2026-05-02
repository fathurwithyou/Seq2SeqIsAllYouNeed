from __future__ import annotations

import numpy as np

from ...tensor import Tensor
from ..layer import Layer


class Flatten(Layer):
    def call(self, inputs):
        array = inputs.data if isinstance(inputs, Tensor) else np.asarray(inputs)
        flat = array.reshape(array.shape[0], -1)
        return Tensor(flat) if isinstance(inputs, Tensor) else flat


__all__ = ["Flatten"]

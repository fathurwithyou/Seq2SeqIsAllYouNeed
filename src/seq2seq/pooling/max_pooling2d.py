from __future__ import annotations

from ... import ops as functional_pooling
from .base_pooling import BasePooling2D, to_array, wrap_like


class MaxPooling2D(BasePooling2D):
    def call(self, inputs):
        outputs = functional_pooling.max_pool2d(
            to_array(inputs), self.pool_size, self.strides, self.padding
        )
        return wrap_like(inputs, outputs)

__all__ = ["MaxPooling2D"]

from __future__ import annotations

from ... import ops as functional_pooling
from ..layer import Layer
from .base_pooling import to_array, wrap_like


class GlobalAveragePooling2D(Layer):
    def call(self, inputs):
        return wrap_like(inputs, functional_pooling.global_avg_pool2d(to_array(inputs)))

__all__ = ["GlobalAveragePooling2D"]

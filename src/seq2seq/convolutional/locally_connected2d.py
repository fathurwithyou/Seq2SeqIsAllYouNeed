from __future__ import annotations

import numpy as np

from ...activations import apply_activation
from ...ops.conv import (
    _normalize_pair,
    _pad_amounts,
    locally_connected2d_with_size,
)
from ...tensor import Tensor
from ..layer import Layer


class LocallyConnected2D(Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int | tuple[int, int],
        *,
        strides: int | tuple[int, int] = (1, 1),
        padding: str = "valid",
        activation: str | None = None,
        use_bias: bool = True,
        kernel_initializer: str = "he_uniform",
        bias_initializer: str = "zeros",
        input_shape: tuple[int, int, int] | None = None,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.filters = int(filters)
        self.kernel_size = _normalize_pair(kernel_size)
        self.strides = _normalize_pair(strides)
        self.padding = str(padding).lower()
        self.activation = activation
        self.use_bias = bool(use_bias)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.seed = seed
        self.output_size = None
        self.kernel = None
        self.bias = None
        if input_shape is not None:
            self.build((None, *input_shape))

    def build(self, input_shape) -> None:
        input_height = int(input_shape[1])
        input_width = int(input_shape[2])
        input_channels = int(input_shape[3])
        kh, kw = self.kernel_size
        out_h, _, _ = _pad_amounts(input_height, kh, self.strides[0], self.padding)
        out_w, _, _ = _pad_amounts(input_width, kw, self.strides[1], self.padding)
        self.output_size = (out_h, out_w)
        self.kernel = self.add_weight(
            (out_h * out_w, kh * kw * input_channels, self.filters),
            initializer=self.kernel_initializer,
            name="kernel",
            seed=self.seed,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                (out_h, out_w, self.filters),
                initializer=self.bias_initializer,
                name="bias",
            )
        self.built = True

    def call(self, inputs):
        array = inputs.data if isinstance(inputs, Tensor) else np.asarray(inputs)
        outputs = locally_connected2d_with_size(
            array,
            self.kernel.data,
            self.bias.data if self.bias is not None else None,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )
        outputs = apply_activation(outputs, self.activation)
        return Tensor(outputs) if isinstance(inputs, Tensor) else outputs

    def extra_repr(self) -> str:
        return (
            f"filters={self.filters}, kernel_size={self.kernel_size}, "
            f"strides={self.strides}, padding={self.padding!r}, "
            f"activation={self.activation!r}"
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "activation": self.activation,
                "use_bias": self.use_bias,
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
                "seed": self.seed,
            }
        )
        return config


__all__ = ["LocallyConnected2D"]

from __future__ import annotations

import numpy as np

from ... import activations
from ...tensor import Tensor
from ..input_spec import InputSpec
from ..layer import Layer


class Dense(Layer):
    def __init__(
        self,
        units: int,
        *,
        activation: str | None = None,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        input_dim: int | None = None,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(units, int) or units <= 0:
            raise ValueError(
                "Received an invalid value for `units`, expected a positive "
                f"integer. Received: units={units}"
            )
        super().__init__(name=name)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = bool(use_bias)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.seed = seed
        self.kernel = None
        self.bias = None
        self.input_spec = InputSpec(min_ndim=2)
        if input_dim is not None:
            self.build((None, input_dim))

    def build(self, input_shape) -> None:
        input_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            (input_dim, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            seed=self.seed,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                (self.units,),
                initializer=self.bias_initializer,
                name="bias",
            )
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        if isinstance(inputs, Tensor):
            outputs = inputs @ self.kernel
            if self.bias is not None:
                outputs = outputs + self.bias
            return self.activation(outputs)
        outputs = np.asarray(inputs) @ self.kernel.data
        if self.bias is not None:
            outputs = outputs + self.bias.data
        return self.activation(outputs)

    def compute_output_shape(self, input_shape) -> tuple:
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def extra_repr(self) -> str:
        return (
            f"units={self.units}, activation={activations.serialize(self.activation)!r}, "
            f"use_bias={self.use_bias}"
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
                "seed": self.seed,
            }
        )
        return config


__all__ = ["Dense"]

from __future__ import annotations

from ...activations import apply_activation
from ..layer import Layer


class Activation(Layer):
    def __init__(self, activation: str, *, axis: int = -1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.activation = str(activation).lower()
        self.axis = int(axis)

    def call(self, x):
        return apply_activation(x, self.activation, axis=self.axis)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"activation": self.activation, "axis": self.axis})
        return config


class ReLU(Layer):
    def call(self, x):
        return apply_activation(x, "relu")


class LeakyReLU(Layer):
    def __init__(self, negative_slope: float = 0.2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.negative_slope = float(negative_slope)

    def call(self, x):
        from ...activations import leaky_relu

        return leaky_relu(x, negative_slope=self.negative_slope)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"negative_slope": self.negative_slope})
        return config


class ELU(Layer):
    def __init__(self, alpha: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = float(alpha)

    def call(self, x):
        from ...activations import elu

        return elu(x, alpha=self.alpha)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"alpha": self.alpha})
        return config


class GELU(Layer):
    def __init__(self, approximate: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.approximate = bool(approximate)

    def call(self, x):
        from ...activations import gelu

        return gelu(x, approximate=self.approximate)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"approximate": self.approximate})
        return config


class Sigmoid(Layer):
    def call(self, x):
        return apply_activation(x, "sigmoid")


class Tanh(Layer):
    def call(self, x):
        return apply_activation(x, "tanh")


class Softmax(Layer):
    def __init__(self, axis: int = -1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.axis = int(axis)

    def call(self, x):
        return apply_activation(x, "softmax", axis=self.axis)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


class Softplus(Layer):
    def call(self, x):
        return apply_activation(x, "softplus")


class Softsign(Layer):
    def call(self, x):
        return apply_activation(x, "softsign")


__all__ = [
    "Activation",
    "ELU",
    "GELU",
    "LeakyReLU",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "Softplus",
    "Softsign",
    "Tanh",
]

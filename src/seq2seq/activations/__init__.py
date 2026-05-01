from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..tensor import Tensor, to_numpy, wrap_like


def relu(x):
    if isinstance(x, Tensor) and x.requires_grad:
        return x.relu()
    return wrap_like(x, np.maximum(to_numpy(x), 0.0))


def relu6(x):
    return wrap_like(x, np.minimum(np.maximum(to_numpy(x), 0.0), 6.0))


def leaky_relu(x, negative_slope: float = 0.2):
    if isinstance(x, Tensor) and x.requires_grad:
        return x.relu() - float(negative_slope) * (-x).relu()
    arr = to_numpy(x)
    return wrap_like(x, np.where(arr >= 0.0, arr, float(negative_slope) * arr))


def sigmoid(x):
    if isinstance(x, Tensor) and x.requires_grad:
        return x.sigmoid()
    clipped = np.clip(to_numpy(x), -60.0, 60.0)
    return wrap_like(x, 1.0 / (1.0 + np.exp(-clipped)))


def tanh(x):
    if isinstance(x, Tensor) and x.requires_grad:
        return x.tanh()
    return wrap_like(x, np.tanh(to_numpy(x)))


def softmax(x, axis: int = -1):
    if isinstance(x, Tensor) and x.requires_grad:
        return x.softmax(axis=axis)
    arr = to_numpy(x)
    shifted = arr - arr.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return wrap_like(x, exp / exp.sum(axis=axis, keepdims=True))


def log_softmax(x, axis: int = -1):
    if isinstance(x, Tensor) and x.requires_grad:
        return x.softmax(axis=axis).log()
    arr = to_numpy(x)
    shifted = arr - arr.max(axis=axis, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    return wrap_like(x, shifted - logsumexp)


def softplus(x):
    if isinstance(x, Tensor) and x.requires_grad:
        return (x.exp() + 1.0).log()
    arr = to_numpy(x)
    return wrap_like(x, np.logaddexp(arr, 0.0))


def softsign(x):
    arr = to_numpy(x)
    return wrap_like(x, arr / (1.0 + np.abs(arr)))


def swish(x):
    if isinstance(x, Tensor) and x.requires_grad:
        return x * sigmoid(x)
    arr = to_numpy(x)
    clipped = np.clip(arr, -60.0, 60.0)
    return wrap_like(x, arr / (1.0 + np.exp(-clipped)))


silu = swish


def gelu(x, approximate: bool = True):
    if isinstance(x, Tensor) and x.requires_grad and approximate:
        coeff = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + (coeff * (x + 0.044715 * (x ** 3))).tanh())
    arr = to_numpy(x)
    if approximate:
        out = 0.5 * arr * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (arr + 0.044715 * arr**3)))
    else:
        erf = np.vectorize(math.erf)
        out = 0.5 * arr * (1.0 + erf(arr / np.sqrt(2.0)))
    return wrap_like(x, out)


def mish(x):
    if isinstance(x, Tensor) and x.requires_grad:
        return x * softplus(x).tanh()
    arr = to_numpy(x)
    return wrap_like(x, arr * np.tanh(np.logaddexp(arr, 0.0)))


def elu(x, alpha: float = 1.0):
    arr = to_numpy(x)
    return wrap_like(x, np.where(arr > 0.0, arr, float(alpha) * np.expm1(arr)))


def selu(x):
    scale = 1.0507009873554805
    alpha = 1.6732632423543772
    arr = to_numpy(x)
    return wrap_like(x, scale * np.where(arr > 0.0, arr, alpha * np.expm1(arr)))


def hard_sigmoid(x):
    arr = to_numpy(x)
    return wrap_like(x, np.clip(arr / 6.0 + 0.5, 0.0, 1.0))


def hard_silu(x):
    arr = to_numpy(x)
    return wrap_like(x, arr * np.clip(arr / 6.0 + 0.5, 0.0, 1.0))


hard_swish = hard_silu


def exponential(x):
    if isinstance(x, Tensor) and x.requires_grad:
        return x.exp()
    return wrap_like(x, np.exp(to_numpy(x)))


def linear(x):
    return x


_ALIASES: dict[str, Callable] = {
    "elu": elu,
    "exponential": exponential,
    "gelu": gelu,
    "hard_sigmoid": hard_sigmoid,
    "hard_silu": hard_silu,
    "hard_swish": hard_swish,
    "relu": relu,
    "relu6": relu6,
    "sigmoid": sigmoid,
    "silu": silu,
    "selu": selu,
    "softplus": softplus,
    "softsign": softsign,
    "tanh": tanh,
    "softmax": softmax,
    "log_softmax": log_softmax,
    "leaky_relu": leaky_relu,
    "mish": mish,
    "swish": swish,
    "linear": linear,
}


def get(identifier):
    if identifier is None:
        return linear
    if isinstance(identifier, str):
        name = identifier.lower()
        if name in _ALIASES:
            return _ALIASES[name]
        raise ValueError(f"Unsupported activation: {identifier!r}")
    if callable(identifier):
        return identifier
    raise ValueError(f"Unsupported activation: {identifier!r}")


def serialize(activation):
    if activation is None:
        return "linear"
    if isinstance(activation, str):
        get(activation)
        return activation.lower()
    for name, fn in _ALIASES.items():
        if activation is fn:
            return name
    if callable(activation) and hasattr(activation, "__name__"):
        return activation.__name__
    raise ValueError(f"Unsupported activation: {activation!r}")


def apply_activation(x, activation, *, axis: int = -1):
    if activation is None:
        return x
    if isinstance(activation, str) and activation.lower() == "linear":
        return x
    fn = get(activation)
    if fn is softmax:
        return softmax(x, axis=axis)
    if fn is log_softmax:
        return log_softmax(x, axis=axis)
    return fn(x)


__all__ = [
    "apply_activation",
    "elu",
    "exponential",
    "gelu",
    "get",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "leaky_relu",
    "linear",
    "log_softmax",
    "mish",
    "relu",
    "relu6",
    "selu",
    "serialize",
    "sigmoid",
    "silu",
    "softmax",
    "softplus",
    "softsign",
    "swish",
    "tanh",
]

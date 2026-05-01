from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def simple_rnn_cell(
    x: np.ndarray,
    h_prev: np.ndarray,
    kernel: np.ndarray,
    recurrent_kernel: np.ndarray,
    bias: np.ndarray | None,
    activation: str = "tanh",
) -> np.ndarray:
    z = x @ kernel + h_prev @ recurrent_kernel
    if bias is not None:
        z = z + bias
    if activation == "tanh":
        return np.tanh(z)
    if activation == "relu":
        return np.maximum(z, 0.0)
    if activation == "sigmoid":
        return _sigmoid(z)
    if activation in (None, "linear"):
        return z
    raise ValueError(f"Unsupported RNN activation {activation!r}")


def lstm_cell(
    x: np.ndarray,
    h_prev: np.ndarray,
    c_prev: np.ndarray,
    kernel: np.ndarray,
    recurrent_kernel: np.ndarray,
    bias: np.ndarray | None,
    activation: str = "tanh",
    recurrent_activation: str = "sigmoid",
) -> tuple[np.ndarray, np.ndarray]:
    z = x @ kernel + h_prev @ recurrent_kernel
    if bias is not None:
        z = z + bias

    h_dim = c_prev.shape[-1]
    i = z[..., :h_dim]
    f = z[..., h_dim : 2 * h_dim]
    c = z[..., 2 * h_dim : 3 * h_dim]
    o = z[..., 3 * h_dim : 4 * h_dim]

    if recurrent_activation == "sigmoid":
        i = _sigmoid(i)
        f = _sigmoid(f)
        o = _sigmoid(o)
    elif recurrent_activation == "tanh":
        i = np.tanh(i)
        f = np.tanh(f)
        o = np.tanh(o)
    else:
        raise ValueError(f"Unsupported recurrent_activation {recurrent_activation!r}")

    if activation == "tanh":
        c_tilde = np.tanh(c)
    elif activation == "relu":
        c_tilde = np.maximum(c, 0.0)
    elif activation in (None, "linear"):
        c_tilde = c
    else:
        raise ValueError(f"Unsupported activation {activation!r}")

    c_t = f * c_prev + i * c_tilde
    if activation == "tanh":
        h_t = o * np.tanh(c_t)
    elif activation == "relu":
        h_t = o * np.maximum(c_t, 0.0)
    else:
        h_t = o * c_t
    return h_t, c_t


__all__ = ["simple_rnn_cell", "lstm_cell"]

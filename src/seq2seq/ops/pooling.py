from __future__ import annotations

import numpy as np

from .conv import _normalize_pair, _pad_amounts


def _pool2d(
    x: np.ndarray,
    pool_size: tuple[int, int],
    strides: tuple[int, int] | int | None,
    padding: str,
    op: str,
) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"expected (N,H,W,C), got shape {x.shape}")
    ph, pw = pool_size
    sh, sw = _normalize_pair(strides if strides is not None else (ph, pw))
    _, h, w, _ = x.shape

    out_h, ph_top, ph_bot = _pad_amounts(h, ph, sh, padding)
    out_w, pw_left, pw_right = _pad_amounts(w, pw, sw, padding)

    if ph_top or ph_bot or pw_left or pw_right:
        pad_value = -np.inf if op == "max" else 0.0
        x = np.pad(
            x,
            ((0, 0), (ph_top, ph_bot), (pw_left, pw_right), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )

    windows = np.lib.stride_tricks.sliding_window_view(x, (ph, pw), axis=(1, 2))
    windows = windows[:, ::sh, ::sw, :, :, :]

    if op == "max":
        return windows.max(axis=(-2, -1))
    if op == "avg":
        return windows.mean(axis=(-2, -1))
    raise ValueError(f"unknown pooling op {op!r}")


def max_pool2d(
    x: np.ndarray,
    pool_size: tuple[int, int] | int = (2, 2),
    strides: tuple[int, int] | int | None = None,
    padding: str = "valid",
) -> np.ndarray:
    return _pool2d(x, _normalize_pair(pool_size), strides, padding, "max")


def avg_pool2d(
    x: np.ndarray,
    pool_size: tuple[int, int] | int = (2, 2),
    strides: tuple[int, int] | int | None = None,
    padding: str = "valid",
) -> np.ndarray:
    return _pool2d(x, _normalize_pair(pool_size), strides, padding, "avg")


def global_avg_pool2d(x: np.ndarray) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"expected (N,H,W,C), got shape {x.shape}")
    return x.mean(axis=(1, 2))


def global_max_pool2d(x: np.ndarray) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"expected (N,H,W,C), got shape {x.shape}")
    return x.max(axis=(1, 2))


__all__ = [
    "avg_pool2d",
    "max_pool2d",
    "global_avg_pool2d",
    "global_max_pool2d",
]

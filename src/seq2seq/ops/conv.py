from __future__ import annotations

import numpy as np


def _normalize_pair(value) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    a, b = value
    return int(a), int(b)


def _pad_amounts(in_size: int, kernel: int, stride: int, padding: str) -> tuple[int, int, int]:
    if padding == "valid":
        out = (in_size - kernel) // stride + 1
        return out, 0, 0
    if padding == "same":
        out = (in_size + stride - 1) // stride
        total_pad = max((out - 1) * stride + kernel - in_size, 0)
        before = total_pad // 2
        after = total_pad - before
        return out, before, after
    raise ValueError(f"padding must be 'valid' or 'same', got {padding!r}")


def im2col(
    x: np.ndarray,
    kernel_size: tuple[int, int],
    strides: tuple[int, int] = (1, 1),
    padding: str = "valid",
) -> tuple[np.ndarray, tuple[int, int]]:
    if x.ndim != 4:
        raise ValueError(f"expected 4D input (N,H,W,C), got shape {x.shape}")

    n, h, w, c = x.shape
    kh, kw = kernel_size
    sh, sw = strides

    out_h, ph_top, ph_bot = _pad_amounts(h, kh, sh, padding)
    out_w, pw_left, pw_right = _pad_amounts(w, kw, sw, padding)

    if ph_top or ph_bot or pw_left or pw_right:
        x = np.pad(
            x,
            ((0, 0), (ph_top, ph_bot), (pw_left, pw_right), (0, 0)),
            mode="constant",
        )

    windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))
    windows = windows[:, ::sh, ::sw, :, :, :]
    windows = np.transpose(windows, (0, 1, 2, 4, 5, 3))
    cols = windows.reshape(n, out_h * out_w, kh * kw * c)
    return cols, (out_h, out_w)


def conv2d(
    x: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    strides: tuple[int, int] | int = (1, 1),
    padding: str = "valid",
) -> np.ndarray:
    kh, kw, c_in, c_out = kernel.shape
    if x.shape[-1] != c_in:
        raise ValueError(f"input channels {x.shape[-1]} != kernel C_in {c_in}")

    strides_pair = _normalize_pair(strides)
    cols, (out_h, out_w) = im2col(x, (kh, kw), strides_pair, padding)
    n = x.shape[0]

    w_flat = kernel.reshape(kh * kw * c_in, c_out)
    out = cols @ w_flat
    if bias is not None:
        out = out + bias
    return out.reshape(n, out_h, out_w, c_out)


def locally_connected2d(
    x: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    strides: tuple[int, int] | int = (1, 1),
    padding: str = "valid",
) -> np.ndarray:
    if kernel.ndim != 3:
        raise ValueError(
            f"locally_connected2d expects kernel rank 3, got shape {kernel.shape}"
        )
    n_pos, k_flat, c_out = kernel.shape

    c_in = x.shape[-1]
    if k_flat % c_in != 0:
        raise ValueError(
            "kernel inner dimension is not a multiple of input channels"
        )
    kernel_area = k_flat // c_in

    kernel_height = int(np.sqrt(kernel_area))
    if kernel_height * kernel_height != kernel_area:
        raise ValueError(
            "could not infer square kernel size; use locally_connected2d_with_size"
        )
    return locally_connected2d_with_size(
        x,
        kernel,
        bias,
        kernel_size=(kernel_height, kernel_height),
        strides=strides,
        padding=padding,
    )


def locally_connected2d_with_size(
    x: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray | None,
    *,
    kernel_size: tuple[int, int],
    strides: tuple[int, int] | int = (1, 1),
    padding: str = "valid",
) -> np.ndarray:
    n_pos, _, c_out = kernel.shape
    strides_pair = _normalize_pair(strides)
    cols, (out_h, out_w) = im2col(x, kernel_size, strides_pair, padding)

    if out_h * out_w != n_pos:
        raise ValueError(
            f"kernel encodes {n_pos} positions but layer produces {out_h * out_w}; "
            "check input shape, strides, and padding."
        )

    n = x.shape[0]
    out = np.einsum("npk,pkc->npc", cols, kernel)
    if bias is not None:
        out = out + bias.reshape(n_pos, c_out)
    return out.reshape(n, out_h, out_w, c_out)


__all__ = [
    "_normalize_pair",
    "_pad_amounts",
    "im2col",
    "conv2d",
    "locally_connected2d",
    "locally_connected2d_with_size",
]


from __future__ import annotations

import numpy as np

import seq2seq.layers as layers
from seq2seq.ops.conv import (
    conv2d as f_conv2d,
    locally_connected2d_with_size,
)
from seq2seq.ops.pooling import (
    avg_pool2d,
    global_avg_pool2d,
    global_max_pool2d,
    max_pool2d,
)


def _reference_conv2d(x, kernel, bias=None, stride=1, padding="valid"):
    n, h, w, c_in = x.shape
    kh, kw, _, c_out = kernel.shape

    if padding == "same":
        out_h = (h + stride - 1) // stride
        out_w = (w + stride - 1) // stride
        ph = max((out_h - 1) * stride + kh - h, 0)
        pw = max((out_w - 1) * stride + kw - w, 0)
        x = np.pad(
            x,
            ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)),
        )
    else:
        out_h = (h - kh) // stride + 1
        out_w = (w - kw) // stride + 1

    out = np.zeros((n, out_h, out_w, c_out), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            patch = x[:, i * stride : i * stride + kh, j * stride : j * stride + kw, :]
            for k in range(c_out):
                out[:, i, j, k] = (patch * kernel[:, :, :, k]).sum(axis=(1, 2, 3))
    if bias is not None:
        out = out + bias
    return out


def test_conv2d_matches_reference_valid_padding():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 7, 7, 3)).astype(np.float32)
    kernel = rng.standard_normal((3, 3, 3, 4)).astype(np.float32)
    bias = rng.standard_normal((4,)).astype(np.float32)

    fast = f_conv2d(x, kernel, bias, strides=(1, 1), padding="valid")
    slow = _reference_conv2d(x, kernel, bias, stride=1, padding="valid")
    np.testing.assert_allclose(fast, slow, rtol=1e-5, atol=1e-5)


def test_conv2d_matches_reference_same_padding_with_stride():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 8, 8, 2)).astype(np.float32)
    kernel = rng.standard_normal((3, 3, 2, 5)).astype(np.float32)
    bias = rng.standard_normal((5,)).astype(np.float32)

    fast = f_conv2d(x, kernel, bias, strides=(2, 2), padding="same")
    slow = _reference_conv2d(x, kernel, bias, stride=2, padding="same")
    assert fast.shape == (1, 4, 4, 5)
    np.testing.assert_allclose(fast, slow, rtol=1e-5, atol=1e-5)


def test_conv2d_module_load_state_dict_roundtrip():
    layer = layers.Conv2D(4, kernel_size=3, input_shape=(5, 5, 3), padding="same", activation="relu", seed=7)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 5, 5, 3)).astype(np.float32)
    y_first = layer(x)

    state = layer.state_dict()
    other = layers.Conv2D(4, kernel_size=3, input_shape=(5, 5, 3), padding="same", activation="relu", seed=999)
    other.load_state_dict(state)
    y_second = other(x)
    np.testing.assert_allclose(y_first, y_second)


def test_locally_connected_against_per_position_reference():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 4, 4, 2)).astype(np.float32)
    kh, kw, c_in, c_out = 3, 3, 2, 3
    out_h = 4 - kh + 1
    out_w = 4 - kw + 1
    n_pos = out_h * out_w
    kernel = rng.standard_normal((n_pos, kh * kw * c_in, c_out)).astype(np.float32)
    bias = rng.standard_normal((out_h, out_w, c_out)).astype(np.float32)

    fast = locally_connected2d_with_size(
        x, kernel, bias, kernel_size=(kh, kw), strides=(1, 1), padding="valid"
    )


    expected = np.zeros((2, out_h, out_w, c_out), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            pos = i * out_w + j
            patch = x[:, i : i + kh, j : j + kw, :]

            flat = patch.reshape(2, kh * kw * c_in)
            expected[:, i, j, :] = flat @ kernel[pos]
    expected = expected + bias[None, :, :, :]
    np.testing.assert_allclose(fast, expected, rtol=1e-5, atol=1e-5)


def test_pool_layers():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((1, 6, 6, 2)).astype(np.float32)

    mp = max_pool2d(x, (2, 2), strides=(2, 2))
    assert mp.shape == (1, 3, 3, 2)

    np.testing.assert_allclose(mp[0, 0, 0, 0], x[0, 0:2, 0:2, 0].max())

    ap = avg_pool2d(x, (2, 2), strides=(2, 2))
    np.testing.assert_allclose(ap[0, 0, 0, 0], x[0, 0:2, 0:2, 0].mean(), rtol=1e-6)

    gap = global_avg_pool2d(x)
    assert gap.shape == (1, 2)
    np.testing.assert_allclose(gap[0, 0], x[0, :, :, 0].mean(), rtol=1e-6)

    gmp = global_max_pool2d(x)
    np.testing.assert_allclose(gmp[0, 1], x[0, :, :, 1].max())


def test_flatten_uses_row_major_order():
    arr = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    flat = layers.Flatten()(arr)
    assert flat.shape == (2, 60)

    np.testing.assert_array_equal(flat[0], arr[0].reshape(-1))

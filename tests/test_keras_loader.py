
from __future__ import annotations

import numpy as np
import pytest

import seq2seq.layers as layers
import seq2seq.models as models
from seq2seq.saving import (
    assign_weights_in_order,
    load_conv2d,
    load_locally_connected2d,
    load_lstm_cell,
    load_simple_rnn_cell,
    load_weights,
    save_weights,
)


def test_load_conv2d_from_keras_weight_list():
    layer = layers.Conv2D(4, kernel_size=3, input_shape=(5, 5, 3), padding="same")
    kernel = np.random.default_rng(0).standard_normal((3, 3, 3, 4)).astype(np.float32)
    bias = np.random.default_rng(1).standard_normal((4,)).astype(np.float32)
    load_conv2d(layer, [kernel, bias])
    np.testing.assert_array_equal(layer.kernel.data, kernel)
    np.testing.assert_array_equal(layer.bias.data, bias)


def test_load_locally_connected2d_from_keras_weight_list():
    layer = layers.LocallyConnected2D(3, kernel_size=3, input_shape=(5, 5, 2), padding="valid")
    out_h, out_w = layer.output_size
    kernel = np.random.default_rng(0).standard_normal((out_h * out_w, 3 * 3 * 2, 3)).astype(np.float32)
    bias = np.random.default_rng(1).standard_normal((out_h, out_w, 3)).astype(np.float32)
    load_locally_connected2d(layer, [kernel, bias])
    np.testing.assert_array_equal(layer.kernel.data, kernel)
    np.testing.assert_array_equal(layer.bias.data, bias)


def test_custom_keras_locally_connected_matches_scratch_after_weight_transfer():
    tf = pytest.importorskip("tensorflow")
    from experiments.cnn import KerasLocallyConnected2D

    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 5, 5, 2)).astype(np.float32)

    keras_layer = KerasLocallyConnected2D(
        3,
        kernel_size=3,
        padding="valid",
        activation=None,
        name="locally_connected_0",
    )
    keras_out = keras_layer(tf.constant(x)).numpy()

    scratch_layer = layers.LocallyConnected2D(
        3,
        kernel_size=3,
        input_shape=(5, 5, 2),
        padding="valid",
        activation=None,
        name="locally_connected_0",
    )
    load_locally_connected2d(scratch_layer, keras_layer.get_weights())
    scratch_out = scratch_layer(x)

    np.testing.assert_allclose(scratch_out, keras_out, rtol=1e-5, atol=1e-5)


def test_load_lstm_and_rnn_cells():
    lstm = layers.LSTMCell(5, input_dim=4)
    Wx = np.random.default_rng(0).standard_normal((4, 20)).astype(np.float32)
    Wh = np.random.default_rng(1).standard_normal((5, 20)).astype(np.float32)
    b = np.random.default_rng(2).standard_normal((20,)).astype(np.float32)
    load_lstm_cell(lstm, [Wx, Wh, b])
    np.testing.assert_array_equal(lstm.kernel.data, Wx)

    rnn = layers.SimpleRNNCell(3, input_dim=4)
    Wx = np.random.default_rng(3).standard_normal((4, 3)).astype(np.float32)
    Wh = np.random.default_rng(4).standard_normal((3, 3)).astype(np.float32)
    b = np.random.default_rng(5).standard_normal((3,)).astype(np.float32)
    load_simple_rnn_cell(rnn, [Wx, Wh, b])
    np.testing.assert_array_equal(rnn.kernel.data, Wx)
    np.testing.assert_array_equal(rnn.bias.data, b)


def test_assign_weights_in_order_dispatches_per_module_type():
    seq = models.Sequential(
        [
            layers.Conv2D(4, kernel_size=3, input_shape=(8, 8, 3)),
            layers.Dense(2, input_dim=8),
            layers.Embedding(10, 3),
        ]
    )
    conv_w = np.zeros((3, 3, 3, 4), dtype=np.float32)
    conv_b = np.ones((4,), dtype=np.float32)
    dense_w = np.full((8, 2), 0.5, dtype=np.float32)
    dense_b = np.zeros((2,), dtype=np.float32)
    emb_w = np.arange(30, dtype=np.float32).reshape(10, 3)
    assign_weights_in_order(list(seq), [[conv_w, conv_b], [dense_w, dense_b], [emb_w]])

    np.testing.assert_array_equal(seq[0].bias.data, conv_b)
    np.testing.assert_allclose(seq[1].kernel.data, dense_w)
    np.testing.assert_array_equal(seq[2].embeddings.data, emb_w)


def test_native_save_and_load_weights_round_trip(tmp_path):
    model = models.Sequential(
        [
            layers.Dense(3, input_dim=2),
            layers.Dense(1, input_dim=3),
        ]
    )
    x = np.random.default_rng(0).standard_normal((4, 2)).astype(np.float32)
    before = model(x)

    path = tmp_path / "model.weights.npz"
    save_weights(model, path)

    reloaded = models.Sequential(
        [
            layers.Dense(3, input_dim=2),
            layers.Dense(1, input_dim=3),
        ]
    )
    load_weights(reloaded, path)
    after = reloaded(x)

    np.testing.assert_allclose(after, before)


def test_native_save_weights_rejects_unsupported_extension(tmp_path):
    model = models.Sequential([layers.Dense(1, input_dim=2)])
    with pytest.raises(ValueError, match=r"\.weights\.npz"):
        save_weights(model, tmp_path / "model.weights.h5")

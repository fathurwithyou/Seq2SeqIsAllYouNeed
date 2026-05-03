
from __future__ import annotations

import numpy as np

import seq2seq.layers as layers
from seq2seq.ops.rnn import lstm_cell, simple_rnn_cell


def test_simple_rnn_cell_matches_naive_loop():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 6)).astype(np.float32)
    h = rng.standard_normal((4, 5)).astype(np.float32)
    Wx = rng.standard_normal((6, 5)).astype(np.float32)
    Wh = rng.standard_normal((5, 5)).astype(np.float32)
    b = rng.standard_normal((5,)).astype(np.float32)

    out = simple_rnn_cell(x, h, Wx, Wh, b, activation="tanh")
    expected = np.tanh(x @ Wx + h @ Wh + b)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_lstm_cell_matches_paper_definition():
    rng = np.random.default_rng(1)
    n, d, h = 3, 4, 5
    x = rng.standard_normal((n, d)).astype(np.float32)
    h_prev = rng.standard_normal((n, h)).astype(np.float32)
    c_prev = rng.standard_normal((n, h)).astype(np.float32)
    Wx = rng.standard_normal((d, 4 * h)).astype(np.float32)
    Wh = rng.standard_normal((h, 4 * h)).astype(np.float32)
    b = rng.standard_normal((4 * h,)).astype(np.float32)

    h_t, c_t = lstm_cell(x, h_prev, c_prev, Wx, Wh, b)

    z = x @ Wx + h_prev @ Wh + b
    sigmoid = lambda u: 1.0 / (1.0 + np.exp(-u))
    i = sigmoid(z[:, :h])
    f = sigmoid(z[:, h : 2 * h])
    c_tilde = np.tanh(z[:, 2 * h : 3 * h])
    o = sigmoid(z[:, 3 * h : 4 * h])
    expected_c = f * c_prev + i * c_tilde
    expected_h = o * np.tanh(expected_c)

    np.testing.assert_allclose(c_t, expected_c, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(h_t, expected_h, rtol=1e-6, atol=1e-6)


def test_simple_rnn_layer_unrolls_correctly():
    layer = layers.SimpleRNN(2, num_layers=1, return_sequences=True, input_dim=3, seed=0)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 4, 3)).astype(np.float32)
    out = layer(x)
    assert out.shape == (2, 4, 2)


    cell = layer.cells[0]
    h = np.zeros((2, 2), dtype=np.float32)
    expected = np.zeros_like(out)
    for t in range(4):
        h = simple_rnn_cell(x[:, t], h, cell.kernel.data, cell.recurrent_kernel.data, cell.bias.data)
        expected[:, t] = h
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_stacked_lstm_returns_top_layer_sequence():
    layer = layers.LSTM(3, num_layers=2, return_sequences=True, input_dim=4, seed=0)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 5, 4)).astype(np.float32)
    out = layer(x)
    assert out.shape == (1, 5, 3)


    layer_last = layers.LSTM(3, num_layers=2, return_sequences=False, input_dim=4, seed=0)
    layer_last.load_state_dict(layer.state_dict())
    final = layer_last(x)
    np.testing.assert_allclose(final, out[:, -1, :], rtol=1e-6)


def test_embedding_lookup_shape_and_values():
    emb = layers.Embedding(10, 4, mask_zero=True, seed=0)
    ids = np.array([[1, 2, 3], [4, 5, 0]])
    out = emb(ids)
    assert out.shape == (2, 3, 4)
    np.testing.assert_array_equal(out[1, 2], np.zeros(4))
    np.testing.assert_array_equal(out[0, 0], emb.embeddings.data[1])

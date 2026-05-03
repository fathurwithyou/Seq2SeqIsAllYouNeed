from __future__ import annotations

import numpy as np

from seq2seq import losses


def test_categorical_crossentropy_function_returns_per_sample_values():
    y_true = np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32)
    y_pred = np.array([[0.05, 0.95, 0.0], [0.1, 0.8, 0.1]], dtype=np.float32)

    out = losses.categorical_crossentropy(y_true, y_pred).data

    np.testing.assert_allclose(out, -np.log([0.95, 0.1]), rtol=1e-6, atol=1e-6)


def test_sparse_categorical_crossentropy_class_uses_keras_argument_order():
    y_true = np.array([1, 2], dtype=np.int64)
    y_pred = np.array([[0.05, 0.95, 0.0], [0.1, 0.8, 0.1]], dtype=np.float32)

    loss = losses.SparseCategoricalCrossentropy(reduction=None)
    out = loss(y_true, y_pred).data

    np.testing.assert_allclose(out, -np.log([0.95, 0.1]), rtol=1e-6, atol=1e-6)


def test_sparse_categorical_crossentropy_ignore_class_masks_loss():
    y_true = np.array([[1, 0, -1]], dtype=np.int64)
    y_pred = np.array(
        [[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]],
        dtype=np.float32,
    )

    loss = losses.SparseCategoricalCrossentropy(ignore_class=-1, reduction=None)
    out = loss(y_true, y_pred).data

    np.testing.assert_allclose(out, [[-np.log(0.9), -np.log(0.8), 0.0]], rtol=1e-6)


def test_losses_get_and_serialize_round_trip():
    loss = losses.get("sparse_categorical_crossentropy")
    config = losses.serialize(loss)
    clone = losses.deserialize(config)

    assert isinstance(clone, losses.SparseCategoricalCrossentropy)
    assert clone.get_config() == loss.get_config()

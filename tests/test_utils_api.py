from __future__ import annotations

import numpy as np

from seq2seq import utils


def test_to_categorical_matches_keras_shape_contract():
    labels = np.array([[0], [2], [1]])

    out = utils.to_categorical(labels, num_classes=3)

    assert out.shape == (3, 3)
    np.testing.assert_array_equal(
        out,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_normalize_keeps_zero_vectors_stable():
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    out = utils.normalize(x, axis=-1)

    np.testing.assert_allclose(out[0], np.array([0.6, 0.8], dtype=np.float32))
    np.testing.assert_allclose(out[1], np.array([0.0, 0.0], dtype=np.float32))


def test_split_dataset_splits_aligned_tuple_data():
    x = np.arange(10)
    y = x * 10

    (x_left, y_left), (x_right, y_right) = utils.split_dataset(
        (x, y), left_size=0.6, shuffle=False
    )

    np.testing.assert_array_equal(x_left, np.arange(6))
    np.testing.assert_array_equal(y_left, np.arange(6) * 10)
    np.testing.assert_array_equal(x_right, np.arange(6, 10))
    np.testing.assert_array_equal(y_right, np.arange(6, 10) * 10)


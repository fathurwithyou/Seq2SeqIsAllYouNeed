from __future__ import annotations

import numpy as np

from seq2seq import Tensor, concat, ones, stack, tensor, zeros


def test_tensor_metadata_and_numpy_interop():
    x = tensor([[1, 2], [3, 4]], dtype=np.float64)

    assert x.shape == (2, 2)
    assert x.ndim == 2
    assert x.size == 4
    assert x.dtype == np.dtype("float64")
    assert len(x) == 2

    np.testing.assert_array_equal(np.asarray(x), x.data)
    np.testing.assert_array_equal(x.numpy(), x.data)
    assert tensor([7]).item() == 7.0


def test_tensor_indexing_and_iteration():
    x = tensor(np.arange(6).reshape(2, 3))

    np.testing.assert_array_equal(x[0].numpy(), np.array([0, 1, 2]))
    rows = [row.numpy() for row in x]
    np.testing.assert_array_equal(rows[0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(rows[1], np.array([3, 4, 5]))


def test_tensor_core_shape_ops_and_reductions():
    x = tensor(np.arange(6).reshape(2, 3), dtype=np.float32)

    np.testing.assert_array_equal(x.transpose().numpy(), np.array([[0, 3], [1, 4], [2, 5]], dtype=np.float32))
    np.testing.assert_array_equal(x.reshape(3, 2).numpy(), np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32))
    np.testing.assert_array_equal(x.sum(axis=0).numpy(), np.array([3, 5, 7], dtype=np.float32))
    np.testing.assert_array_equal(x.mean(axis=1).numpy(), np.array([1, 4], dtype=np.float32))


def test_tensor_creation_and_join_helpers():
    a = zeros(2, 2)
    b = ones(2, 2)

    np.testing.assert_array_equal(concat([a, b], axis=0).numpy(), np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.float32))
    assert stack([a, b], axis=1).shape == (2, 2, 2)
    assert isinstance(Tensor([1, 2, 3]), Tensor)

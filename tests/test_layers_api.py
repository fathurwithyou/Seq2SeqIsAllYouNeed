from __future__ import annotations

import numpy as np
import pytest

from seq2seq import activations
import seq2seq.layers as layers


def test_layer_serialize_deserialize_round_trip():
    dense = layers.Dense(7, activation="relu", input_dim=3, seed=123, name="head")

    config = layers.serialize(dense)
    clone = layers.deserialize(config)

    assert isinstance(clone, layers.Dense)
    assert clone.name == "head"
    assert clone.units == 7
    assert clone.activation is activations.relu
    assert clone.seed == 123


def test_layers_get_matches_keras_identifier_shapes():
    assert layers.get("Dense") is layers.Dense
    assert layers.get("dense") is layers.Dense
    assert isinstance(layers.get({"class_name": "Flatten", "config": {}}), layers.Flatten)
    assert layers.get(None) is None


def test_input_spec_validates_shape_dtype_and_axes():
    spec = layers.InputSpec(dtype="float32", ndim=2, axes={-1: 4})
    spec.assert_compatible(np.zeros((2, 4), dtype=np.float32), layer_name="dense")

    with pytest.raises(ValueError, match="axis -1"):
        spec.assert_compatible(np.zeros((2, 3), dtype=np.float32), layer_name="dense")

    with pytest.raises(ValueError, match="dtype"):
        spec.assert_compatible(np.zeros((2, 4), dtype=np.float64), layer_name="dense")


def test_layer_call_checks_input_spec():
    class Echo(layers.Layer):
        def __init__(self):
            super().__init__()
            self.input_spec = layers.InputSpec(ndim=2)

        def call(self, inputs):
            return inputs

    with pytest.raises(ValueError, match="ndim=2"):
        Echo()(np.zeros((1, 2, 3), dtype=np.float32))


def test_dense_matches_keras_shape_contract_for_rank_greater_than_two():
    dense = layers.Dense(5, input_dim=4, seed=0)
    x = np.zeros((2, 3, 4), dtype=np.float32)

    y = dense(x)

    assert y.shape == (2, 3, 5)
    assert dense.compute_output_shape(x.shape) == (2, 3, 5)


def test_dense_rejects_invalid_units():
    with pytest.raises(ValueError, match="positive integer"):
        layers.Dense(0)

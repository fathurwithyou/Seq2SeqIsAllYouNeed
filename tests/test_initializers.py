from __future__ import annotations

import numpy as np

from seq2seq import initializers
from seq2seq import layers


def test_get_returns_initializer_instances():
    initializer = initializers.get("glorot_uniform")

    assert isinstance(initializer, initializers.GlorotUniform)
    values = initializer((3, 4), dtype=np.float32)
    assert values.shape == (3, 4)
    assert values.dtype == np.float32


def test_initializer_serialize_deserialize_round_trip():
    initializer = initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=7)

    config = initializers.serialize(initializer)
    clone = initializers.deserialize(config)

    assert isinstance(clone, initializers.RandomUniform)
    assert clone.get_config() == initializer.get_config()
    np.testing.assert_allclose(
        clone((2, 2), dtype=np.float32),
        initializer((2, 2), dtype=np.float32),
    )


def test_add_weight_uses_initializer_object_contract():
    layer = layers.Layer()

    weight = layer.add_weight((2, 3), initializer=initializers.Ones(), name="kernel")

    np.testing.assert_array_equal(weight.data, np.ones((2, 3), dtype=np.float32))
    assert layer.kernel is weight

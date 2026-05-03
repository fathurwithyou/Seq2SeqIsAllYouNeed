from __future__ import annotations

import numpy as np

from seq2seq.optimizers import Adam, SGD, deserialize, get, serialize
from seq2seq.tensor import Tensor


def test_sgd_can_be_built_like_keras_optimizer():
    variable = Tensor([1.0, -1.0], requires_grad=True)
    optimizer = SGD(learning_rate=0.1, momentum=0.0, name="SGD")

    assert optimizer.name == "SGD"
    assert optimizer.learning_rate == 0.1
    assert optimizer.variables == []

    result = optimizer.apply_gradients([(np.array([0.5, -0.5], dtype=np.float32), variable)])

    assert optimizer.iterations == 1
    assert result == optimizer.iterations
    assert optimizer.variables == [variable]
    np.testing.assert_allclose(variable.data, np.array([0.95, -0.95], dtype=np.float32))


def test_learning_rate_setter_updates_param_groups():
    variable = Tensor([1.0], requires_grad=True)
    optimizer = SGD([variable], learning_rate=0.1)

    optimizer.learning_rate = 0.01
    optimizer.apply_gradients([(np.array([1.0], dtype=np.float32), variable)])

    assert optimizer.learning_rate == 0.01
    np.testing.assert_allclose(variable.data, np.array([0.99], dtype=np.float32))


def test_sgd_momentum_matches_keras_velocity_rule():
    variable = Tensor([1.0], requires_grad=True)
    optimizer = SGD([variable], learning_rate=0.1, momentum=0.9)

    optimizer.apply_gradients([(np.array([1.0], dtype=np.float32), variable)])
    np.testing.assert_allclose(variable.data, np.array([0.9], dtype=np.float32))

    optimizer.apply_gradients([(np.array([1.0], dtype=np.float32), variable)])
    np.testing.assert_allclose(variable.data, np.array([0.71], dtype=np.float32), rtol=1e-6)


def test_sgd_validates_momentum_like_keras():
    for value in (-0.1, 1.1, 1):
        try:
            SGD(momentum=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"momentum={value!r} should be rejected")


def test_adam_uses_keras_constructor_names():
    variable = Tensor([1.0], requires_grad=True)
    optimizer = Adam(
        learning_rate=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=True,
    )

    optimizer.apply_gradients([(np.array([1.0], dtype=np.float32), variable)])

    assert optimizer.name == "adam"
    assert optimizer.iterations == 1
    assert "vhat" in optimizer.state[id(variable)]
    assert variable.data[0] < 1.0


def test_optimizer_get_serialize_deserialize_api():
    optimizer = Adam(learning_rate=0.01, beta_1=0.8, beta_2=0.9, name="adam")

    config = serialize(optimizer)
    assert config["class_name"] == "Adam"
    assert config["config"]["learning_rate"] == 0.01

    restored = deserialize(config)
    assert isinstance(restored, Adam)
    assert restored.learning_rate == 0.01
    assert restored.get_config()["beta_1"] == 0.8

    assert isinstance(get("sgd"), SGD)
    assert get(optimizer) is optimizer
    assert get(None) is None

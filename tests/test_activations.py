from __future__ import annotations

import numpy as np

import seq2seq.activations as activations
import seq2seq.layers as layers


def test_common_activation_aliases_are_available():
    expected = [
        "elu",
        "exponential",
        "gelu",
        "hard_sigmoid",
        "hard_silu",
        "leaky_relu",
        "log_softmax",
        "mish",
        "relu6",
        "selu",
        "silu",
        "softplus",
        "softsign",
        "swish",
    ]
    for name in expected:
        assert activations.get(name) is not None


def test_extended_activations_match_reference_values():
    x = np.array([[-2.0, 0.0, 2.0]], dtype=np.float32)

    np.testing.assert_allclose(
        activations.leaky_relu(x, negative_slope=0.1),
        np.array([[-0.2, 0.0, 2.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        activations.relu6(np.array([[-1.0, 3.0, 8.0]], dtype=np.float32)),
        np.array([[0.0, 3.0, 6.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(activations.swish(x), activations.silu(x))

    log_probs = activations.log_softmax(x, axis=-1)
    np.testing.assert_allclose(np.exp(log_probs).sum(axis=-1), np.ones(1), rtol=1e-6)


def test_activation_layers_wrap_extended_functions():
    x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)

    np.testing.assert_allclose(
        layers.LeakyReLU(negative_slope=0.5)(x),
        np.array([[-0.5, 0.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(layers.Softplus()(x), activations.softplus(x))
    assert layers.GELU()(x).shape == x.shape

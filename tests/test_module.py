
from __future__ import annotations

import numpy as np

import seq2seq.layers as layers
import seq2seq.models as models


def test_parameter_auto_registration_and_state_dict():
    class Toy(models.Model):
        def __init__(self):
            super().__init__()
            self.fc1 = layers.Dense(4, input_dim=3, seed=0)
            self.fc2 = layers.Dense(2, input_dim=4, seed=1)

        def call(self, x):
            return self.fc2(self.fc1(x))

    toy = Toy()
    names = sorted(name for name, _ in toy.named_weights())
    assert names == ["fc1.bias", "fc1.kernel", "fc2.bias", "fc2.kernel"]

    state = toy.state_dict()
    assert set(state.keys()) == set(names)

    twin = Toy()
    twin.load_state_dict(state)
    x = np.random.default_rng(0).standard_normal((1, 3)).astype(np.float32)
    np.testing.assert_allclose(toy(x), twin(x))


def test_train_eval_toggle_is_recursive():
    class Sub(models.Model):
        def __init__(self):
            super().__init__()
            self.fc = layers.Dense(2, input_dim=2, seed=0)

        def call(self, x):
            return self.fc(x)

    class Parent(models.Model):
        def __init__(self):
            super().__init__()
            self.a = Sub()
            self.b = Sub()

        def call(self, x):
            return self.b(self.a(x))

    parent = Parent()
    parent.eval()
    assert parent.training is False
    assert parent.a.training is False
    assert parent.b.fc.training is False

    parent.train()
    assert parent.a.fc.training is True


def test_total_parameters_counts_every_leaf():
    seq = models.Sequential(
        [layers.Dense(8, input_dim=10, seed=0), layers.ReLU(), layers.Dense(4, input_dim=8, seed=1)]
    )
    expected = 10 * 8 + 8 + 8 * 4 + 4
    assert seq.count_params() == expected


def test_compile_stores_optimizer_loss_and_metrics():
    seq = models.Sequential([layers.Dense(4, input_dim=3, seed=0)])
    seq.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    assert seq.compiled is True
    assert seq.optimizer is not None
    assert seq.loss is not None
    assert seq.metrics == ["accuracy"]

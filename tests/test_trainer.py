from __future__ import annotations

import numpy as np

import seq2seq.layers as layers
import seq2seq.models as models
from seq2seq.trainers import History


def test_fit_returns_keras_like_history_and_updates_autograd_model():
    model = models.Sequential(
        [layers.Dense(2, input_dim=2, activation="softmax", seed=0)]
    )
    model.compile(
        optimizer="sgd",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    x = np.array([[1, 0], [0, 1]], dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)
    before = model[0].kernel.data.copy()

    history = model.fit(x, y, epochs=2, batch_size=2, verbose=0)

    assert isinstance(history, History)
    assert history.history is history
    assert set(history) == {"loss", "accuracy"}
    assert len(history["loss"]) == 2
    assert not np.array_equal(before, model[0].kernel.data)


def test_predict_and_evaluate_preserve_training_mode():
    model = models.Sequential(
        [layers.Dense(2, input_dim=2, activation="softmax", seed=0)]
    )
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    x = np.array([[1, 0], [0, 1]], dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)

    model.eval()
    _ = model.predict(x, verbose=0)
    assert model.training is False

    result = model.evaluate(x, y, verbose=0)
    assert model.training is False
    assert set(result) == {"loss", "accuracy"}

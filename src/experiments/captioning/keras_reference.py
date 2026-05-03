from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from seq2seq.saving import (
    load_dense,
    load_embedding,
    load_lstm_cell,
    load_simple_rnn_cell,
)

from .decoder import CaptioningDecoder


@dataclass(frozen=True)
class ComparisonResult:
    scratch_output: np.ndarray
    keras_output: np.ndarray
    max_abs_error: float
    mean_abs_error: float
    allclose: bool


def build_keras_captioning_decoder(
    feature_dim: int,
    vocab_size: int,
    embed_dim: int,
    hidden_size: int,
    *,
    rnn_type: str = "lstm",
    num_layers: int = 1,
    seed: int | None = None,
    mask_zero: bool = False,
    name: str | None = None,
):
    if rnn_type not in {"rnn", "lstm"}:
        raise ValueError("rnn_type must be 'rnn' or 'lstm'")

    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    feature_input = tf.keras.Input(shape=(feature_dim,), name="image_features")
    token_input = tf.keras.Input(shape=(None,), dtype="int32", name="token_ids")

    image_projection = tf.keras.layers.Dense(embed_dim, name="image_projection")
    embedding = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=mask_zero, name="embedding")

    image_step = tf.keras.layers.Reshape((1, embed_dim), name="image_step")(image_projection(feature_input))
    token_steps = embedding(token_input)
    x = tf.keras.layers.Concatenate(axis=1, name="decoder_inputs")([image_step, token_steps])

    recurrent_cls = tf.keras.layers.LSTM if rnn_type == "lstm" else tf.keras.layers.SimpleRNN
    for idx in range(num_layers):
        x = recurrent_cls(hidden_size, return_sequences=True, name=f"recurrent_{idx}")(x)

    output = tf.keras.layers.Dense(vocab_size, activation="softmax", name="output")(x)
    return tf.keras.Model(
        [feature_input, token_input],
        output,
        name=name or f"{rnn_type}_captioning_reference",
    )


def load_captioning_decoder_from_keras(decoder: CaptioningDecoder, keras_model) -> CaptioningDecoder:
    load_dense(decoder.image_projection, keras_model.get_layer("image_projection").get_weights())
    load_embedding(decoder.embedding, keras_model.get_layer("embedding").get_weights())

    for idx, cell in enumerate(decoder.recurrent.cells):
        weights = keras_model.get_layer(f"recurrent_{idx}").get_weights()
        if decoder.rnn_type == "lstm":
            load_lstm_cell(cell, weights)
        else:
            load_simple_rnn_cell(cell, weights)

    load_dense(decoder.output, keras_model.get_layer("output").get_weights())
    return decoder


def compare_captioning_decoder_outputs(
    decoder: CaptioningDecoder,
    keras_model,
    features: np.ndarray,
    token_ids: np.ndarray,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> ComparisonResult:
    features = np.asarray(features, dtype=np.float32)
    token_ids = np.asarray(token_ids, dtype=np.int64)

    scratch_output = np.asarray(decoder(features, token_ids), dtype=np.float32)
    keras_output = np.asarray(keras_model.predict([features, token_ids], verbose=0), dtype=np.float32)
    diff = np.abs(scratch_output - keras_output)

    return ComparisonResult(
        scratch_output=scratch_output,
        keras_output=keras_output,
        max_abs_error=float(diff.max(initial=0.0)),
        mean_abs_error=float(diff.mean()),
        allclose=bool(np.allclose(scratch_output, keras_output, atol=atol, rtol=rtol)),
    )

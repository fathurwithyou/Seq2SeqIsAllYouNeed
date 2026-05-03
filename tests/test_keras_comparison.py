from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from experiments.captioning import (
    CaptioningDecoder,
    build_keras_captioning_decoder,
    compare_captioning_decoder_outputs,
    load_captioning_decoder_from_keras,
)


@pytest.mark.parametrize("rnn_type", ["rnn", "lstm"])
def test_captioning_decoder_matches_keras_reference_after_weight_transfer(rnn_type: str):
    feature_dim = 8
    vocab_size = 13
    embed_dim = 6
    hidden_size = 5
    num_layers = 2
    seed = 7

    keras_model = build_keras_captioning_decoder(
        feature_dim,
        vocab_size,
        embed_dim,
        hidden_size,
        rnn_type=rnn_type,
        num_layers=num_layers,
        seed=seed,
    )
    decoder = CaptioningDecoder(
        feature_dim,
        vocab_size,
        embed_dim,
        hidden_size,
        rnn_type=rnn_type,
        num_layers=num_layers,
        seed=0,
    )
    load_captioning_decoder_from_keras(decoder, keras_model)

    rng = np.random.default_rng(3)
    features = rng.standard_normal((3, feature_dim)).astype(np.float32)
    token_ids = np.array(
        [
            [1, 4, 5, 2, 0],
            [1, 3, 2, 0, 0],
            [1, 6, 7, 8, 2],
        ],
        dtype=np.int64,
    )

    result = compare_captioning_decoder_outputs(decoder, keras_model, features, token_ids)

    assert result.allclose
    assert result.max_abs_error < 1e-5
    assert result.mean_abs_error < 1e-6

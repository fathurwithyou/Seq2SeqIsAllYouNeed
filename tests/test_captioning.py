
from __future__ import annotations

import numpy as np

from seq2seq.utils import (
    END_TOKEN,
    PAD_TOKEN,
    START_TOKEN,
    Vocabulary,
    pad_sequences,
    tokenize,
)
from experiments.captioning import CaptioningDecoder, beam_search_decode


def _toy_vocab() -> Vocabulary:
    captions = [
        "a dog is running on the grass",
        "two children play with a ball",
        "the cat sits on a mat",
    ]
    return Vocabulary.build(captions, min_count=1)


def test_vocab_round_trip_and_specials():
    vocab = _toy_vocab()
    assert vocab.token_to_id[PAD_TOKEN] == 0
    assert vocab.token_to_id[START_TOKEN] == 1
    assert vocab.token_to_id[END_TOKEN] == 2

    ids = vocab.encode("a dog is running")
    decoded = vocab.decode(ids)
    assert decoded == "a dog is running"


def test_pad_sequences_post_padding_and_truncation():
    out = pad_sequences([[1, 2, 3], [4, 5], [6]], maxlen=4, value=0, padding="post")
    np.testing.assert_array_equal(
        out,
        np.array([[1, 2, 3, 0], [4, 5, 0, 0], [6, 0, 0, 0]]),
    )

    truncated = pad_sequences([[1, 2, 3, 4, 5]], maxlen=3, padding="post", truncating="post")
    np.testing.assert_array_equal(truncated, np.array([[1, 2, 3]]))


def test_pad_sequences_default_matches_keras_pre_padding():
    out = pad_sequences([[1], [2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(
        out,
        np.array([[0, 0, 1], [0, 2, 3], [4, 5, 6]], dtype=np.int32),
    )


def test_decoder_forward_shape_and_greedy_decoding_terminates():
    vocab = _toy_vocab()
    decoder = CaptioningDecoder(
        feature_dim=8,
        vocab_size=vocab.size,
        embed_dim=6,
        hidden_size=4,
        rnn_type="lstm",
        num_layers=1,
        seed=0,
    )

    rng = np.random.default_rng(0)
    features = rng.standard_normal((3, 8)).astype(np.float32)
    captions = pad_sequences(
        [vocab.encode("a dog is running"), vocab.encode("two children"), vocab.encode("the cat")],
        maxlen=10,
    )
    probs = decoder(features, captions)

    assert probs.shape == (3, captions.shape[1] + 1, vocab.size)

    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-5)

    sequences = decoder.greedy_decode(features, start_id=vocab.start_id, end_id=vocab.end_id, max_length=8)
    assert all(seq[0] == vocab.start_id for seq in sequences)
    assert all(len(seq) <= 9 for seq in sequences)


def test_rnn_decoder_works_just_like_lstm_decoder():
    vocab = _toy_vocab()
    dec = CaptioningDecoder(8, vocab.size, 6, 4, rnn_type="rnn", num_layers=2, seed=0)
    feat = np.random.default_rng(1).standard_normal((1, 8)).astype(np.float32)
    out = dec.greedy_decode(feat, start_id=vocab.start_id, end_id=vocab.end_id, max_length=5)
    assert isinstance(out, list) and len(out[0]) >= 1


def test_beam_search_returns_a_sequence():
    vocab = _toy_vocab()
    dec = CaptioningDecoder(8, vocab.size, 6, 4, rnn_type="lstm", num_layers=1, seed=0)
    feat = np.random.default_rng(2).standard_normal((8,)).astype(np.float32)
    seq = beam_search_decode(
        dec, feat, start_id=vocab.start_id, end_id=vocab.end_id, beam_width=3, max_length=6
    )
    assert isinstance(seq, list) and len(seq) >= 1


def test_beam_search_accepts_beam_width_at_vocab_size():
    vocab = _toy_vocab()
    dec = CaptioningDecoder(8, vocab.size, 6, 4, rnn_type="lstm", num_layers=1, seed=0)
    feat = np.random.default_rng(2).standard_normal((8,)).astype(np.float32)
    seq = beam_search_decode(
        dec,
        feat,
        start_id=vocab.start_id,
        end_id=vocab.end_id,
        beam_width=vocab.size,
        max_length=2,
    )
    assert isinstance(seq, list) and len(seq) >= 1


def test_tokenize_strips_punctuation_and_lowercases():
    assert tokenize("A Dog, running!") == ["a", "dog", "running"]

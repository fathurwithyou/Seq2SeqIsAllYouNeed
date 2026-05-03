from __future__ import annotations

import numpy as np

from seq2seq.metrics import (
    accuracy,
    bleu_score,
    confusion_matrix,
    corpus_bleu,
    f1_per_class,
    macro_f1,
    meteor_score,
)


def test_accuracy_argmax_and_labels():
    y_true = [0, 1, 2, 1]
    logits = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.0, 0.2, 0.8],
            [0.4, 0.4, 0.2],
        ]
    )
    assert accuracy(y_true, logits) == 0.75


def test_confusion_matrix_shape_and_values():
    y_true = [0, 1, 2, 2, 1]
    y_pred = [0, 2, 2, 2, 1]
    matrix = confusion_matrix(y_true, y_pred, num_classes=3)
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == 1
    assert matrix[1, 2] == 1
    assert matrix[2, 2] == 2


def test_macro_f1_balanced():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    scores = f1_per_class(y_true, y_pred, num_classes=3)
    assert scores.shape == (3,)
    np.testing.assert_allclose(scores[1], 2 * (2 / 3 * 1.0) / (2 / 3 + 1.0), atol=1e-6)
    value = macro_f1(y_true, y_pred, num_classes=3)
    assert 0.0 <= value <= 1.0


def test_bleu_identical_hypothesis_scores_one():
    tokens = "a man is riding a bicycle on a road".split()
    assert abs(bleu_score(tokens, [tokens]) - 1.0) < 1e-6


def test_bleu_fully_different_is_low():
    hyp = "a cat is jumping over the chair".split()
    ref = "the bicycle is leaning on the wall".split()
    assert bleu_score(hyp, [ref]) < 0.2


def test_corpus_bleu_consistent_with_single_pair():
    tokens = "a red car is parked by the street".split()
    single = bleu_score(tokens, [tokens])
    corpus = corpus_bleu([tokens], [[tokens]])
    assert abs(single - corpus) < 1e-6


def test_meteor_identical_hypothesis_scores_near_one():
    tokens = "a dog is running on grass".split()
    assert meteor_score(tokens, [tokens]) > 0.99


def test_meteor_picks_best_reference():
    hyp = "a man is riding a bike".split()
    ref_good = "a man is riding a bike".split()
    ref_bad = "completely unrelated sentence".split()
    assert meteor_score(hyp, [ref_bad, ref_good]) > 0.99


def test_meteor_nonzero_for_partial_match():
    hyp = "a dog runs fast".split()
    ref = "a dog is running fast".split()
    score = meteor_score(hyp, [ref])
    assert 0.2 < score < 1.0

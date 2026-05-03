from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np

from experiments.captioning import build_keras_captioning_decoder, load_features
from seq2seq.utils import Vocabulary
from seq2seq.utils.sequence_utils import pad_sequences


def _build_training_pairs(
    captions: dict[str, list[list[str]]],
    feature_lookup: dict[str, np.ndarray],
    vocab: Vocabulary,
    *,
    max_length: int,
):
    features: list[np.ndarray] = []
    input_tokens: list[list[int]] = []
    target_tokens: list[list[int]] = []
    for image, token_lists in captions.items():
        feature = feature_lookup.get(image)
        if feature is None:
            continue
        for tokens in token_lists:
            ids = [vocab.start_id] + [vocab.token_to_id.get(t, vocab.unk_id) for t in tokens] + [vocab.end_id]
            ids = ids[: max_length + 1]
            input_ids = ids[:-1]
            target_ids = ids[1:]
            features.append(feature)
            input_tokens.append(input_ids)
            target_tokens.append(target_ids)
    if not features:
        raise RuntimeError("no (feature, caption) pairs constructed — check paths and captions")
    features_arr = np.stack(features, axis=0).astype(np.float32)
    inputs_arr = pad_sequences(input_tokens, maxlen=max_length, value=vocab.pad_id, padding="post")
    target_length = max_length + 1
    targets_arr = pad_sequences(target_tokens, maxlen=target_length, value=vocab.pad_id, padding="post")
    return features_arr, inputs_arr, targets_arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--paths-file", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--captions-json", required=True)
    parser.add_argument("--output-dir", default="artifacts/captioning")
    parser.add_argument("--rnn-type", choices=("rnn", "lstm"), default="lstm")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import tensorflow as tf

    if not 0 <= args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be in the range [0, 1).")
    tf.keras.utils.set_random_seed(args.seed)

    vocab = Vocabulary.load(args.vocab)
    captions = json.loads(Path(args.captions_json).read_text(encoding="utf-8"))
    feature_lookup = load_features(Path(args.features), Path(args.paths_file))

    features, inputs, targets = _build_training_pairs(
        captions, feature_lookup, vocab, max_length=args.max_length
    )
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(features))
    val_size = int(len(indices) * args.validation_fraction)
    if val_size > 0:
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        train_data = ([features[train_indices], inputs[train_indices]], targets[train_indices])
        validation_data = ([features[val_indices], inputs[val_indices]], targets[val_indices])
    else:
        train_data = ([features, inputs], targets)
        validation_data = None

    model = build_keras_captioning_decoder(
        feature_dim=features.shape[1],
        vocab_size=vocab.size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        rnn_type=args.rnn_type,
        num_layers=args.num_layers,
        seed=args.seed,
        mask_zero=True,
        name=f"{args.rnn_type}_captioning",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
    )
    history = model.fit(
        train_data[0],
        train_data[1],
        validation_data=validation_data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=2,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.rnn_type}_L{args.num_layers}_H{args.hidden_size}"
    weights_path = output_dir / f"{tag}.weights.h5"
    model.save_weights(str(weights_path))
    with (output_dir / f"{tag}.history.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": vars(args),
                "history": {k: [float(v) for v in values] for k, values in history.history.items()},
                "feature_dim": int(features.shape[1]),
                "vocab_size": vocab.size,
                "num_train_examples": int(len(train_data[1])),
                "num_validation_examples": int(len(validation_data[1]) if validation_data else 0),
            },
            handle,
            indent=2,
        )
    print(f"Saved weights to {weights_path}")


if __name__ == "__main__":
    main()

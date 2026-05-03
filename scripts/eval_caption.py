from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np

from experiments.captioning import (
    CaptioningDecoder,
    beam_search_decode,
    build_keras_captioning_decoder,
    load_captioning_decoder_from_keras,
    load_features,
)
from seq2seq.metrics import corpus_bleu, meteor_score
from seq2seq.utils import Vocabulary


def _load_test_split(split_file: Path | None, captions: dict[str, list[list[str]]]) -> list[str]:
    if split_file is None:
        return list(captions.keys())
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _keras_greedy_decode(
    keras_model,
    feature: np.ndarray,
    *,
    start_id: int,
    end_id: int,
    max_length: int,
) -> list[int]:
    sequence = [start_id]
    for _ in range(max_length):
        token_ids = np.array([sequence], dtype=np.int64)
        probs = keras_model.predict([feature[None, :], token_ids], verbose=0)[0, -1]
        next_id = int(np.argmax(probs))
        sequence.append(next_id)
        if next_id == end_id:
            break
    return sequence


def _score_outputs(
    hypotheses: list[list[str]],
    references: list[list[list[str]]],
) -> tuple[float, float]:
    bleu = corpus_bleu(hypotheses, references)
    meteor_values = [
        meteor_score(hyp, refs) for hyp, refs in zip(hypotheses, references)
    ]
    meteor = float(np.mean(meteor_values)) if meteor_values else 0.0
    return bleu, meteor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Keras weights (.weights.h5)")
    parser.add_argument("--config", required=True, help="train_caption history JSON with config")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--captions-json", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--paths-file", required=True)
    parser.add_argument("--test-split", default=None, help="Optional file listing test image filenames")
    parser.add_argument("--mode", choices=("scratch", "keras", "both"), default="scratch")
    parser.add_argument("--decoding", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=None, help="Override max caption length from config")
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--output", default="artifacts/captioning/eval.json")
    args = parser.parse_args()

    import tensorflow as tf

    vocab = Vocabulary.load(args.vocab)
    captions = json.loads(Path(args.captions_json).read_text(encoding="utf-8"))
    feature_lookup = load_features(Path(args.features), Path(args.paths_file))
    config_payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = config_payload["config"]

    test_images = _load_test_split(
        Path(args.test_split) if args.test_split else None,
        captions,
    )

    keras_model = tf.keras.models.load_model(args.weights, compile=False) if args.weights.endswith(".keras") else None
    if keras_model is None:
        keras_model = build_keras_captioning_decoder(
            feature_dim=config_payload["feature_dim"],
            vocab_size=config_payload["vocab_size"],
            embed_dim=config["embed_dim"],
            hidden_size=config["hidden_size"],
            rnn_type=config["rnn_type"],
            num_layers=config["num_layers"],
            seed=config["seed"],
            mask_zero=True,
            name=f"{config['rnn_type']}_captioning",
        )
        keras_model.load_weights(args.weights)

    decoder = CaptioningDecoder(
        feature_dim=config_payload["feature_dim"],
        vocab_size=config_payload["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_size=config["hidden_size"],
        rnn_type=config["rnn_type"],
        num_layers=config["num_layers"],
        seed=config["seed"],
    )
    load_captioning_decoder_from_keras(decoder, keras_model)

    max_length = int(args.max_length or config["max_length"])
    run_scratch = args.mode in {"scratch", "both"}
    run_keras = args.mode in {"keras", "both"}
    if args.decoding == "beam" and run_keras:
        raise ValueError("Keras comparison mode currently supports greedy decoding only.")

    scratch_hypotheses: list[list[str]] = []
    keras_hypotheses: list[list[str]] = []
    references: list[list[list[str]]] = []
    qualitative: list[dict] = []
    scratch_seconds = 0.0
    keras_seconds = 0.0
    for image in test_images:
        feature = feature_lookup.get(image)
        if feature is None:
            continue
        refs = captions.get(image, [])
        references.append([list(ref) for ref in refs])

        scratch_text = None
        keras_text = None
        if run_scratch:
            started = time.perf_counter()
            if args.decoding == "greedy":
                token_ids = decoder.greedy_decode(
                    feature[None, :],
                    start_id=vocab.start_id,
                    end_id=vocab.end_id,
                    max_length=max_length,
                )[0]
            else:
                token_ids = beam_search_decode(
                    decoder,
                    feature,
                    start_id=vocab.start_id,
                    end_id=vocab.end_id,
                    beam_width=args.beam_width,
                    max_length=max_length,
                )
            scratch_seconds += time.perf_counter() - started
            scratch_hypothesis = vocab.decode(token_ids).split()
            scratch_hypotheses.append(scratch_hypothesis)
            scratch_text = " ".join(scratch_hypothesis)
        if run_keras:
            started = time.perf_counter()
            token_ids = _keras_greedy_decode(
                keras_model,
                feature,
                start_id=vocab.start_id,
                end_id=vocab.end_id,
                max_length=max_length,
            )
            keras_seconds += time.perf_counter() - started
            keras_hypothesis = vocab.decode(token_ids).split()
            keras_hypotheses.append(keras_hypothesis)
            keras_text = " ".join(keras_hypothesis)

        if len(qualitative) < args.num_examples:
            qualitative.append(
                {
                    "image": image,
                    "scratch_hypothesis": scratch_text,
                    "keras_hypothesis": keras_text,
                    "references": [" ".join(ref) for ref in refs],
                }
            )

    results = {}
    if run_scratch:
        bleu, meteor = _score_outputs(scratch_hypotheses, references)
        results["scratch"] = {
            "bleu_4": bleu,
            "meteor": meteor,
            "seconds": scratch_seconds,
            "seconds_per_image": scratch_seconds / max(len(scratch_hypotheses), 1),
        }
    if run_keras:
        bleu, meteor = _score_outputs(keras_hypotheses, references)
        results["keras"] = {
            "bleu_4": bleu,
            "meteor": meteor,
            "seconds": keras_seconds,
            "seconds_per_image": keras_seconds / max(len(keras_hypotheses), 1),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "results": results,
                "num_evaluated": len(references),
                "qualitative": qualitative,
                "config": config,
                "max_length": max_length,
                "decoding": args.decoding,
                "mode": args.mode,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    for name, metrics in results.items():
        print(f"{name} BLEU-4:  {metrics['bleu_4']:.4f}")
        print(f"{name} METEOR:  {metrics['meteor']:.4f}")
        print(f"{name} seconds/image: {metrics['seconds_per_image']:.4f}")
    for example in qualitative:
        print("---")
        print(f"image: {example['image']}")
        if example["scratch_hypothesis"] is not None:
            print(f"scratch: {example['scratch_hypothesis']}")
        if example["keras_hypothesis"] is not None:
            print(f"keras:   {example['keras_hypothesis']}")
        for ref in example["references"]:
            print(f"ref:   {ref}")


if __name__ == "__main__":
    main()

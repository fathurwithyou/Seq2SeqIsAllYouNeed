from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np

from experiments.cnn import build_keras_cnn
from seq2seq.datasets import download_intel_image_dataset, load_intel_image_dataset
from seq2seq.layers import (
    AveragePooling2D,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    LocallyConnected2D,
    MaxPooling2D,
)
from seq2seq.metrics import confusion_matrix, macro_f1
from seq2seq.models import Sequential
from seq2seq.saving import load_conv2d, load_dense, load_locally_connected2d


def _pooling_layer(kind: str):
    if kind == "max":
        return MaxPooling2D(pool_size=2)
    if kind == "average":
        return AveragePooling2D(pool_size=2)
    raise ValueError(f"Unknown pooling: {kind!r}")


def _build_scratch(config: dict, input_shape: tuple[int, int, int], num_classes: int) -> Sequential:
    model = Sequential()
    filters = config["filters"]
    if len(filters) == 1:
        filters = filters * config["num_conv_layers"]
    current_shape = input_shape
    for index in range(config["num_conv_layers"]):
        if config["layer_type"] == "conv2d":
            layer = Conv2D(
                filters[index],
                kernel_size=config["kernel_size"],
                padding="same",
                activation="relu",
                input_shape=current_shape,
                name=f"conv2d_{index}",
            )
        else:
            layer = LocallyConnected2D(
                filters[index],
                kernel_size=config["kernel_size"],
                padding="valid",
                activation="relu",
                input_shape=current_shape,
                name=f"locally_connected_{index}",
            )
        pooling_layer = _pooling_layer(config["pooling"])
        model.add(layer)
        model.add(pooling_layer)
        dummy = np.zeros((1, *current_shape), dtype=np.float32)
        current_shape = pooling_layer(layer(dummy)).shape[1:]
    model.add(GlobalAveragePooling2D())
    model.add(Dense(config["dense_units"], activation="relu", input_dim=filters[-1], name="head"))
    model.add(Dense(num_classes, activation="softmax", input_dim=config["dense_units"], name="classifier"))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to Intel Image dataset root. Defaults to KaggleHub download.",
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default=None, help="Optional JSON output path for saved metrics.")
    args = parser.parse_args()

    import tensorflow as tf

    with Path(args.config).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload["config"]
    target_size = tuple(config["target_size"])
    data_root = Path(args.data_root) if args.data_root else download_intel_image_dataset()
    splits = load_intel_image_dataset(
        data_root,
        validation_fraction=config["validation_fraction"],
        seed=config["seed"],
    )
    test_images, test_labels = splits["test"].to_arrays(target_size=target_size)

    if not Path(args.weights).is_file():
        raise FileNotFoundError(f"Could not find Keras weights at {args.weights}")

    if args.weights.endswith(".keras"):
        keras_model = tf.keras.models.load_model(args.weights, compile=False)
    else:
        keras_model = build_keras_cnn(
            input_shape=(*target_size, 3),
            num_classes=splits["test"].num_classes,
            layer_type=config["layer_type"],
            num_conv_layers=config["num_conv_layers"],
            filters=config["filters"],
            kernel_size=config["kernel_size"],
            pooling=config["pooling"],
            dense_units=config["dense_units"],
        )
        keras_model.load_weights(args.weights)

    scratch = _build_scratch(config, input_shape=(*target_size, 3), num_classes=splits["test"].num_classes)
    scratch(np.zeros((1, *target_size, 3), dtype=np.float32))

    for scratch_layer in scratch.layers:
        if isinstance(scratch_layer, Conv2D):
            load_conv2d(scratch_layer, keras_model.get_layer(scratch_layer.name).get_weights())
        elif isinstance(scratch_layer, LocallyConnected2D):
            load_locally_connected2d(scratch_layer, keras_model.get_layer(scratch_layer.name).get_weights())
        elif isinstance(scratch_layer, Dense):
            load_dense(scratch_layer, keras_model.get_layer(scratch_layer.name).get_weights())

    scratch_probs: list[np.ndarray] = []
    for start in range(0, len(test_images), args.batch_size):
        end = start + args.batch_size
        scratch_probs.append(scratch(test_images[start:end].astype(np.float32)))
    scratch_probs = np.concatenate(scratch_probs, axis=0)
    scratch_pred = scratch_probs.argmax(axis=-1)
    scratch_f1 = macro_f1(test_labels, scratch_pred, num_classes=splits["test"].num_classes)

    keras_pred = keras_model.predict(test_images, batch_size=args.batch_size, verbose=0).argmax(axis=-1)
    keras_f1 = macro_f1(test_labels, keras_pred, num_classes=splits["test"].num_classes)

    agreement = float(np.mean(scratch_pred == keras_pred))
    results = {
        "kind": "cnn_scratch_eval",
        "weights": args.weights,
        "config_path": args.config,
        "config": config,
        "keras_macro_f1": keras_f1,
        "scratch_macro_f1": scratch_f1,
        "prediction_agreement": agreement,
        "scratch_confusion_matrix": confusion_matrix(
            test_labels,
            scratch_pred,
            num_classes=splits["test"].num_classes,
        ).tolist(),
        "class_names": list(splits["test"].class_names),
        "num_evaluated": int(len(test_labels)),
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    print(f"Keras macro-F1:   {keras_f1:.4f}")
    print(f"Scratch macro-F1: {scratch_f1:.4f}")
    print(f"Prediction agreement: {agreement:.4f}")
    print("Confusion matrix (scratch):")
    print(np.array(results["scratch_confusion_matrix"]))
    if args.output:
        print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()

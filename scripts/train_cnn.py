from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np

from experiments.cnn import build_keras_cnn
from seq2seq.datasets import download_intel_image_dataset, load_intel_image_dataset
from seq2seq.metrics import confusion_matrix, macro_f1


def _materialise(dataset, target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    images, labels = dataset.to_arrays(target_size=target_size, normalize=True)
    return images.astype(np.float32), labels.astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to Intel Image dataset root. Defaults to KaggleHub download.",
    )
    parser.add_argument("--output-dir", default="artifacts/cnn")
    parser.add_argument("--target-size", type=int, nargs=2, default=(150, 150))
    parser.add_argument("--layer-type", choices=("conv2d", "locally_connected"), default="conv2d")
    parser.add_argument("--num-conv-layers", type=int, default=3)
    parser.add_argument("--filters", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--pooling", choices=("max", "average"), default="max")
    parser.add_argument("--dense-units", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)

    data_root = Path(args.data_root) if args.data_root else download_intel_image_dataset()
    splits = load_intel_image_dataset(
        data_root,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    train_x, train_y = _materialise(splits["train"], tuple(args.target_size))
    val_x, val_y = _materialise(splits["validation"], tuple(args.target_size))
    test_x, test_y = _materialise(splits["test"], tuple(args.target_size))

    num_classes = splits["train"].num_classes
    model = build_keras_cnn(
        input_shape=(*args.target_size, 3),
        num_classes=num_classes,
        layer_type=args.layer_type,
        num_conv_layers=args.num_conv_layers,
        filters=args.filters,
        kernel_size=args.kernel_size,
        pooling=args.pooling,
        dense_units=args.dense_units,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=2,
    )

    test_probs = model.predict(test_x, batch_size=args.batch_size, verbose=0)
    test_pred = test_probs.argmax(axis=-1)
    test_f1 = macro_f1(test_y, test_pred, num_classes=num_classes)
    matrix = confusion_matrix(test_y, test_pred, num_classes=num_classes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.layer_type}_{args.num_conv_layers}x{'-'.join(str(f) for f in args.filters)}_k{args.kernel_size}_{args.pooling}"
    weights_path = output_dir / f"{tag}.weights.h5"
    model.save_weights(str(weights_path))
    with (output_dir / f"{tag}.history.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": vars(args),
                "history": {k: [float(v) for v in values] for k, values in history.history.items()},
                "test_macro_f1": test_f1,
                "test_confusion_matrix": matrix.tolist(),
                "class_names": list(splits["train"].class_names),
            },
            handle,
            indent=2,
        )
    print(f"Saved weights to {weights_path}")
    print(f"Test macro-F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()

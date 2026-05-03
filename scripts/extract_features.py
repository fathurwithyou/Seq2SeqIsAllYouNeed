from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _collect_image_paths(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)


def _get_encoder(name: str):
    import tensorflow as tf

    name = name.lower()
    if name == "inception_v3":
        base = tf.keras.applications.InceptionV3(include_top=False, pooling="avg", weights="imagenet")
        preprocess = tf.keras.applications.inception_v3.preprocess_input
        target_size = (299, 299)
    elif name == "mobilenet_v2":
        base = tf.keras.applications.MobileNetV2(include_top=False, pooling="avg", weights="imagenet")
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        target_size = (224, 224)
    elif name == "vgg16":
        base = tf.keras.applications.VGG16(include_top=False, pooling="avg", weights="imagenet")
        preprocess = tf.keras.applications.vgg16.preprocess_input
        target_size = (224, 224)
    else:
        raise ValueError(f"Unknown encoder: {name!r}")
    base.trainable = False
    return base, preprocess, target_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", default=None, help="Path to Flickr8k Images directory.")
    parser.add_argument("--flickr-root", default=None, help="Path to Flickr8k root. Defaults to KaggleHub download.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--paths-output", required=True)
    parser.add_argument("--encoder", choices=("inception_v3", "mobilenet_v2", "vgg16"), default="inception_v3")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=None,
        const=None,
        help="Optional maximum number of images to process.",
    )
    args = parser.parse_args()

    from experiments.captioning import download_flickr8k_dataset
    from seq2seq.utils.image_utils import extract_features_with_keras

    flickr_root = Path(args.flickr_root) if args.flickr_root else None
    if args.images_dir:
        images_dir = Path(args.images_dir)
    else:
        flickr_root = flickr_root or download_flickr8k_dataset()
        images_dir = flickr_root / "Images"
    paths = _collect_image_paths(images_dir)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No images found under {images_dir}")

    encoder, preprocess, target_size = _get_encoder(args.encoder)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features = extract_features_with_keras(
        paths,
        encoder,
        target_size=target_size,
        batch_size=args.batch_size,
        preprocess=preprocess,
        save_path=output_path,
    )
    with Path(args.paths_output).open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{path.relative_to(images_dir).as_posix()}\n")
    print(f"Saved {features.shape[0]} features of dim {features.shape[1]} to {output_path}")


if __name__ == "__main__":
    main()

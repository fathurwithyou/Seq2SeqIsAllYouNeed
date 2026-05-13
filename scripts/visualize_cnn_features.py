from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np

from experiments.cnn import (
    build_keras_cnn,
    compute_grad_cam,
    feature_layer_names,
    make_feature_grid,
    overlay_heatmap,
    predict_feature_maps,
    save_array_image,
)
from seq2seq.datasets import download_intel_image_dataset, load_intel_image_dataset
from seq2seq.utils import load_image


def _load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def _load_visualisation_image(args, config: dict) -> tuple[np.ndarray, dict]:
    target_size = tuple(config["target_size"])
    if args.image:
        path = Path(args.image)
        image = load_image(path, target_size=target_size, normalize=True)
        return image, {"source": str(path), "label": None, "label_name": None}

    data_root = Path(args.data_root) if args.data_root else download_intel_image_dataset()
    splits = load_intel_image_dataset(
        data_root,
        validation_fraction=config.get("validation_fraction", 0.1),
        seed=config.get("seed", 42),
    )
    split = splits[args.split]
    if len(split) == 0:
        raise ValueError(f"Split {args.split!r} is empty")
    if not 0 <= args.sample_index < len(split):
        raise IndexError(f"--sample-index must be in [0, {len(split) - 1}]")

    path = split.paths[args.sample_index]
    label = int(split.labels[args.sample_index])
    image = load_image(path, target_size=target_size, normalize=True)
    return image, {
        "source": str(path),
        "split": args.split,
        "sample_index": int(args.sample_index),
        "label": label,
        "label_name": split.class_names[label],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualise CNN intermediate feature maps and Grad-CAM from a trained "
            "Keras CNN artifact."
        )
    )
    parser.add_argument("--weights", required=True, help="Path to .weights.h5 CNN weights")
    parser.add_argument("--config", required=True, help="Path to matching .history.json")
    parser.add_argument("--output-dir", default="artifacts/cnn/visualizations")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--image", default=None, help="Optional explicit image path")
    parser.add_argument("--class-index", type=int, default=None, help="Target class for Grad-CAM")
    parser.add_argument("--gradcam-layer", default=None, help="Layer name for Grad-CAM")
    parser.add_argument("--max-feature-maps", type=int, default=16)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()

    payload = _load_config(args.config)
    config = payload["config"]
    class_names = payload.get("class_names")
    target_size = tuple(config["target_size"])

    model = build_keras_cnn(
        input_shape=(*target_size, 3),
        num_classes=len(class_names) if class_names else 6,
        layer_type=config["layer_type"],
        num_conv_layers=config["num_conv_layers"],
        filters=config["filters"],
        kernel_size=config["kernel_size"],
        pooling=config["pooling"],
        dense_units=config["dense_units"],
    )
    model.load_weights(args.weights)

    image, image_meta = _load_visualisation_image(args, config)
    image_batch = image[None, ...].astype(np.float32)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_names = feature_layer_names(model, layer_type=config.get("layer_type"))
    if not layer_names:
        raise ValueError("No convolution-like feature layers found in the model")

    feature_outputs = predict_feature_maps(model, image_batch, layer_names)
    feature_paths: dict[str, str] = {}
    for layer_name, maps in feature_outputs.items():
        grid = make_feature_grid(
            maps,
            max_channels=args.max_feature_maps,
            columns=args.columns,
        )
        path = output_dir / f"{layer_name}_feature_maps.png"
        save_array_image(path, grid, cmap="gray")
        feature_paths[layer_name] = str(path)

    gradcam_layer = args.gradcam_layer or layer_names[-1]
    heatmaps, predictions, class_indices = compute_grad_cam(
        model,
        image_batch,
        layer_name=gradcam_layer,
        class_index=args.class_index,
    )
    predicted_index = int(np.argmax(predictions[0]))
    target_index = int(class_indices[0])
    overlay = overlay_heatmap(image, heatmaps[0], alpha=args.alpha)

    heatmap_path = output_dir / f"{gradcam_layer}_gradcam_heatmap.png"
    overlay_path = output_dir / f"{gradcam_layer}_gradcam_overlay.png"
    input_path = output_dir / "input_image.png"
    save_array_image(input_path, image)
    save_array_image(heatmap_path, heatmaps[0], cmap="jet")
    save_array_image(overlay_path, overlay)

    manifest = {
        "weights": str(args.weights),
        "config": str(args.config),
        "image": image_meta,
        "layer_type": config["layer_type"],
        "feature_layers": layer_names,
        "feature_map_paths": feature_paths,
        "gradcam_layer": gradcam_layer,
        "input_image_path": str(input_path),
        "gradcam_heatmap_path": str(heatmap_path),
        "gradcam_overlay_path": str(overlay_path),
        "predicted_class": predicted_index,
        "predicted_class_name": class_names[predicted_index] if class_names else None,
        "target_class": target_index,
        "target_class_name": class_names[target_index] if class_names else None,
        "prediction_probabilities": [float(value) for value in predictions[0]],
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Feature layers: {', '.join(layer_names)}")
    print(f"Predicted class: {predicted_index} ({manifest['predicted_class_name']})")
    print(f"Grad-CAM target: {target_index} ({manifest['target_class_name']})")
    print(f"Saved visualisations to {output_dir}")


if __name__ == "__main__":
    main()

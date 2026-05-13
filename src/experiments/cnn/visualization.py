from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def feature_layer_names(model, *, layer_type: str | None = None) -> list[str]:
    """Return convolution-like layer names whose outputs are spatial feature maps."""
    prefixes: tuple[str, ...]
    if layer_type == "conv2d":
        prefixes = ("conv2d_",)
    elif layer_type == "locally_connected":
        prefixes = ("locally_connected_",)
    else:
        prefixes = ("conv2d_", "locally_connected_")

    names: list[str] = []
    for layer in model.layers:
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is None:
            try:
                output_shape = layer.output.shape
            except AttributeError:
                output_shape = None
        if layer.name.startswith(prefixes) and output_shape is not None and len(output_shape) == 4:
            names.append(layer.name)
    return names


def normalise_map(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    array = array - float(np.min(array))
    denom = float(np.max(array))
    if denom <= eps:
        return np.zeros_like(array, dtype=np.float32)
    return (array / denom).astype(np.float32, copy=False)


def make_feature_grid(
    feature_maps: np.ndarray,
    *,
    max_channels: int = 16,
    columns: int = 4,
    pad: int = 2,
) -> np.ndarray:
    maps = np.asarray(feature_maps, dtype=np.float32)
    if maps.ndim == 4:
        if maps.shape[0] != 1:
            raise ValueError("make_feature_grid expects a single example when input is rank 4")
        maps = maps[0]
    if maps.ndim != 3:
        raise ValueError(f"feature_maps must have shape (H,W,C) or (1,H,W,C); got {maps.shape}")
    if max_channels < 1:
        raise ValueError("max_channels must be positive")
    if columns < 1:
        raise ValueError("columns must be positive")

    height, width, channels = maps.shape
    count = min(channels, max_channels)
    rows = int(np.ceil(count / columns))
    grid = np.ones(
        (
            rows * height + (rows - 1) * pad,
            columns * width + (columns - 1) * pad,
        ),
        dtype=np.float32,
    )
    for index in range(count):
        row, col = divmod(index, columns)
        y0 = row * (height + pad)
        x0 = col * (width + pad)
        grid[y0 : y0 + height, x0 : x0 + width] = normalise_map(maps[:, :, index])
    return grid


def colourise_heatmap(heatmap: np.ndarray, cmap: str = "jet") -> np.ndarray:
    import matplotlib.pyplot as plt

    coloured = plt.get_cmap(cmap)(normalise_map(heatmap))[..., :3]
    return coloured.astype(np.float32, copy=False)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.45,
    cmap: str = "jet",
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim == 4:
        if image_array.shape[0] != 1:
            raise ValueError("overlay_heatmap expects a single image when input is rank 4")
        image_array = image_array[0]
    if image_array.ndim != 3 or image_array.shape[-1] != 3:
        raise ValueError(f"image must have shape (H,W,3) or (1,H,W,3); got {image_array.shape}")
    if image_array.max(initial=0.0) > 1.0:
        image_array = image_array / 255.0

    heat = np.asarray(heatmap, dtype=np.float32)
    if heat.ndim == 3 and heat.shape[-1] == 1:
        heat = heat[:, :, 0]
    if heat.ndim != 2:
        raise ValueError(f"heatmap must have shape (H,W); got {heat.shape}")
    if heat.shape != image_array.shape[:2]:
        raise ValueError(
            f"heatmap spatial shape {heat.shape} must match image shape {image_array.shape[:2]}"
        )

    coloured = colourise_heatmap(heat, cmap=cmap)
    return np.clip((1.0 - alpha) * image_array + alpha * coloured, 0.0, 1.0)


def save_array_image(path: str | Path, image: np.ndarray, *, cmap: str | None = None) -> None:
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output, np.asarray(image), cmap=cmap, vmin=0.0, vmax=1.0)


def _model_call_input(model, image_batch):
    return [image_batch] if len(getattr(model, "inputs", [])) == 1 else image_batch


def predict_feature_maps(model, image_batch: np.ndarray, layer_names: Sequence[str]) -> dict[str, np.ndarray]:
    import tensorflow as tf

    if not layer_names:
        return {}
    outputs = [model.get_layer(name).output for name in layer_names]
    extractor = tf.keras.Model(model.inputs, outputs)
    images = np.asarray(image_batch, dtype=np.float32)
    values = extractor(_model_call_input(model, images), training=False)
    if len(layer_names) == 1:
        values = [values]
    return {name: np.asarray(value) for name, value in zip(layer_names, values)}


def compute_grad_cam(
    model,
    image_batch: np.ndarray,
    *,
    layer_name: str | None = None,
    class_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import tensorflow as tf

    images = tf.convert_to_tensor(np.asarray(image_batch, dtype=np.float32))
    if images.shape.rank != 4:
        raise ValueError(f"image_batch must have shape (N,H,W,C); got {images.shape}")

    if layer_name is None:
        names = feature_layer_names(model)
        if not names:
            raise ValueError("Could not find a convolution-like layer for Grad-CAM")
        layer_name = names[-1]

    grad_model = tf.keras.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(_model_call_input(model, images), training=False)
        if class_index is None:
            class_indices = tf.argmax(predictions, axis=-1, output_type=tf.int32)
        else:
            batch_size = tf.shape(predictions)[0]
            class_indices = tf.fill((batch_size,), tf.cast(class_index, tf.int32))
        selected = tf.gather(predictions, class_indices, batch_dims=1)

    grads = tape.gradient(selected, conv_outputs)
    if grads is None:
        raise RuntimeError(f"Could not compute gradients for layer {layer_name!r}")

    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_outputs * weights[:, None, None, :], axis=-1)
    cam = tf.nn.relu(cam)
    cam = tf.image.resize(cam[..., None], images.shape[1:3], method="bilinear")[..., 0]

    cam_np = cam.numpy()
    normalised = np.stack([normalise_map(item) for item in cam_np], axis=0)
    return normalised, predictions.numpy(), class_indices.numpy()


__all__ = [
    "colourise_heatmap",
    "compute_grad_cam",
    "feature_layer_names",
    "make_feature_grid",
    "normalise_map",
    "overlay_heatmap",
    "predict_feature_maps",
    "save_array_image",
]

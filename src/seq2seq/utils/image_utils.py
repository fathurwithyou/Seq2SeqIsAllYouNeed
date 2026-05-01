from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _pil_image():
    from PIL import Image

    return Image


def _interpolation_methods():
    image = _pil_image()
    resampling = getattr(image, "Resampling", image)
    return {
        "nearest": resampling.NEAREST,
        "bilinear": resampling.BILINEAR,
        "bicubic": resampling.BICUBIC,
        "hamming": resampling.HAMMING,
        "box": resampling.BOX,
        "lanczos": resampling.LANCZOS,
    }


def load_img(
    path: str | Path,
    color_mode: str = "rgb",
    target_size: tuple[int, int] | None = None,
    interpolation: str = "nearest",
):
    """Loads an image file as a PIL image, following Keras utility semantics."""
    image_module = _pil_image()
    image = image_module.open(path)

    color_mode = color_mode.lower()
    if color_mode == "grayscale":
        if image.mode not in ("L", "I;16", "I"):
            image = image.convert("L")
    elif color_mode == "rgb":
        if image.mode != "RGB":
            image = image.convert("RGB")
    elif color_mode == "rgba":
        if image.mode != "RGBA":
            image = image.convert("RGBA")
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')

    if target_size is not None:
        width_height = (target_size[1], target_size[0])
        if image.size != width_height:
            methods = _interpolation_methods()
            if interpolation not in methods:
                raise ValueError(
                    "Invalid interpolation method "
                    f"{interpolation!r}. Supported methods are "
                    f"{', '.join(methods)}"
                )
            image = image.resize(width_height, methods[interpolation])
    return image


def img_to_array(
    img,
    data_format: str = "channels_last",
    dtype: str | np.dtype = np.float32,
) -> np.ndarray:
    """Converts a PIL image to a NumPy array."""
    if data_format not in {"channels_last", "channels_first"}:
        raise ValueError('data_format must be "channels_last" or "channels_first"')
    array = np.asarray(img, dtype=dtype)
    if array.ndim == 3:
        if data_format == "channels_first":
            array = array.transpose(2, 0, 1)
    elif array.ndim == 2:
        if data_format == "channels_first":
            array = array.reshape((1, array.shape[0], array.shape[1]))
        else:
            array = array.reshape((array.shape[0], array.shape[1], 1))
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}")
    return array


def array_to_img(
    array,
    data_format: str = "channels_last",
    scale: bool = True,
):
    """Converts a 3D NumPy array to a PIL image."""
    if data_format not in {"channels_last", "channels_first"}:
        raise ValueError('data_format must be "channels_last" or "channels_first"')
    image_module = _pil_image()
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(
            "Expected image array to have rank 3. "
            f"Got array with shape: {array.shape}"
        )
    if data_format == "channels_first":
        array = array.transpose(1, 2, 0)
    if scale:
        array = array - np.min(array)
        maximum = np.max(array)
        if maximum != 0:
            array = array / maximum
        array = array * 255
    if array.shape[-1] == 4:
        return image_module.fromarray(array.astype("uint8"), "RGBA")
    if array.shape[-1] == 3:
        return image_module.fromarray(array.astype("uint8"), "RGB")
    if array.shape[-1] == 1:
        return image_module.fromarray(array[:, :, 0].astype("uint8"), "L")
    raise ValueError(f"Unsupported channel number: {array.shape[-1]}")


def save_img(
    path: str | Path,
    array,
    data_format: str = "channels_last",
    file_format: str | None = None,
    scale: bool = True,
    **kwargs,
) -> None:
    image = array_to_img(array, data_format=data_format, scale=scale)
    if file_format is None:
        file_format = Path(path).suffix[1:].lower() or None
    if file_format == "jpg":
        file_format = "jpeg"
    if image.mode == "RGBA" and file_format == "jpeg":
        image = image.convert("RGB")
    image.save(path, format=file_format, **kwargs)


def load_image(
    path: str | Path,
    target_size: tuple[int, int] | None = None,
    *,
    color_mode: str = "rgb",
    normalize: bool = True,
    dtype: str | np.dtype = np.float32,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Loads an image as a normalized array for the assignment pipelines."""
    image = load_img(
        path,
        color_mode=color_mode,
        target_size=target_size,
        interpolation=interpolation,
    )
    array = img_to_array(image, dtype=dtype)
    if normalize:
        array = array / np.asarray(255.0, dtype=dtype)
    return array.astype(dtype, copy=False)


def load_image_batch(
    paths: Sequence[str | Path],
    target_size: tuple[int, int],
    *,
    color_mode: str = "rgb",
    normalize: bool = True,
    dtype: str | np.dtype = np.float32,
) -> np.ndarray:
    if target_size is None:
        raise ValueError("target_size is required for batch loading")
    images = [
        load_image(
            path,
            target_size,
            color_mode=color_mode,
            normalize=normalize,
            dtype=dtype,
        )
        for path in paths
    ]
    return np.stack(images, axis=0).astype(dtype, copy=False)


def extract_features_with_keras(
    paths: Iterable[str | Path],
    encoder,
    target_size: tuple[int, int],
    *,
    batch_size: int = 32,
    preprocess=None,
    save_path: str | Path | None = None,
    return_paths: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[Path]]:
    path_list = [Path(path) for path in paths]
    features: list[np.ndarray] = []
    for start in range(0, len(path_list), batch_size):
        batch_paths = path_list[start : start + batch_size]
        images = load_image_batch(
            batch_paths,
            target_size,
            color_mode="rgb",
            normalize=False,
            dtype=np.float32,
        )
        images = preprocess(images) if preprocess is not None else images / 255.0
        batch_features = encoder.predict(images, verbose=0)
        features.append(np.asarray(batch_features, dtype=np.float32))
    output = np.concatenate(features, axis=0)
    if save_path is not None:
        np.save(Path(save_path), output)
    if return_paths:
        return output, path_list
    return output


__all__ = [
    "array_to_img",
    "img_to_array",
    "load_img",
    "save_img",
    "load_image",
    "load_image_batch",
    "extract_features_with_keras",
]

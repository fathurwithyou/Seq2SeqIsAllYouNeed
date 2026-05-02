from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from ..layers import (
    Conv2D,
    Dense,
    Embedding,
    LSTM,
    LSTMCell,
    LocallyConnected2D,
    SimpleRNN,
    SimpleRNNCell,
)
from ..layers.layer import Layer


_NATIVE_WEIGHTS_EXTENSION = ".weights.npz"


def save_weights(model: Layer, filepath, overwrite: bool = True, **kwargs) -> None:
    if kwargs:
        raise ValueError(f"Invalid keyword arguments: {kwargs}")
    filepath = Path(filepath)
    filepath_str = str(filepath)
    if not filepath_str.endswith(_NATIVE_WEIGHTS_EXTENSION):
        raise ValueError(
            f"The filename must end in `{_NATIVE_WEIGHTS_EXTENSION}`. "
            f"Received: filepath={filepath_str}"
        )
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(
            f"File already exists: {filepath_str}. Pass overwrite=True to replace it."
        )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("wb") as handle:
        np.savez(handle, **model.state_dict())


def load_weights(model: Layer, filepath, skip_mismatch: bool = False, **kwargs) -> Layer:
    if kwargs:
        raise ValueError(f"Invalid keyword arguments: {kwargs}")
    filepath = Path(filepath)
    filepath_str = str(filepath)
    if not filepath_str.endswith(_NATIVE_WEIGHTS_EXTENSION):
        raise ValueError(
            f"File format not supported: filepath={filepath_str}. "
            f"Seq2Seq native weights must use `{_NATIVE_WEIGHTS_EXTENSION}`. "
            "Use the Keras weight-transfer helpers for Keras `.weights.h5` files."
        )
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: filepath={filepath_str}")
    with np.load(filepath, allow_pickle=False) as data:
        state = {key: data[key] for key in data.files}
    model.load_state_dict(state, strict=not skip_mismatch)
    return model


def _assign(weight, array, name: str) -> None:
    arr = np.asarray(array)
    if arr.shape != weight.shape:
        raise ValueError(f"shape mismatch for {name}: expected {weight.shape}, got {arr.shape}")
    weight.data = arr.astype(weight.data.dtype, copy=False)


def _assign_kernel_bias(
    layer: Layer,
    weights: Sequence[np.ndarray],
    *,
    label: str,
    bias_attr: str = "bias",
) -> None:
    if len(weights) not in (1, 2):
        raise ValueError(f"{label} expects 1 or 2 weight arrays, got {len(weights)}")
    _assign(layer.kernel, weights[0], f"{label}.kernel")
    bias = getattr(layer, bias_attr, None)
    if bias is not None and len(weights) == 2:
        _assign(bias, weights[1], f"{label}.{bias_attr}")


def _assign_recurrent_cell(
    cell: Layer,
    weights: Sequence[np.ndarray],
    *,
    label: str,
) -> None:
    if len(weights) not in (2, 3):
        raise ValueError(f"{label} expects 2 or 3 weight arrays, got {len(weights)}")
    _assign(cell.kernel, weights[0], f"{label}.kernel")
    _assign(cell.recurrent_kernel, weights[1], f"{label}.recurrent_kernel")
    if cell.bias is not None and len(weights) == 3:
        _assign(cell.bias, weights[2], f"{label}.bias")


def _load_stacked(
    layer: Layer,
    per_cell_weights: Sequence[Sequence[np.ndarray]],
    cell_loader,
    *,
    label: str,
) -> None:
    if not layer.built:
        layer.build((None, None, per_cell_weights[0][0].shape[0]))
    if len(per_cell_weights) != layer.num_layers:
        raise ValueError(
            f"{label} has {layer.num_layers} cells but received "
            f"{len(per_cell_weights)} weight groups"
        )
    for cell, weights in zip(layer.cells, per_cell_weights):
        cell_loader(cell, weights)


def load_conv2d(layer: Conv2D, weights: Sequence[np.ndarray]) -> None:
    if not layer.built:
        layer.build((None, None, None, weights[0].shape[2]))
    _assign_kernel_bias(layer, weights, label="Conv2D")


def load_locally_connected2d(layer: LocallyConnected2D, weights: Sequence[np.ndarray]) -> None:
    if not layer.built:
        raise ValueError("LocallyConnected2D must be built before loading weights")
    _assign_kernel_bias(layer, weights, label="LocallyConnected2D")


def load_dense(layer: Dense, weights: Sequence[np.ndarray]) -> None:
    if not layer.built:
        layer.build((None, weights[0].shape[0]))
    _assign_kernel_bias(layer, weights, label="Dense")


def load_embedding(layer: Embedding, weights: Sequence[np.ndarray]) -> None:
    if len(weights) != 1:
        raise ValueError(f"Embedding expects 1 weight array, got {len(weights)}")
    _assign(layer.embeddings, weights[0], "Embedding.embeddings")


def load_simple_rnn_cell(cell: SimpleRNNCell, weights: Sequence[np.ndarray]) -> None:
    if not cell.built:
        cell.build((None, weights[0].shape[0]))
    _assign_recurrent_cell(cell, weights, label="SimpleRNNCell")


def load_lstm_cell(cell: LSTMCell, weights: Sequence[np.ndarray]) -> None:
    if not cell.built:
        cell.build((None, weights[0].shape[0]))
    _assign_recurrent_cell(cell, weights, label="LSTMCell")


def load_simple_rnn(layer: SimpleRNN, layer_weights_per_cell: Sequence[Sequence[np.ndarray]]) -> None:
    _load_stacked(layer, layer_weights_per_cell, load_simple_rnn_cell, label="SimpleRNN")


def load_lstm(layer: LSTM, layer_weights_per_cell: Sequence[Sequence[np.ndarray]]) -> None:
    _load_stacked(layer, layer_weights_per_cell, load_lstm_cell, label="LSTM")


_LOADERS: tuple[tuple[type, Callable[[Layer, Sequence[np.ndarray]], None]], ...] = (
    (Conv2D, load_conv2d),
    (LocallyConnected2D, load_locally_connected2d),
    (Dense, load_dense),
    (Embedding, load_embedding),
    (SimpleRNNCell, load_simple_rnn_cell),
    (LSTMCell, load_lstm_cell),
)


def load_layer_from_keras(layer: Layer, weights: Sequence[np.ndarray]) -> None:
    for cls, loader in _LOADERS:
        if isinstance(layer, cls):
            return loader(layer, weights)
    raise TypeError(f"No Keras loader registered for layer type {type(layer).__name__}")


def assign_weights_in_order(layers: Iterable[Layer], all_weights: Sequence[Sequence[np.ndarray]]) -> None:
    layer_list = list(layers)
    if len(layer_list) != len(all_weights):
        raise ValueError(f"got {len(layer_list)} layers but {len(all_weights)} weight groups")
    for layer, weights in zip(layer_list, all_weights):
        load_layer_from_keras(layer, weights)


def extract_keras_weights(model) -> dict[str, list[np.ndarray]]:
    output: dict[str, list[np.ndarray]] = {}
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            output[layer.name] = [np.asarray(weight) for weight in weights]
    return output

__all__ = [
    "save_weights",
    "load_weights",
    "extract_keras_weights",
    "load_conv2d",
    "load_locally_connected2d",
    "load_dense",
    "load_embedding",
    "load_lstm",
    "load_lstm_cell",
    "load_simple_rnn",
    "load_simple_rnn_cell",
    "load_layer_from_keras",
    "assign_weights_in_order",
]

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def pad_sequences(
    sequences: Sequence[Sequence[Any]],
    maxlen: int | None = None,
    dtype: str | np.dtype = "int32",
    padding: str = "pre",
    truncating: str = "pre",
    value: float | str = 0.0,
) -> np.ndarray:
    """Pads sequences to the same length.

    This follows the public Keras utility semantics for the subset needed by
    the project: pre/post padding, pre/post truncation, dtype control, and
    support for sequence items with a consistent trailing sample shape.
    """
    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    if padding not in {"pre", "post"}:
        raise ValueError(f'Padding type "{padding}" not understood')
    if truncating not in {"pre", "post"}:
        raise ValueError(f'Truncating type "{truncating}" not understood')

    num_samples = len(sequences)
    lengths: list[int] = []
    sample_shape: tuple[int, ...] = ()

    for sequence in sequences:
        try:
            length = len(sequence)
        except TypeError as exc:
            raise ValueError(
                "`sequences` must be a list of iterables. "
                f"Found non-iterable: {sequence!r}"
            ) from exc
        lengths.append(length)
        if length and not sample_shape:
            sample_shape = np.asarray(sequence).shape[1:]

    if maxlen is None:
        maxlen = int(np.max(lengths)) if lengths else 0

    is_string_dtype = np.issubdtype(np.dtype(dtype), np.str_)
    if isinstance(value, str) and dtype is not object and not is_string_dtype:
        raise ValueError(
            f"`dtype` {dtype} is not compatible with `value`'s type: "
            f"{type(value)}. Use `dtype=object` for variable length strings."
        )

    output = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for index, sequence in enumerate(sequences):
        if not len(sequence):
            continue
        trunc = sequence[-maxlen:] if truncating == "pre" else sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                f"Shape of sample {trunc.shape[1:]} of sequence at position "
                f"{index} is different from expected shape {sample_shape}"
            )
        if padding == "post":
            output[index, : len(trunc)] = trunc
        else:
            output[index, -len(trunc) :] = trunc
    return output


__all__ = ["pad_sequences"]

from __future__ import annotations

import numpy as np

from ...tensor import Tensor
from ..layer import Layer


class Embedding(Layer):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        embeddings_initializer: str = "random_uniform",
        mask_zero: bool = False,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.embeddings_initializer = embeddings_initializer
        self.mask_zero = bool(mask_zero)
        self.seed = seed
        self.embeddings = None
        self.build((None, None))

    def build(self, input_shape) -> None:
        self.embeddings = self.add_weight(
            (self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name="embeddings",
            seed=self.seed,
        )
        if self.mask_zero:
            self.embeddings.data[0] = 0.0
        self.built = True

    def call(self, inputs):
        ids = (
            inputs.data.astype(np.int64)
            if isinstance(inputs, Tensor)
            else np.asarray(inputs, dtype=np.int64)
        )
        if ids.size and (ids.min() < 0 or ids.max() >= self.input_dim):
            raise IndexError(
                f"Embedding indices out of range [0, {self.input_dim}); "
                f"got min={int(ids.min())}, max={int(ids.max())}"
            )
        outputs = self.embeddings.data[ids]
        return Tensor(outputs) if isinstance(inputs, Tensor) else outputs

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"mask_zero={self.mask_zero}"
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "embeddings_initializer": self.embeddings_initializer,
                "mask_zero": self.mask_zero,
                "seed": self.seed,
            }
        )
        return config


__all__ = ["Embedding"]

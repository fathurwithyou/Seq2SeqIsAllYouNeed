from __future__ import annotations

import math
from typing import Any

import numpy as np

from .initializer import Initializer


def _as_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def compute_fans(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) < 1:
        return 1, 1
    if len(shape) == 1:
        return int(shape[0]), int(shape[0])
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    receptive_field_size = int(np.prod(shape[:-2]))
    fan_in = int(shape[-2] * receptive_field_size)
    fan_out = int(shape[-1] * receptive_field_size)
    return fan_in, fan_out


class RandomInitializer(Initializer):
    def __init__(self, seed: Any = None) -> None:
        self.seed = seed

    def get_config(self) -> dict:
        return {"seed": self.seed}


class RandomUniform(RandomInitializer):
    def __init__(
        self,
        minval: float = -0.05,
        maxval: float = 0.05,
        seed: Any = None,
    ) -> None:
        self.minval = minval
        self.maxval = maxval
        super().__init__(seed=seed)

    def __call__(self, shape, dtype=None):
        rng = _as_rng(self.seed)
        return rng.uniform(self.minval, self.maxval, size=shape).astype(
            np.dtype(dtype or np.float32)
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"minval": self.minval, "maxval": self.maxval})
        return config


class RandomNormal(RandomInitializer):
    def __init__(
        self,
        mean: float = 0.0,
        stddev: float = 0.05,
        seed: Any = None,
    ) -> None:
        self.mean = mean
        self.stddev = stddev
        super().__init__(seed=seed)

    def __call__(self, shape, dtype=None):
        rng = _as_rng(self.seed)
        return rng.normal(self.mean, self.stddev, size=shape).astype(
            np.dtype(dtype or np.float32)
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"mean": self.mean, "stddev": self.stddev})
        return config


class VarianceScaling(RandomInitializer):
    def __init__(
        self,
        scale: float = 1.0,
        mode: str = "fan_in",
        distribution: str = "truncated_normal",
        seed: Any = None,
    ) -> None:
        if scale <= 0.0:
            raise ValueError(f"`scale` must be positive. Received: scale={scale}")
        if mode not in {"fan_in", "fan_out", "fan_avg"}:
            raise ValueError(f"Invalid `mode`: {mode!r}")
        distribution = distribution.lower()
        if distribution == "normal":
            distribution = "truncated_normal"
        if distribution not in {"uniform", "truncated_normal", "untruncated_normal"}:
            raise ValueError(f"Invalid `distribution`: {distribution!r}")
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        super().__init__(seed=seed)

    def __call__(self, shape, dtype=None):
        fan_in, fan_out = compute_fans(tuple(shape))
        scale = self.scale
        if self.mode == "fan_in":
            scale /= max(1.0, fan_in)
        elif self.mode == "fan_out":
            scale /= max(1.0, fan_out)
        else:
            scale /= max(1.0, (fan_in + fan_out) / 2.0)

        if self.distribution == "uniform":
            limit = math.sqrt(3.0 * scale)
            return RandomUniform(-limit, limit, seed=self.seed)(shape, dtype=dtype)

        stddev = math.sqrt(scale)
        if self.distribution == "truncated_normal":
            stddev = stddev / 0.87962566103423978
        return RandomNormal(0.0, stddev, seed=self.seed)(shape, dtype=dtype)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "scale": self.scale,
                "mode": self.mode,
                "distribution": self.distribution,
            }
        )
        return config


class GlorotUniform(VarianceScaling):
    def __init__(self, seed: Any = None) -> None:
        super().__init__(scale=1.0, mode="fan_avg", distribution="uniform", seed=seed)

    def get_config(self) -> dict:
        return {"seed": self.seed}


class GlorotNormal(VarianceScaling):
    def __init__(self, seed: Any = None) -> None:
        super().__init__(
            scale=1.0,
            mode="fan_avg",
            distribution="truncated_normal",
            seed=seed,
        )

    def get_config(self) -> dict:
        return {"seed": self.seed}


class HeUniform(VarianceScaling):
    def __init__(self, seed: Any = None) -> None:
        super().__init__(scale=2.0, mode="fan_in", distribution="uniform", seed=seed)

    def get_config(self) -> dict:
        return {"seed": self.seed}


class HeNormal(VarianceScaling):
    def __init__(self, seed: Any = None) -> None:
        super().__init__(
            scale=2.0,
            mode="fan_in",
            distribution="truncated_normal",
            seed=seed,
        )

    def get_config(self) -> dict:
        return {"seed": self.seed}


__all__ = [
    "RandomInitializer",
    "RandomUniform",
    "RandomNormal",
    "VarianceScaling",
    "GlorotUniform",
    "GlorotNormal",
    "HeUniform",
    "HeNormal",
    "compute_fans",
]

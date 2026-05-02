from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .. import initializers
from ..tensor import Tensor


class Weight(Tensor):
    def __init__(self, data: Any, *, trainable: bool = True) -> None:
        super().__init__(data, requires_grad=trainable)
        self.trainable = bool(trainable)

    def __repr__(self) -> str:
        return (
            f"Weight(shape={self.shape}, dtype={self.data.dtype}, "
            f"trainable={self.trainable})"
        )


class Layer:
    _name_counters: dict[str, int] = {}

    def __init__(
        self,
        *,
        name: str | None = None,
        trainable: bool = True,
        dtype: np.dtype | str = np.float32,
    ) -> None:
        super().__setattr__("_weights", OrderedDict())
        super().__setattr__("_layers", OrderedDict())
        super().__setattr__("name", name or self._generate_name())
        super().__setattr__("trainable", bool(trainable))
        super().__setattr__("dtype", np.dtype(dtype))
        super().__setattr__("built", False)
        super().__setattr__("training", True)
        super().__setattr__("input_spec", None)

    @classmethod
    def _generate_name(cls) -> str:
        base = cls.__name__
        chars: list[str] = []
        for index, char in enumerate(base):
            if char.isupper() and index > 0:
                chars.append("_")
            chars.append(char.lower())
        base_name = "".join(chars)
        count = cls._name_counters.get(base_name, 0) + 1
        cls._name_counters[base_name] = count
        return f"{base_name}_{count}"

    def __setattr__(self, name: str, value: Any) -> None:
        if "_weights" not in self.__dict__:
            super().__setattr__(name, value)
            return

        self._weights.pop(name, None)
        self._layers.pop(name, None)

        if isinstance(value, Weight):
            self._weights[name] = value
        elif isinstance(value, Layer):
            self._layers[name] = value

        super().__setattr__(name, value)

    def add_weight(
        self,
        shape: tuple[int, ...] | list[int],
        *,
        initializer: str | Callable | None = "zeros",
        trainable: bool = True,
        name: str | None = None,
        dtype: np.dtype | str | None = None,
        seed: int | None = None,
    ) -> Weight:
        shape = tuple(shape)
        dtype = np.dtype(dtype or self.dtype)
        init = initializers.get(initializer)
        if seed is not None and hasattr(init, "get_config"):
            config = init.get_config()
            if config.get("seed") is None and "seed" in config:
                config["seed"] = seed
                init = init.__class__.from_config(config)
        values = init(shape=shape, dtype=dtype)
        weight = Weight(values, trainable=trainable)
        if name is not None:
            setattr(self, name, weight)
        return weight

    def add_layer(self, name: str, layer: "Layer | None") -> None:
        if layer is None:
            self._layers[name] = None
            super().__setattr__(name, None)
            return
        if not isinstance(layer, Layer):
            raise TypeError(f"Expected Layer, got {type(layer).__name__}")
        self._layers[name] = layer
        super().__setattr__(name, layer)

    def build(self, input_shape) -> None:
        self.built = True

    def call(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(f"{type(self).__name__} does not implement call()")

    def _infer_input_shape(self, value):
        if isinstance(value, Tensor):
            return value.shape
        if hasattr(value, "shape"):
            return tuple(value.shape)
        return None

    def __call__(self, *args: Any, **kwargs: Any):
        if args and self.input_spec is not None:
            specs = (
                self.input_spec
                if isinstance(self.input_spec, (list, tuple))
                else [self.input_spec]
            )
            values = args if isinstance(self.input_spec, (list, tuple)) else [args[0]]
            for spec, value in zip(specs, values):
                spec.assert_compatible(value, layer_name=self.name)
        if not self.built:
            input_shape = self._infer_input_shape(args[0]) if args else None
            self.build(input_shape)
            self.built = True
        return self.call(*args, **kwargs)

    @property
    def layers(self) -> list["Layer"]:
        return [layer for layer in self._layers.values() if layer is not None]

    def named_layers(self, prefix: str = "") -> Iterator[tuple[str, "Layer"]]:
        for name, layer in self._layers.items():
            if layer is None:
                continue
            full_name = f"{prefix}{name}" if prefix else name
            yield full_name, layer
            yield from layer.named_layers(prefix=f"{full_name}.")

    @property
    def weights(self) -> list[Weight]:
        out: list[Weight] = []
        out.extend(weight for weight in self._weights.values() if weight is not None)
        for layer in self.layers:
            out.extend(layer.weights)
        return out

    @property
    def trainable_weights(self) -> list[Weight]:
        return [weight for weight in self.weights if weight.trainable]

    @property
    def non_trainable_weights(self) -> list[Weight]:
        return [weight for weight in self.weights if not weight.trainable]

    @property
    def variables(self) -> list[Weight]:
        return self.weights

    def named_weights(self, prefix: str = "") -> Iterator[tuple[str, Weight]]:
        for name, weight in self._weights.items():
            if weight is None:
                continue
            yield (f"{prefix}{name}" if prefix else name), weight
        for layer_name, layer in self._layers.items():
            if layer is None:
                continue
            sub_prefix = f"{prefix}{layer_name}." if prefix else f"{layer_name}."
            yield from layer.named_weights(prefix=sub_prefix)

    def get_weights(self) -> list[np.ndarray]:
        return [weight.data.copy() for weight in self.weights]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        own_weights = self.weights
        if len(weights) != len(own_weights):
            raise ValueError(
                f"Expected {len(own_weights)} weight arrays, got {len(weights)}"
            )
        for weight, array in zip(own_weights, weights):
            arr = np.asarray(array)
            if arr.shape != weight.shape:
                raise ValueError(
                    f"shape mismatch: expected {weight.shape}, got {arr.shape}"
                )
            weight.data = arr.astype(weight.data.dtype, copy=False)

    def apply(self, fn: Callable[["Layer"], None]) -> "Layer":
        fn(self)
        for layer in self.layers:
            layer.apply(fn)
        return self

    def zero_grad(self) -> None:
        for weight in self.weights:
            weight.zero_grad()

    def train(self, mode: bool = True) -> "Layer":
        self.training = bool(mode)
        for layer in self.layers:
            layer.train(mode)
        return self

    def eval(self) -> "Layer":
        return self.train(False)

    def state_dict(self) -> OrderedDict:
        state = OrderedDict()
        for name, weight in self.named_weights():
            state[name] = weight.data.copy()
        return state

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        own_weights = dict(self.named_weights())
        missing: list[str] = []
        unexpected: list[str] = list(state.keys())
        for name, weight in own_weights.items():
            if name not in state:
                missing.append(name)
                continue
            arr = np.asarray(state[name])
            if arr.shape != weight.shape:
                if strict:
                    raise ValueError(
                        f"shape mismatch for '{name}': expected {weight.shape}, got {arr.shape}"
                    )
                continue
            weight.data = arr.astype(weight.data.dtype, copy=False)
            unexpected.remove(name)
        if strict and (missing or unexpected):
            raise KeyError(
                f"load_state_dict mismatch: missing={missing}, unexpected={unexpected}"
            )

    def save_weights(self, filepath: str | Path) -> None:
        with Path(filepath).open("wb") as handle:
            np.savez(handle, **self.state_dict())

    def load_weights(self, filepath: str | Path, *, skip_mismatch: bool = False) -> "Layer":
        with np.load(Path(filepath), allow_pickle=False) as data:
            state = {key: data[key] for key in data.files}
        self.load_state_dict(state, strict=not skip_mismatch)
        return self

    def count_params(self) -> int:
        return sum(weight.size for weight in self.weights)

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Layer":
        return cls(**config)

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        extra = self.extra_repr()
        child_lines = []
        for name, layer in self._layers.items():
            if layer is None:
                continue
            child_lines.append(f"  ({name}): {layer!r}".replace("\n", "\n  "))
        main = type(self).__name__ + "("
        if extra and not child_lines:
            return f"{main}{extra})"
        body = []
        if extra:
            body.append(extra)
        body.extend(child_lines)
        return main + ("\n" + "\n".join(body) + "\n)" if body else ")")

__all__ = ["Layer", "Weight"]

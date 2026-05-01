from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import numpy as np

from ..tensor import Tensor


class Optimizer:
    def __init__(
        self,
        params: Iterable[Tensor] | None = None,
        defaults: dict[str, Any] | None = None,
        *,
        name: str | None = None,
    ) -> None:
        if isinstance(params, Tensor):
            raise TypeError("params must be an iterable of tensors, not a single tensor")
        param_list = list(params or [])
        self.defaults = dict(defaults or {})
        self.name = name or type(self).__name__
        self.built = bool(param_list)
        self.state: dict[int, dict[str, Any]] = {}
        self.iterations = 0
        self.param_groups = [{"params": param_list, **deepcopy(self.defaults)}]
        self._validate_params(param_list)

    @staticmethod
    def _validate_params(params: Iterable[Tensor]) -> None:
        for param in params:
            if not isinstance(param, Tensor):
                raise TypeError(f"optimizer can only optimize Tensors, got {type(param)}")

    @property
    def params(self) -> list[Tensor]:
        params: list[Tensor] = []
        for group in self.param_groups:
            params.extend(group["params"])
        return params

    @property
    def variables(self) -> list[Tensor]:
        return self.params

    @property
    def learning_rate(self) -> float | None:
        if not self.param_groups or "learning_rate" not in self.param_groups[0]:
            return None
        return self.param_groups[0]["learning_rate"]

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        for group in self.param_groups:
            group["learning_rate"] = float(value)

    def build(self, variables: Iterable[Tensor]) -> None:
        param_list = list(variables)
        if not param_list:
            raise ValueError("optimizer got an empty variable list")
        self._validate_params(param_list)
        self.param_groups = [{"params": param_list, **deepcopy(self.defaults)}]
        self.built = True

    def apply_gradients(self, grads_and_vars) -> None:
        pairs = list(grads_and_vars)
        if not pairs:
            return self.iterations
        grads: list[np.ndarray] = []
        variables: list[Tensor] = []
        for grad, variable in pairs:
            if not isinstance(variable, Tensor):
                raise TypeError(f"apply_gradients expects Tensor variables, got {type(variable)}")
            if grad is None:
                continue
            grads.append(grad.data if isinstance(grad, Tensor) else np.asarray(grad, dtype=variable.data.dtype))
            variables.append(variable)
        if not variables:
            raise ValueError("No gradients provided for any variable.")
        if not self.built:
            self.build(variables)
        known = {id(param) for param in self.params}
        for variable in variables:
            if id(variable) not in known:
                raise ValueError(
                    "Unknown variable passed to optimizer. Create a new optimizer "
                    "for a different set of variables."
                )
        for grad, variable in zip(grads, variables):
            self.update_step(grad, variable, self.learning_rate)
        self.iterations += 1
        return self.iterations

    def update_step(self, gradient, variable, learning_rate) -> None:
        raise NotImplementedError

    def get_config(self) -> dict[str, Any]:
        config = {"name": self.name}
        config.update(self.defaults)
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        return cls(**dict(config))


__all__ = ["Optimizer"]

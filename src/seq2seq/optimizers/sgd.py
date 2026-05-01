from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..tensor import Tensor
from .base_optimizer import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor] | None = None,
        *,
        learning_rate: float = 1e-2,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float | None = None,
        name: str = "SGD",
    ) -> None:
        if not isinstance(momentum, float) or momentum < 0 or momentum > 1:
            raise ValueError("`momentum` must be a float between [0, 1].")
        super().__init__(
            params,
            {
                "learning_rate": float(learning_rate),
                "momentum": float(momentum),
                "nesterov": bool(nesterov),
                "weight_decay": None if weight_decay is None else float(weight_decay),
            },
            name=name,
        )

    def update_step(self, gradient, variable, learning_rate) -> None:
        group = self.param_groups[0]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        weight_decay = group["weight_decay"]
        grad = np.asarray(gradient, dtype=variable.data.dtype)
        if weight_decay is not None:
            grad = grad + weight_decay * variable.data
        if momentum != 0.0:
            state = self.state.setdefault(id(variable), {})
            velocity = state.get("momentum")
            if velocity is None:
                velocity = np.zeros_like(variable.data)
            velocity = momentum * velocity - learning_rate * grad
            state["momentum"] = velocity
            if nesterov:
                variable.data = variable.data + momentum * velocity - learning_rate * grad
            else:
                variable.data = variable.data + velocity
        else:
            variable.data = variable.data - learning_rate * grad


__all__ = ["SGD"]

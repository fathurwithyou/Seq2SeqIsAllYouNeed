from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..tensor import Tensor
from .base_optimizer import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor] | None = None,
        *,
        learning_rate: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        weight_decay: float | None = None,
        name: str = "adam",
    ) -> None:
        super().__init__(
            params,
            {
                "learning_rate": float(learning_rate),
                "beta_1": float(beta_1),
                "beta_2": float(beta_2),
                "epsilon": float(epsilon),
                "amsgrad": bool(amsgrad),
                "weight_decay": None if weight_decay is None else float(weight_decay),
            },
            name=name,
        )

    def update_step(self, gradient, variable, learning_rate) -> None:
        group = self.param_groups[0]
        beta1 = group["beta_1"]
        beta2 = group["beta_2"]
        epsilon = group["epsilon"]
        amsgrad = group["amsgrad"]
        weight_decay = group["weight_decay"]
        grad = np.asarray(gradient, dtype=variable.data.dtype)
        if weight_decay is not None:
            grad = grad + weight_decay * variable.data
        local_step = self.iterations + 1
        beta1_power = beta1 ** local_step
        beta2_power = beta2 ** local_step
        state = self.state.setdefault(id(variable), {})
        m = state.setdefault("m", np.zeros_like(variable.data))
        v = state.setdefault("v", np.zeros_like(variable.data))
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        state["m"] = m
        state["v"] = v
        alpha = learning_rate * np.sqrt(1.0 - beta2_power) / (1.0 - beta1_power)
        if amsgrad:
            vhat = state.setdefault("vhat", np.zeros_like(variable.data))
            vhat = np.maximum(vhat, v)
            state["vhat"] = vhat
            v = vhat
        variable.data = variable.data - (m * alpha) / (np.sqrt(v) + epsilon)


__all__ = ["Adam"]

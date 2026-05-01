from __future__ import annotations

from ..layers.layer import Layer
from ..trainers import Trainer


class Model(Trainer, Layer):
    def get_layer(self, name: str):
        for layer_name, layer in self.named_layers():
            if layer_name == name or layer.name == name:
                return layer
        raise ValueError(f"No layer named {name!r}")


__all__ = ["Model"]

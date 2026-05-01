from __future__ import annotations

from collections import OrderedDict

from ..layers.layer import Layer
from .model import Model


class Sequential(Model):
    def __init__(self, layers=None, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._stack: list[Layer] = []
        if layers is None:
            return
        if isinstance(layers, OrderedDict):
            for layer_name, layer in layers.items():
                self.add(layer, name=layer_name)
            return
        for layer in layers:
            self.add(layer)

    def add(self, layer: Layer, *, name: str | None = None) -> None:
        layer_name = name or str(len(self._stack))
        self._stack.append(layer)
        self.add_layer(layer_name, layer)

    def pop(self):
        if not self._stack:
            raise IndexError("pop from empty Sequential")
        index = len(self._stack) - 1
        layer = self._stack.pop()
        self._layers.pop(str(index), None)
        self.__dict__.pop(str(index), None)
        return layer

    def call(self, inputs):
        outputs = inputs
        for layer in self._stack:
            outputs = layer(outputs)
        return outputs

    def __len__(self) -> int:
        return len(self._stack)

    def __iter__(self):
        return iter(self._stack)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Sequential(self._stack[index])
        return self._stack[index]


__all__ = ["Sequential"]

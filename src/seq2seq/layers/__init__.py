from .activations import (
    Activation,
    ELU,
    GELU,
    LeakyReLU,
    ReLU,
    Sigmoid,
    Softmax,
    Softplus,
    Softsign,
    Tanh,
)
from .convolutional import Conv2D, LocallyConnected2D
from .core import Dense, Embedding
from .input_spec import InputSpec
from .layer import Layer, Weight
from .pooling import (
    AveragePooling2D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    MaxPooling2D,
)
from .reshaping import Flatten
from .rnn import LSTM, LSTMCell, SimpleRNN, SimpleRNNCell

_ALL_OBJECTS = {
    cls.__name__: cls
    for cls in (
        Activation,
        ELU,
        GELU,
        LeakyReLU,
        ReLU,
        Sigmoid,
        Softmax,
        Softplus,
        Softsign,
        Tanh,
        Dense,
        Embedding,
        Flatten,
        Conv2D,
        LocallyConnected2D,
        MaxPooling2D,
        AveragePooling2D,
        GlobalAveragePooling2D,
        GlobalMaxPooling2D,
        SimpleRNN,
        SimpleRNNCell,
        LSTM,
        LSTMCell,
    )
}
_ALL_OBJECTS.update({name.lower(): value for name, value in _ALL_OBJECTS.items()})


def serialize(layer: Layer) -> dict:
    if not isinstance(layer, Layer):
        raise ValueError(f"`serialize` expects a Layer instance. Received: {layer!r}")
    return {"class_name": type(layer).__name__, "config": layer.get_config()}


def deserialize(config: dict, custom_objects: dict | None = None) -> Layer:
    custom_objects = custom_objects or {}
    class_name = config.get("class_name")
    layer_config = config.get("config", {})
    cls = custom_objects.get(class_name) or _ALL_OBJECTS.get(class_name)
    if cls is None:
        raise ValueError(f"Unknown layer: {class_name!r}")
    layer = cls.from_config(layer_config)
    if not isinstance(layer, Layer):
        raise ValueError(
            f"`deserialize` expected a Layer config. Received class {class_name!r}"
        )
    return layer


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, Layer):
        return identifier
    if isinstance(identifier, dict):
        return deserialize(identifier)
    if isinstance(identifier, str):
        layer = _ALL_OBJECTS.get(identifier) or _ALL_OBJECTS.get(identifier.lower())
        if layer is None:
            raise ValueError(f"Unknown layer: {identifier!r}")
        return layer
    if isinstance(identifier, type) and issubclass(identifier, Layer):
        return identifier
    raise ValueError(f"Could not interpret layer identifier: {identifier!r}")


__all__ = [
    "Layer",
    "Weight",
    "InputSpec",
    "serialize",
    "deserialize",
    "get",
    "Activation",
    "ELU",
    "GELU",
    "LeakyReLU",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "Softplus",
    "Softsign",
    "Dense",
    "Embedding",
    "Flatten",
    "Conv2D",
    "LocallyConnected2D",
    "MaxPooling2D",
    "AveragePooling2D",
    "GlobalAveragePooling2D",
    "GlobalMaxPooling2D",
    "SimpleRNN",
    "SimpleRNNCell",
    "LSTM",
    "LSTMCell",
]

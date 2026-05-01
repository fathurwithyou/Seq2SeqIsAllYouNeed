from .constant_initializers import Constant, Ones, Zeros
from .initializer import Initializer
from .random_initializers import (
    GlorotNormal,
    GlorotUniform,
    HeNormal,
    HeUniform,
    RandomInitializer,
    RandomNormal,
    RandomUniform,
    VarianceScaling,
    compute_fans,
)


ALL_OBJECTS = {
    "Constant": Constant,
    "Zeros": Zeros,
    "Ones": Ones,
    "RandomUniform": RandomUniform,
    "RandomNormal": RandomNormal,
    "VarianceScaling": VarianceScaling,
    "GlorotUniform": GlorotUniform,
    "GlorotNormal": GlorotNormal,
    "HeUniform": HeUniform,
    "HeNormal": HeNormal,
    "constant": Constant,
    "zeros": Zeros,
    "ones": Ones,
    "random_uniform": RandomUniform,
    "random_normal": RandomNormal,
    "variance_scaling": VarianceScaling,
    "glorot_uniform": GlorotUniform,
    "glorot_normal": GlorotNormal,
    "he_uniform": HeUniform,
    "he_normal": HeNormal,
}


def serialize(initializer):
    if initializer is None:
        return None
    initializer = get(initializer)
    if isinstance(initializer, Initializer):
        return {
            "class_name": initializer.__class__.__name__,
            "config": initializer.get_config(),
        }
    return initializer


def deserialize(config, custom_objects=None):
    if config is None:
        return None
    if isinstance(config, str):
        return get(config)
    custom_objects = custom_objects or {}
    class_name = config.get("class_name")
    initializer_config = config.get("config", {})
    cls = custom_objects.get(class_name) or ALL_OBJECTS.get(class_name)
    if cls is None:
        raise ValueError(f"Unknown initializer: {class_name!r}")
    return cls.from_config(initializer_config)


def get(identifier):
    if identifier is None:
        return Zeros()
    if isinstance(identifier, Initializer):
        return identifier
    if isinstance(identifier, dict):
        return deserialize(identifier)
    if isinstance(identifier, str):
        cls = ALL_OBJECTS.get(identifier) or ALL_OBJECTS.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Unsupported initializer: {identifier!r}")
        return cls()
    if isinstance(identifier, type) and issubclass(identifier, Initializer):
        return identifier()
    if callable(identifier):
        return identifier
    raise ValueError(f"Unsupported initializer: {identifier!r}")


__all__ = [
    "Initializer",
    "deserialize",
    "get",
    "serialize",
    "Constant",
    "Zeros",
    "Ones",
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

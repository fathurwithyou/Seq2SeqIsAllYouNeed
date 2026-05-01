from .adam import Adam
from .base_optimizer import Optimizer
from .sgd import SGD


_OBJECTS = {
    "optimizer": Optimizer,
    "sgd": SGD,
    "adam": Adam,
}


def serialize(optimizer: Optimizer) -> dict:
    if not isinstance(optimizer, Optimizer):
        raise TypeError(f"Expected Optimizer, got {type(optimizer).__name__}")
    return {
        "class_name": type(optimizer).__name__,
        "config": optimizer.get_config(),
    }


def deserialize(config: dict, custom_objects: dict | None = None) -> Optimizer:
    if not isinstance(config, dict) or "class_name" not in config:
        raise ValueError("Optimizer config must contain 'class_name'.")
    objects = dict(_OBJECTS)
    if custom_objects:
        objects.update({name.lower(): obj for name, obj in custom_objects.items()})
    class_name = str(config["class_name"]).lower()
    if class_name not in objects:
        raise ValueError(f"Unknown optimizer: {config['class_name']!r}")
    return objects[class_name].from_config(config.get("config", {}))


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, Optimizer):
        return identifier
    if isinstance(identifier, str):
        return deserialize({"class_name": identifier, "config": {}})
    if isinstance(identifier, dict):
        return deserialize(identifier)
    raise ValueError(f"Could not interpret optimizer identifier: {identifier!r}")


__all__ = [
    "Optimizer",
    "SGD",
    "Adam",
    "deserialize",
    "get",
    "serialize",
]

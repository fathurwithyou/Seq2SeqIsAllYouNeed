from .loss import Loss, squeeze_or_expand_to_same_rank
from .losses import (
    CategoricalCrossentropy,
    LossFunctionWrapper,
    SparseCategoricalCrossentropy,
    categorical_crossentropy,
    sparse_categorical_crossentropy,
)


ALL_OBJECTS = {
    "CategoricalCrossentropy": CategoricalCrossentropy,
    "SparseCategoricalCrossentropy": SparseCategoricalCrossentropy,
    "categorical_crossentropy": CategoricalCrossentropy,
    "sparse_categorical_crossentropy": SparseCategoricalCrossentropy,
    "categoricalcrossentropy": CategoricalCrossentropy,
    "sparsecategoricalcrossentropy": SparseCategoricalCrossentropy,
}


def serialize(loss):
    if loss is None:
        return None
    loss = get(loss)
    if isinstance(loss, Loss):
        return {"class_name": loss.__class__.__name__, "config": loss.get_config()}
    return loss


def deserialize(config, custom_objects=None):
    if config is None:
        return None
    if isinstance(config, str):
        return get(config)
    custom_objects = custom_objects or {}
    class_name = config.get("class_name")
    loss_config = config.get("config", {})
    cls = custom_objects.get(class_name) or ALL_OBJECTS.get(class_name)
    if cls is None:
        raise ValueError(f"Unknown loss: {class_name!r}")
    return cls.from_config(loss_config)


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, Loss):
        return identifier
    if isinstance(identifier, dict):
        return deserialize(identifier)
    if isinstance(identifier, str):
        cls = ALL_OBJECTS.get(identifier) or ALL_OBJECTS.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Unsupported loss: {identifier!r}")
        return cls()
    if isinstance(identifier, type) and issubclass(identifier, Loss):
        return identifier()
    if callable(identifier):
        return identifier
    raise ValueError(f"Unsupported loss: {identifier!r}")


__all__ = [
    "Loss",
    "LossFunctionWrapper",
    "squeeze_or_expand_to_same_rank",
    "CategoricalCrossentropy",
    "SparseCategoricalCrossentropy",
    "categorical_crossentropy",
    "sparse_categorical_crossentropy",
    "deserialize",
    "get",
    "serialize",
]

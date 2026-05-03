_KERAS_EXPORTS = {"KerasLocallyConnected2D", "build_keras_cnn"}


def __getattr__(name: str):
    if name in _KERAS_EXPORTS:
        from . import keras_reference

        return getattr(keras_reference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["KerasLocallyConnected2D", "build_keras_cnn"]

_KERAS_EXPORTS = {"KerasLocallyConnected2D", "build_keras_cnn"}
_VISUALIZATION_EXPORTS = {
    "colourise_heatmap",
    "compute_grad_cam",
    "feature_layer_names",
    "make_feature_grid",
    "normalise_map",
    "overlay_heatmap",
    "predict_feature_maps",
    "save_array_image",
}


def __getattr__(name: str):
    if name in _KERAS_EXPORTS:
        from . import keras_reference

        return getattr(keras_reference, name)
    if name in _VISUALIZATION_EXPORTS:
        from . import visualization

        return getattr(visualization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = sorted(_KERAS_EXPORTS | _VISUALIZATION_EXPORTS)

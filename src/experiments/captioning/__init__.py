from .data import FLICKR8K_KAGGLE_DATASET, download_flickr8k_dataset, load_features
from .decoder import CaptioningDecoder, beam_search_decode

_KERAS_EXPORTS = {
    "ComparisonResult",
    "build_keras_captioning_decoder",
    "compare_captioning_decoder_outputs",
    "load_captioning_decoder_from_keras",
}


def __getattr__(name: str):
    if name in _KERAS_EXPORTS:
        from . import keras_reference

        return getattr(keras_reference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CaptioningDecoder",
    "FLICKR8K_KAGGLE_DATASET",
    "beam_search_decode",
    "ComparisonResult",
    "build_keras_captioning_decoder",
    "compare_captioning_decoder_outputs",
    "download_flickr8k_dataset",
    "load_captioning_decoder_from_keras",
    "load_features",
]

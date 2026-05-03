from __future__ import annotations

from typing import Any
import tensorflow as tf


def _to_pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"{name} must be an int or a pair of ints")
    return (int(value[0]), int(value[1]))


class KerasLocallyConnected2D:

    def __new__(cls, *args, **kwargs):

        class _Layer(tf.keras.layers.Layer):
            def __init__(
                self,
                filters: int,
                kernel_size: int | tuple[int, int],
                strides: int | tuple[int, int] = 1,
                padding: str = "valid",
                activation: str | None = None,
                use_bias: bool = True,
                kernel_initializer: str = "glorot_uniform",
                bias_initializer: str = "zeros",
                **layer_kwargs: Any,
            ) -> None:
                super().__init__(**layer_kwargs)
                self.filters = int(filters)
                self.kernel_size = _to_pair(kernel_size, "kernel_size")
                self.strides = _to_pair(strides, "strides")
                self.padding = str(padding).lower()
                if self.padding not in {"valid", "same"}:
                    raise ValueError("padding must be 'valid' or 'same'")
                self.activation = tf.keras.activations.get(activation)
                self.activation_identifier = activation
                self.use_bias = bool(use_bias)
                self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
                self.bias_initializer = tf.keras.initializers.get(bias_initializer)

            def build(self, input_shape) -> None:
                _, height, width, channels = input_shape
                if height is None or width is None or channels is None:
                    raise ValueError(
                        "KerasLocallyConnected2D requires static H, W, and C dimensions."
                    )
                kh, kw = self.kernel_size
                sh, sw = self.strides
                if self.padding == "same":
                    out_h = (int(height) + sh - 1) // sh
                    out_w = (int(width) + sw - 1) // sw
                else:
                    out_h = (int(height) - kh) // sh + 1
                    out_w = (int(width) - kw) // sw + 1
                if out_h <= 0 or out_w <= 0:
                    raise ValueError(
                        f"Invalid output size {(out_h, out_w)} for input_shape={input_shape}"
                    )
                self.output_rows = out_h
                self.output_cols = out_w
                patch_dim = kh * kw * int(channels)
                self.kernel = self.add_weight(
                    name="kernel",
                    shape=(out_h * out_w, patch_dim, self.filters),
                    initializer=self.kernel_initializer,
                    trainable=True,
                )
                if self.use_bias:
                    self.bias = self.add_weight(
                        name="bias",
                        shape=(out_h, out_w, self.filters),
                        initializer=self.bias_initializer,
                        trainable=True,
                    )
                else:
                    self.bias = None
                super().build(input_shape)

            def call(self, inputs):
                import tensorflow as tf

                kh, kw = self.kernel_size
                sh, sw = self.strides
                patches = tf.image.extract_patches(
                    inputs,
                    sizes=[1, kh, kw, 1],
                    strides=[1, sh, sw, 1],
                    rates=[1, 1, 1, 1],
                    padding=self.padding.upper(),
                )
                batch = tf.shape(inputs)[0]
                patches = tf.reshape(
                    patches,
                    (batch, self.output_rows * self.output_cols, kh * kw * inputs.shape[-1]),
                )
                outputs = tf.einsum("bpk,pkf->bpf", patches, self.kernel)
                outputs = tf.reshape(
                    outputs,
                    (batch, self.output_rows, self.output_cols, self.filters),
                )
                if self.bias is not None:
                    outputs = outputs + self.bias
                if self.activation is not None:
                    outputs = self.activation(outputs)
                return outputs

            def compute_output_shape(self, input_shape):
                return (input_shape[0], self.output_rows, self.output_cols, self.filters)

            def get_config(self) -> dict[str, Any]:
                config = super().get_config()
                config.update(
                    {
                        "filters": self.filters,
                        "kernel_size": self.kernel_size,
                        "strides": self.strides,
                        "padding": self.padding,
                        "activation": tf.keras.activations.serialize(self.activation),
                        "use_bias": self.use_bias,
                        "kernel_initializer": tf.keras.initializers.serialize(
                            self.kernel_initializer
                        ),
                        "bias_initializer": tf.keras.initializers.serialize(
                            self.bias_initializer
                        ),
                    }
                )
                return config

        _Layer.__name__ = cls.__name__
        return _Layer(*args, **kwargs)


def build_keras_cnn(
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
    layer_type: str,
    num_conv_layers: int,
    filters: list[int],
    kernel_size: int,
    pooling: str,
    dense_units: int,
):
    import tensorflow as tf

    if layer_type == "conv2d":
        conv_cls = tf.keras.layers.Conv2D
        padding = "same"
    elif layer_type == "locally_connected":
        conv_cls = KerasLocallyConnected2D
        padding = "valid"
    else:
        raise ValueError(f"Unknown layer_type: {layer_type!r}")

    if pooling == "max":
        pool_cls = tf.keras.layers.MaxPooling2D
    elif pooling == "average":
        pool_cls = tf.keras.layers.AveragePooling2D
    else:
        raise ValueError(f"Unknown pooling: {pooling!r}")

    if len(filters) == 1:
        filters = filters * num_conv_layers
    if len(filters) != num_conv_layers:
        raise ValueError("--filters must be length 1 or num_conv_layers")

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for index in range(num_conv_layers):
        x = conv_cls(
            filters[index],
            kernel_size=kernel_size,
            padding=padding,
            activation="relu",
            name=f"{layer_type}_{index}",
        )(x)
        x = pool_cls(pool_size=2, name=f"pool_{index}")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="head")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs, outputs, name=f"cnn_{layer_type}")

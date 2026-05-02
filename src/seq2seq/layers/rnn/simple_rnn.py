from __future__ import annotations

import numpy as np

from ...ops.rnn import simple_rnn_cell
from ...tensor import Tensor
from ..layer import Layer


class SimpleRNNCell(Layer):
    def __init__(
        self,
        units: int,
        *,
        activation: str = "tanh",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        input_dim: int | None = None,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.units = int(units)
        self.activation = activation
        self.use_bias = bool(use_bias)
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.seed = seed
        self.kernel = None
        self.recurrent_kernel = None
        self.bias = None
        if input_dim is not None:
            self.build((None, input_dim))

    def build(self, input_shape) -> None:
        input_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            (input_dim, self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            seed=self.seed,
        )
        recurrent_seed = None if self.seed is None else self.seed + 1
        self.recurrent_kernel = self.add_weight(
            (self.units, self.units),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
            seed=recurrent_seed,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                (self.units,),
                initializer=self.bias_initializer,
                name="bias",
            )
        self.built = True

    def call(self, inputs, states):
        previous = states.data if isinstance(states, Tensor) else np.asarray(states)
        array = inputs.data if isinstance(inputs, Tensor) else np.asarray(inputs)
        outputs = simple_rnn_cell(
            array,
            previous,
            self.kernel.data,
            self.recurrent_kernel.data,
            self.bias.data if self.bias is not None else None,
            activation=self.activation,
        )
        return Tensor(outputs) if isinstance(inputs, Tensor) else outputs

    def extra_repr(self) -> str:
        return f"units={self.units}, activation={self.activation!r}"

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": self.activation,
                "use_bias": self.use_bias,
                "kernel_initializer": self.kernel_initializer,
                "recurrent_initializer": self.recurrent_initializer,
                "bias_initializer": self.bias_initializer,
                "seed": self.seed,
            }
        )
        return config


class SimpleRNN(Layer):
    def __init__(
        self,
        units: int,
        *,
        activation: str = "tanh",
        use_bias: bool = True,
        return_sequences: bool = False,
        return_state: bool = False,
        num_layers: int = 1,
        input_dim: int | None = None,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.units = int(units)
        self.activation = activation
        self.use_bias = bool(use_bias)
        self.return_sequences = bool(return_sequences)
        self.return_state = bool(return_state)
        self.num_layers = int(num_layers)
        self.seed = seed
        self.cells: list[SimpleRNNCell] = []
        if input_dim is not None:
            self.build((None, None, input_dim))

    def build(self, input_shape) -> None:
        input_dim = int(input_shape[-1])
        self.cells = []
        for index in range(self.num_layers):
            cell = SimpleRNNCell(
                self.units,
                activation=self.activation,
                use_bias=self.use_bias,
                input_dim=input_dim,
                seed=None if self.seed is None else self.seed + index,
                name=f"simple_rnn_cell_{index}",
            )
            self.cells.append(cell)
            self.add_layer(f"cell_{index}", cell)
            input_dim = self.units
        self.built = True

    def call(self, inputs, initial_state=None):
        array = (
            inputs.data if isinstance(inputs, Tensor) else np.asarray(inputs, dtype=np.float32)
        )
        if array.ndim != 3:
            raise ValueError(f"SimpleRNN expects (N, T, D); got {array.shape}")
        batch_size, timesteps, _ = array.shape
        if initial_state is None:
            states = [np.zeros((batch_size, self.units), dtype=array.dtype) for _ in self.cells]
        else:
            state_array = (
                initial_state.data
                if isinstance(initial_state, Tensor)
                else np.asarray(initial_state)
            )
            if state_array.ndim == 2:
                states = [state_array.copy() for _ in self.cells]
            elif state_array.ndim == 3 and state_array.shape[0] == self.num_layers:
                states = [state_array[index].copy() for index in range(self.num_layers)]
            else:
                raise ValueError(
                    f"initial_state must be (N,H) or (L,N,H); got {state_array.shape}"
                )

        outputs = np.empty((batch_size, timesteps, self.units), dtype=array.dtype)
        for step in range(timesteps):
            layer_inputs = array[:, step, :]
            for index, cell in enumerate(self.cells):
                states[index] = simple_rnn_cell(
                    layer_inputs,
                    states[index],
                    cell.kernel.data,
                    cell.recurrent_kernel.data,
                    cell.bias.data if cell.bias is not None else None,
                    activation=cell.activation,
                )
                layer_inputs = states[index]
            outputs[:, step, :] = layer_inputs

        primary = outputs if self.return_sequences else outputs[:, -1, :]
        if self.return_state:
            return primary, np.stack(states, axis=0)
        return primary

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": self.activation,
                "use_bias": self.use_bias,
                "return_sequences": self.return_sequences,
                "return_state": self.return_state,
                "num_layers": self.num_layers,
                "seed": self.seed,
            }
        )
        return config


__all__ = ["SimpleRNN", "SimpleRNNCell"]

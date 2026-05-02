from __future__ import annotations

import numpy as np

from ...ops.rnn import lstm_cell
from ...tensor import Tensor
from ..layer import Layer


class LSTMCell(Layer):
    def __init__(
        self,
        units: int,
        *,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        unit_forget_bias: bool = True,
        input_dim: int | None = None,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.units = int(units)
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = bool(use_bias)
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.unit_forget_bias = bool(unit_forget_bias)
        self.seed = seed
        self.kernel = None
        self.recurrent_kernel = None
        self.bias = None
        if input_dim is not None:
            self.build((None, input_dim))

    def build(self, input_shape) -> None:
        input_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            (input_dim, 4 * self.units),
            initializer=self.kernel_initializer,
            name="kernel",
            seed=self.seed,
        )
        recurrent_seed = None if self.seed is None else self.seed + 1
        self.recurrent_kernel = self.add_weight(
            (self.units, 4 * self.units),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
            seed=recurrent_seed,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                (4 * self.units,),
                initializer=self.bias_initializer,
                name="bias",
            )
            if self.unit_forget_bias:
                self.bias.data[self.units : 2 * self.units] = 1.0
        self.built = True

    def call(self, inputs, states):
        previous_h, previous_c = states
        array = inputs.data if isinstance(inputs, Tensor) else np.asarray(inputs)
        h_prev = previous_h.data if isinstance(previous_h, Tensor) else np.asarray(previous_h)
        c_prev = previous_c.data if isinstance(previous_c, Tensor) else np.asarray(previous_c)
        hidden, cell = lstm_cell(
            array,
            h_prev,
            c_prev,
            self.kernel.data,
            self.recurrent_kernel.data,
            self.bias.data if self.bias is not None else None,
            activation=self.activation,
            recurrent_activation=self.recurrent_activation,
        )
        if isinstance(inputs, Tensor):
            return Tensor(hidden), (Tensor(hidden), Tensor(cell))
        return hidden, (hidden, cell)

    def extra_repr(self) -> str:
        return f"units={self.units}"

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": self.activation,
                "recurrent_activation": self.recurrent_activation,
                "use_bias": self.use_bias,
                "kernel_initializer": self.kernel_initializer,
                "recurrent_initializer": self.recurrent_initializer,
                "bias_initializer": self.bias_initializer,
                "unit_forget_bias": self.unit_forget_bias,
                "seed": self.seed,
            }
        )
        return config


class LSTM(Layer):
    def __init__(
        self,
        units: int,
        *,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
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
        self.recurrent_activation = recurrent_activation
        self.use_bias = bool(use_bias)
        self.return_sequences = bool(return_sequences)
        self.return_state = bool(return_state)
        self.num_layers = int(num_layers)
        self.seed = seed
        self.cells: list[LSTMCell] = []
        if input_dim is not None:
            self.build((None, None, input_dim))

    def build(self, input_shape) -> None:
        input_dim = int(input_shape[-1])
        self.cells = []
        for index in range(self.num_layers):
            cell = LSTMCell(
                self.units,
                activation=self.activation,
                recurrent_activation=self.recurrent_activation,
                use_bias=self.use_bias,
                input_dim=input_dim,
                seed=None if self.seed is None else self.seed + index,
                name=f"lstm_cell_{index}",
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
            raise ValueError(f"LSTM expects (N, T, D); got {array.shape}")
        batch_size, timesteps, _ = array.shape
        if initial_state is None:
            hidden_states = [np.zeros((batch_size, self.units), dtype=array.dtype) for _ in self.cells]
            cell_states = [np.zeros((batch_size, self.units), dtype=array.dtype) for _ in self.cells]
        else:
            h0, c0 = initial_state
            h0 = h0.data if isinstance(h0, Tensor) else np.asarray(h0)
            c0 = c0.data if isinstance(c0, Tensor) else np.asarray(c0)
            if h0.ndim == 2:
                hidden_states = [h0.copy() for _ in self.cells]
                cell_states = [c0.copy() for _ in self.cells]
            elif h0.ndim == 3 and h0.shape[0] == self.num_layers:
                hidden_states = [h0[index].copy() for index in range(self.num_layers)]
                cell_states = [c0[index].copy() for index in range(self.num_layers)]
            else:
                raise ValueError(
                    f"initial_state must be ((N,H),(N,H)) or ((L,N,H),(L,N,H)); got {h0.shape}"
                )

        outputs = np.empty((batch_size, timesteps, self.units), dtype=array.dtype)
        for step in range(timesteps):
            layer_inputs = array[:, step, :]
            for index, cell in enumerate(self.cells):
                hidden_states[index], cell_states[index] = lstm_cell(
                    layer_inputs,
                    hidden_states[index],
                    cell_states[index],
                    cell.kernel.data,
                    cell.recurrent_kernel.data,
                    cell.bias.data if cell.bias is not None else None,
                    activation=cell.activation,
                    recurrent_activation=cell.recurrent_activation,
                )
                layer_inputs = hidden_states[index]
            outputs[:, step, :] = layer_inputs

        primary = outputs if self.return_sequences else outputs[:, -1, :]
        if self.return_state:
            return primary, np.stack(hidden_states, axis=0), np.stack(cell_states, axis=0)
        return primary

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": self.activation,
                "recurrent_activation": self.recurrent_activation,
                "use_bias": self.use_bias,
                "return_sequences": self.return_sequences,
                "return_state": self.return_state,
                "num_layers": self.num_layers,
                "seed": self.seed,
            }
        )
        return config


__all__ = ["LSTM", "LSTMCell"]

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .. import losses as losses_module
from .. import optimizers as optimizers_module
from ..callbacks import Callback, CallbackList
from ..metrics import accuracy
from ..tensor import Tensor, no_grad, to_numpy


class History(dict):

    @property
    def history(self) -> "History":
        return self


class Trainer:

    def compile(
        self,
        optimizer: Any = "adam",
        loss: Any = None,
        metrics: Sequence[str] | None = None,
        *,
        loss_weights: Any = None,
        weighted_metrics: Sequence[str] | None = None,
        run_eagerly: bool = False,
        steps_per_execution: int = 1,
        auto_scale_loss: bool = True,
    ) -> None:
        def resolve_loss(value):
            if value is None:
                return None
            return losses_module.get(value)

        def resolve_metrics(values):
            if values is None:
                return []
            resolved = []
            for item in values:
                if isinstance(item, str):
                    resolved.append(item.lower())
                else:
                    resolved.append(item)
            return resolved

        self._optimizer_spec = optimizer
        self.optimizer = self._build_optimizer_if_possible(optimizer)
        self.loss = resolve_loss(loss)
        self.loss_weights = loss_weights
        self.metrics = resolve_metrics(metrics)
        self.weighted_metrics = resolve_metrics(weighted_metrics)
        self.run_eagerly = bool(run_eagerly)
        self.steps_per_execution = int(steps_per_execution)
        self.auto_scale_loss = bool(auto_scale_loss)
        self.compiled = True

    def _build_optimizer_if_possible(self, spec):
        if spec is None:
            return None
        params = getattr(self, "trainable_weights", None) or getattr(self, "weights", None)
        if isinstance(spec, str):
            if not params:
                return None
            name = spec.lower()
            if name == "adam":
                return optimizers_module.Adam(params)
            if name == "sgd":
                return optimizers_module.SGD(params)
            raise ValueError(f"Unsupported optimizer: {spec!r}")
        if not getattr(spec, "params", None) and params and hasattr(spec, "build"):
            spec.build(params)
        return spec

    def _ensure_optimizer(self):
        if getattr(self, "optimizer", None) is not None:
            return self.optimizer
        spec = getattr(self, "_optimizer_spec", None)
        if spec is None:
            raise RuntimeError("Model is not compiled; call compile() first.")
        optimizer = self._build_optimizer_if_possible(spec)
        if optimizer is None:
            raise RuntimeError(
                "Model has no trainable weights yet; build the model by calling it on "
                "a sample input before fit()."
            )
        self.optimizer = optimizer
        return optimizer

    def _forward(self, *inputs):
        if len(inputs) == 1:
            return self(inputs[0])
        return self(*inputs)

    def predict(
        self,
        x,
        *,
        batch_size: int = 32,
        verbose: int = 0,
    ) -> np.ndarray:
        was_training = getattr(self, "training", True)
        self.eval()
        arrays = _as_tuple_of_arrays(x)
        total = arrays[0].shape[0]
        outputs: list[np.ndarray] = []
        try:
            with no_grad():
                for start in range(0, total, batch_size):
                    end = min(start + batch_size, total)
                    batch = tuple(array[start:end] for array in arrays)
                    result = self._forward(*batch)
                    outputs.append(to_numpy(result))
                    if verbose:
                        print(f"predict {end}/{total}")
        finally:
            self.train(was_training)
        return np.concatenate(outputs, axis=0)

    def evaluate(
        self,
        x,
        y,
        *,
        batch_size: int = 32,
        verbose: int = 0,
        return_dict: bool = True,
    ):
        if getattr(self, "loss", None) is None:
            raise RuntimeError("evaluate() requires a compiled loss; call compile(loss=...) first.")
        was_training = getattr(self, "training", True)
        self.eval()
        arrays = _as_tuple_of_arrays(x)
        targets = np.asarray(y)
        total = arrays[0].shape[0]
        if targets.shape[0] != total:
            raise ValueError("x and y must share the same number of samples")
        metric_fns = _build_metric_fns(getattr(self, "metrics", None))

        loss_sum = 0.0
        metric_sums: dict[str, float] = {name: 0.0 for name in metric_fns}
        count = 0
        try:
            with no_grad():
                for start in range(0, total, batch_size):
                    end = min(start + batch_size, total)
                    batch = tuple(array[start:end] for array in arrays)
                    y_batch = targets[start:end]
                    preds = self._forward(*batch)
                    loss_value = float(to_numpy(self.loss(y_batch, preds)).mean())
                    size = end - start
                    loss_sum += loss_value * size
                    preds_np = to_numpy(preds)
                    for name, fn in metric_fns.items():
                        metric_sums[name] += fn(preds_np, y_batch) * size
                    count += size
                    if verbose:
                        print(f"evaluate {end}/{total} loss={loss_value:.4f}")
        finally:
            self.train(was_training)
        result = {"loss": loss_sum / max(count, 1)}
        for name in metric_fns:
            result[name] = metric_sums[name] / max(count, 1)
        return result if return_dict else [result["loss"], *[result[n] for n in metric_fns]]

    def fit(
        self,
        x,
        y,
        *,
        batch_size: int = 32,
        epochs: int = 1,
        verbose: int = 1,
        validation_data: tuple | None = None,
        shuffle: bool = True,
        seed: int | None = None,
        callbacks: Sequence[Callback] | None = None,
    ) -> dict[str, list[float]]:
        if getattr(self, "loss", None) is None:
            raise RuntimeError("fit() requires a compiled loss; call compile(loss=...) first.")
        arrays = _as_tuple_of_arrays(x)
        targets = np.asarray(y)
        total = arrays[0].shape[0]
        if targets.shape[0] != total:
            raise ValueError("x and y must share the same number of samples")

        self._build_with_sample(arrays)
        optimizer = self._ensure_optimizer()
        metric_fns = _build_metric_fns(getattr(self, "metrics", None))
        callback_list = CallbackList(callbacks)
        callback_list.set_model(self)

        history: History = History({"loss": []})
        for name in metric_fns:
            history[name] = []
        if validation_data is not None:
            history["val_loss"] = []
            for name in metric_fns:
                history[f"val_{name}"] = []

        rng = np.random.default_rng(seed)
        gradient_seen = False
        callback_list.on_train_begin()

        for epoch in range(epochs):
            callback_list.on_epoch_begin(epoch)
            indices = np.arange(total)
            if shuffle:
                rng.shuffle(indices)
            self.train()
            loss_sum = 0.0
            metric_sums = {name: 0.0 for name in metric_fns}
            count = 0
            for batch_index, start in enumerate(range(0, total, batch_size)):
                batch_idx = indices[start : start + batch_size]
                batch = tuple(_wrap_tensor(array[batch_idx]) for array in arrays)
                y_batch = targets[batch_idx]
                callback_list.on_batch_begin(batch_index)
                preds = self._forward(*batch)
                preds_tensor = preds if isinstance(preds, Tensor) else Tensor(np.asarray(preds))
                loss_tensor = self.loss(y_batch, preds_tensor)
                if not isinstance(loss_tensor, Tensor):
                    loss_tensor = Tensor(np.asarray(loss_tensor))
                if loss_tensor.requires_grad:
                    loss_tensor.backward()
                    optimizer.apply_gradients(
                        (weight.grad, weight)
                        for weight in getattr(self, "trainable_weights", [])
                    )
                    gradient_seen = True
                loss_value = float(loss_tensor.data.mean())
                size = batch_idx.shape[0]
                loss_sum += loss_value * size
                preds_np = to_numpy(preds_tensor)
                for name, fn in metric_fns.items():
                    metric_sums[name] += fn(preds_np, y_batch) * size
                count += size
                callback_list.on_batch_end(batch_index, {"loss": loss_value})

            epoch_loss = loss_sum / max(count, 1)
            history["loss"].append(epoch_loss)
            for name in metric_fns:
                history[name].append(metric_sums[name] / max(count, 1))

            log = {"epoch": epoch + 1, "loss": epoch_loss}
            for name in metric_fns:
                log[name] = history[name][-1]
            if validation_data is not None:
                x_val, y_val = validation_data
                val = self.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)
                history["val_loss"].append(val["loss"])
                log["val_loss"] = val["loss"]
                for name in metric_fns:
                    history[f"val_{name}"].append(val[name])
                    log[f"val_{name}"] = val[name]
            if verbose:
                summary = " - ".join(
                    f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in log.items()
                )
                print(f"Epoch {epoch + 1}/{epochs} - {summary}")
            callback_list.on_epoch_end(epoch, log)

        callback_list.on_train_end()
        if not gradient_seen and epochs > 0:
            raise RuntimeError(
                "fit() did not receive gradients from any layer. The scratch "
                "Conv2D / RNN / LSTM layers in seq2seq are forward-only; train "
                "with Keras (see scripts/train_cnn.py, scripts/train_caption.py) "
                "and load the weights into the scratch model for inference."
            )
        return history

    def _build_with_sample(self, arrays: tuple[np.ndarray, ...]) -> None:
        sample = tuple(array[:1] for array in arrays)
        with no_grad():
            self._forward(*sample)


def _as_tuple_of_arrays(x) -> tuple[np.ndarray, ...]:
    if isinstance(x, (list, tuple)):
        arrays = tuple(np.asarray(item) for item in x)
    else:
        arrays = (np.asarray(x),)
    if not arrays:
        raise ValueError("x must contain at least one input array")
    first_dim = arrays[0].shape[0]
    for array in arrays:
        if array.shape[0] != first_dim:
            raise ValueError("all input arrays must share the same number of samples")
    return arrays


def _wrap_tensor(array: np.ndarray):
    if array.dtype.kind in ("i", "u", "b"):
        return array
    return Tensor(array, requires_grad=False)


def _accuracy_from_preds(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return accuracy(y_true, y_pred)


def _build_metric_fns(metrics) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    if not metrics:
        return {}
    mapping: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {}
    for metric in metrics:
        if isinstance(metric, str):
            name = metric.lower()
            if name in {"accuracy", "acc"}:
                mapping["accuracy"] = _accuracy_from_preds
                continue
            if name in {"sparse_categorical_accuracy"}:
                mapping["sparse_categorical_accuracy"] = _accuracy_from_preds
                continue
            raise ValueError(f"Unsupported metric: {metric!r}")
        name = getattr(metric, "__name__", type(metric).__name__)
        mapping[name] = metric
    return mapping


__all__ = ["History", "Trainer"]

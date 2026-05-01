from __future__ import annotations

from typing import Any, Sequence


class Callback:
    def set_model(self, model) -> None:
        self.model = model

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        pass

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        pass

    def on_batch_begin(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        pass

    def on_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        pass


class CallbackList:
    def __init__(self, callbacks: Sequence[Callback] | None = None) -> None:
        self.callbacks: list[Callback] = list(callbacks or [])

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def set_model(self, model) -> None:
        for callback in self.callbacks:
            callback.set_model(model)

    def _broadcast(self, name: str, *args, **kwargs) -> None:
        for callback in self.callbacks:
            method = getattr(callback, name, None)
            if method is not None:
                method(*args, **kwargs)

    def on_train_begin(self, logs=None):
        self._broadcast("on_train_begin", logs)

    def on_train_end(self, logs=None):
        self._broadcast("on_train_end", logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._broadcast("on_epoch_begin", epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self._broadcast("on_epoch_end", epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        self._broadcast("on_batch_begin", batch, logs)

    def on_batch_end(self, batch, logs=None):
        self._broadcast("on_batch_end", batch, logs)


__all__ = ["Callback", "CallbackList"]

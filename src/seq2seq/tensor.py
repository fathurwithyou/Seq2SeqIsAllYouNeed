
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

_DEFAULT_DTYPE = np.float32


class _GradMode:

    enabled: bool = True


@contextmanager
def no_grad() -> Iterator[None]:
    prev = _GradMode.enabled
    _GradMode.enabled = False
    try:
        yield
    finally:
        _GradMode.enabled = prev


@contextmanager
def enable_grad() -> Iterator[None]:
    prev = _GradMode.enabled
    _GradMode.enabled = True
    try:
        yield
    finally:
        _GradMode.enabled = prev


def is_grad_enabled() -> bool:
    return _GradMode.enabled


def _as_array(data: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(data, Tensor):
        arr = data.data
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        arr = np.asarray(data)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    elif arr.dtype.kind == "f" and arr.dtype != _DEFAULT_DTYPE:
        arr = arr.astype(_DEFAULT_DTYPE, copy=False)
    elif arr.dtype.kind not in ("f", "i", "u", "b"):
        arr = arr.astype(_DEFAULT_DTYPE, copy=False)
    return arr


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    g = grad
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, (g_dim, s_dim) in enumerate(zip(g.shape, shape)):
        if s_dim == 1 and g_dim != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g.reshape(shape)


class Tensor:

    __array_priority__ = 1000
    __slots__ = ("data", "requires_grad", "grad", "_backward", "_prev", "_op")

    def __init__(
        self,
        data: Any,
        *,
        requires_grad: bool = False,
        dtype: np.dtype | None = None,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
    ) -> None:
        self.data: np.ndarray = _as_array(data, dtype)
        self.requires_grad: bool = bool(requires_grad) and is_grad_enabled()
        self.grad: np.ndarray | None = None
        self._backward = _noop
        self._prev: tuple[Tensor, ...] = tuple(_children) if _children else ()
        self._op = _op


    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        return int(self.data.ndim)

    @property
    def size(self) -> int:
        return int(self.data.size)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def item(self) -> float:
        if self.data.size != 1:
            raise ValueError("only a 1-element tensor can be converted to Python scalar")
        return float(self.data.reshape(()))

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        arr = self.data.astype(dtype, copy=False) if dtype is not None else self.data
        return arr.copy() if copy else arr

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator["Tensor"]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: Any) -> "Tensor":
        out_data = self.data[index]

        def backward():
            if out.grad is not None and self.requires_grad:
                grad = np.zeros_like(self.data)
                np.add.at(grad, index, out.grad)
                self._accumulate(grad)

        out = self._track(out_data, (self,), "slice", backward)
        return out

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def clone(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def zero_grad(self) -> None:
        self.grad = None


    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.data.dtype}, requires_grad={self.requires_grad})"


    @staticmethod
    def _coerce(x: Any) -> "Tensor":
        return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)

    def _track(self, data: np.ndarray, parents: Sequence["Tensor"], op: str, backward) -> "Tensor":
        track = is_grad_enabled() and any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=track, _children=parents if track else (), _op=op)
        if track:
            out._backward = backward
        return out

    def __add__(self, other: Any) -> "Tensor":
        o = self._coerce(other)
        out_data = self.data + o.data

        def backward():
            g = out.grad
            if g is None:
                return
            if self.requires_grad:
                self._accumulate(_unbroadcast(g, self.shape))
            if o.requires_grad:
                o._accumulate(_unbroadcast(g, o.shape))

        out = self._track(out_data, (self, o), "+", backward)
        return out

    __radd__ = __add__

    def __neg__(self) -> "Tensor":
        def backward():
            if out.grad is not None and self.requires_grad:
                self._accumulate(-out.grad)

        out = self._track(-self.data, (self,), "neg", backward)
        return out

    def __sub__(self, other: Any) -> "Tensor":
        return self + (-self._coerce(other))

    def __rsub__(self, other: Any) -> "Tensor":
        return self._coerce(other) + (-self)

    def __mul__(self, other: Any) -> "Tensor":
        o = self._coerce(other)
        out_data = self.data * o.data

        def backward():
            g = out.grad
            if g is None:
                return
            if self.requires_grad:
                self._accumulate(_unbroadcast(g * o.data, self.shape))
            if o.requires_grad:
                o._accumulate(_unbroadcast(g * self.data, o.shape))

        out = self._track(out_data, (self, o), "*", backward)
        return out

    __rmul__ = __mul__

    def __abs__(self) -> "Tensor":
        res = np.abs(self.data)

        def backward():
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * np.sign(self.data))

        out = self._track(res, (self,), "abs", backward)
        return out

    def __truediv__(self, other: Any) -> "Tensor":
        o = self._coerce(other)
        out_data = self.data / o.data

        def backward():
            g = out.grad
            if g is None:
                return
            if self.requires_grad:
                self._accumulate(_unbroadcast(g / o.data, self.shape))
            if o.requires_grad:
                o._accumulate(_unbroadcast(-g * self.data / (o.data ** 2), o.shape))

        out = self._track(out_data, (self, o), "/", backward)
        return out

    def __rtruediv__(self, other: Any) -> "Tensor":
        return self._coerce(other) / self

    def __pow__(self, exponent: float) -> "Tensor":
        def backward():
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * (exponent * self.data ** (exponent - 1.0)))

        out = self._track(self.data ** exponent, (self,), f"**{exponent}", backward)
        return out

    def __matmul__(self, other: Any) -> "Tensor":
        o = self._coerce(other)
        out_data = self.data @ o.data

        def backward():
            g = out.grad
            if g is None:
                return
            if self.requires_grad:
                self._accumulate(g @ np.swapaxes(o.data, -1, -2))
            if o.requires_grad:
                o._accumulate(np.swapaxes(self.data, -1, -2) @ g)

        out = self._track(out_data, (self, o), "@", backward)
        return out


    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out_data = self.data.sum(axis=axis, keepdims=keepdims)

        def backward():
            g = out.grad
            if g is None or not self.requires_grad:
                return
            if axis is None:
                grad = np.broadcast_to(g, self.shape).copy()
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                expanded = g if keepdims else np.expand_dims(g, axis=axes)
                grad = np.broadcast_to(expanded, self.shape).copy()
            self._accumulate(grad)

        out = self._track(out_data, (self,), "sum", backward)
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            denom = self.data.size
        elif isinstance(axis, tuple):
            denom = int(np.prod([self.shape[a] for a in axis]))
        else:
            denom = self.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / denom

    def log(self) -> "Tensor":
        def backward():
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad / self.data)

        out = self._track(np.log(self.data), (self,), "log", backward)
        return out

    def exp(self) -> "Tensor":
        res = np.exp(self.data)

        def backward():
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * res)

        out = self._track(res, (self,), "exp", backward)
        return out

    def tanh(self) -> "Tensor":
        res = np.tanh(self.data)

        def backward():
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * (1.0 - res * res))

        out = self._track(res, (self,), "tanh", backward)
        return out

    def sigmoid(self) -> "Tensor":
        res = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60.0, 60.0)))

        def backward():
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * res * (1.0 - res))

        out = self._track(res, (self,), "sigmoid", backward)
        return out

    def relu(self) -> "Tensor":
        mask = self.data > 0
        res = np.where(mask, self.data, 0.0)

        def backward():
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * mask)

        out = self._track(res, (self,), "relu", backward)
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        exp = np.exp(shifted)
        res = exp / exp.sum(axis=axis, keepdims=True)

        def backward():
            g = out.grad
            if g is None or not self.requires_grad:
                return
            self._accumulate(res * (g - (g * res).sum(axis=axis, keepdims=True)))

        out = self._track(res, (self,), "softmax", backward)
        return out


    def reshape(self, *shape: int) -> "Tensor":
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        old_shape = self.shape

        def backward():
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad.reshape(old_shape))

        out = self._track(self.data.reshape(new_shape), (self,), "reshape", backward)
        return out

    def transpose(self, *axes: int) -> "Tensor":
        perm = axes if axes else None

        def backward():
            if out.grad is not None and self.requires_grad:
                if perm is None:
                    self._accumulate(out.grad.transpose())
                else:
                    inv = np.argsort(perm)
                    self._accumulate(out.grad.transpose(inv))

        out = self._track(self.data.transpose(perm), (self,), "transpose", backward)
        return out

    def squeeze(self, axis: int | None = None) -> "Tensor":
        old_shape = self.shape

        def backward():
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad.reshape(old_shape))

        out = self._track(np.squeeze(self.data, axis=axis), (self,), "squeeze", backward)
        return out

    def _accumulate(self, grad_contribution: np.ndarray) -> None:
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad = self.grad + grad_contribution

    def backward(self, gradient: np.ndarray | None = None) -> None:
        if not self.requires_grad:
            return
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar outputs")
            gradient = np.ones_like(self.data)
        self._accumulate(gradient)

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        for node in reversed(topo):
            node._backward()


def _noop() -> None:
    return None


def tensor(data: Any, *, requires_grad: bool = False, dtype: np.dtype | None = None) -> Tensor:
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def to_numpy(x: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    arr = x.data if isinstance(x, Tensor) else np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def wrap_like(ref: Any, value: np.ndarray):
    return Tensor(value) if isinstance(ref, Tensor) else value


def zeros(*shape: int, requires_grad: bool = False, dtype: np.dtype = _DEFAULT_DTYPE) -> Tensor:
    s = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
    return Tensor(np.zeros(s, dtype=dtype), requires_grad=requires_grad)


def ones(*shape: int, requires_grad: bool = False, dtype: np.dtype = _DEFAULT_DTYPE) -> Tensor:
    s = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
    return Tensor(np.ones(s, dtype=dtype), requires_grad=requires_grad)


def concat(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    arrs = [t.data for t in tensors]
    out_data = np.concatenate(arrs, axis=axis)
    sizes = [a.shape[axis] for a in arrs]
    track = is_grad_enabled() and any(t.requires_grad for t in tensors)

    def backward():
        g = out.grad
        if g is None:
            return
        parts = np.split(g, np.cumsum(sizes)[:-1], axis=axis)
        for t, part in zip(tensors, parts):
            if t.requires_grad:
                t._accumulate(part)

    out = Tensor(out_data, requires_grad=track, _children=tensors if track else (), _op="concat")
    if track:
        out._backward = backward
    return out


def stack(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:

    expanded = [np.expand_dims(t.data, axis=axis) for t in tensors]
    out_data = np.concatenate(expanded, axis=axis)
    track = is_grad_enabled() and any(t.requires_grad for t in tensors)

    def backward():
        g = out.grad
        if g is None:
            return
        parts = np.split(g, len(tensors), axis=axis)
        for t, part in zip(tensors, parts):
            if t.requires_grad:
                t._accumulate(np.squeeze(part, axis=axis))

    out = Tensor(out_data, requires_grad=track, _children=tensors if track else (), _op="stack")
    if track:
        out._backward = backward
    return out

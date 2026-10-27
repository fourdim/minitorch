import builtins
from typing import Any

import numpy as np

from minitorch.autograd import is_grad_enabled


def _ensure_tensor(x, like: Tensor | None = None) -> Tensor:
    """Convert scalar or array to Tensor if needed."""
    if isinstance(x, Tensor):
        return x
    if like is not None and isinstance(x, (bool, int, float)):
        # keep python scalars "weak" so that float32 * 0.5 stays float32
        return Tensor(np.asarray(x, dtype=np.result_type(like.data, x)))
    return Tensor(x)


def _unwrap_key(key):
    """Replace Tensors inside an index key with their ndarrays."""
    if isinstance(key, Tensor):
        return key.data
    if isinstance(key, tuple):
        return tuple(_unwrap_key(k) for k in key)
    return key


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum `grad` over the dims that were broadcast to reach its shape."""
    if grad.shape == shape:
        return grad
    dims = compute_broadcast_dims(shape, grad.ndim)
    grad = grad.sum(dims, keepdims=True)
    return grad.reshape(shape)


def compute_broadcast_dims(shape: tuple[int, ...], ndim: int):
    padded = (1,) * (ndim - len(shape)) + shape
    return tuple(i for i, s in enumerate(padded) if s == 1)


class Tensor:
    def __init__(self, data: Any):
        if isinstance(data, Tensor):
            self.__dict__.update(data.__dict__)
            return

        if isinstance(data, np.ndarray):
            self.data: np.ndarray = data
        else:
            self.data = np.array(data)
            # like torch, python floats default to float32
            if self.data.dtype == np.float64:
                self.data = self.data.astype(np.float32)
        self.requires_grad = False
        self.grad: Tensor | None = None

        self._grad_fn = lambda: None
        self._children: tuple[Tensor, ...] = ()
        self._op = ""

    @staticmethod
    def _from_op(data: np.ndarray, children: tuple[Tensor, ...], op: str) -> Tensor:
        """Create the result of an op, recording the graph only if grad is required."""
        result = Tensor(data)
        # If any of children requires grad, then its result will require grad.
        # This is required for loss = a + b + c case
        # where a + b creates a intermediate tensor with `require_grad` default to false
        result.requires_grad = is_grad_enabled() and any(c.requires_grad for c in children)
        if result.requires_grad:
            # No need for computation graph if grad is not required.
            # Otherwise reference to children can be immediately dropped.
            result._children = children
            result._op = op
        return result

    @property
    def _grad_data(self) -> np.ndarray:
        # only read inside backward, after the grad has flowed into this tensor
        assert self.grad is not None
        return self.grad.data

    def _accumulate_grad(self, grad: np.ndarray):
        if not self.requires_grad:
            return
        if self.grad is None:
            # copy: `grad` may be a view of another tensor's data
            self.grad = Tensor(np.array(grad, dtype=self.dtype))
        else:
            self.grad.data += grad

    def __repr__(self):
        grad = ", requires_grad=True" if self.requires_grad else ""
        return f"tensor({self.data}{grad})"

    def __len__(self):
        return len(self.data)

    def __bool__(self):
        return bool(self.data)

    __hash__ = object.__hash__

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def size(self, dim: int | None = None):
        return self.shape if dim is None else self.shape[dim]

    def dim(self):
        return self.ndim

    def numel(self):
        return self.data.size

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def tolist(self):
        return self.data.tolist()

    def detach(self) -> Tensor:
        return Tensor(self.data)

    def to(self, dtype) -> Tensor:
        return Tensor(self.data.astype(dtype))

    def long(self) -> Tensor:
        return self.to(np.int64)

    def float(self) -> Tensor:
        return self.to(np.float32)

    def backward(self, gradient=None):
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "grad can be implicitly created only for scalar outputs"
                )
            gradient = Tensor(np.ones_like(self.data))
        self.grad = _ensure_tensor(gradient)
        visited = set()
        order: list[Tensor] = []

        def build_topo(t: Tensor):
            if t not in visited:
                visited.add(t)
                for child in t._children:
                    build_topo(child)
                order.append(t)

        build_topo(self)

        for t in reversed(order):
            # result grad must be calculated before its children
            if t.grad is not None:
                t._grad_fn()

    # ---------------------------------------------------------------- shape ops

    def reshape(self, *shape) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        result = Tensor._from_op(self.data.reshape(shape), (self,), "Reshape")

        def backward():
            self._accumulate_grad(result._grad_data.reshape(self.shape))

        result._grad_fn = backward
        return result

    view = reshape

    def transpose(self, dim0: int = 0, dim1: int = 1):
        result = Tensor._from_op(np.swapaxes(self.data, dim0, dim1), (self,), "T")

        def backward():
            self._accumulate_grad(np.swapaxes(result._grad_data, dim0, dim1))

        result._grad_fn = backward
        return result

    @property
    def T(self):
        return self.permute(*reversed(range(self.ndim)))

    def permute(self, *dims):
        result = Tensor._from_op(self.data.transpose(dims), (self,), "Permute")

        def backward():
            self._accumulate_grad(result._grad_data.transpose(np.argsort(dims)))

        result._grad_fn = backward
        return result

    def unsqueeze(self, dim: int):
        return self.reshape(np.expand_dims(self.data, dim).shape)

    def squeeze(self, dim: int):
        if self.shape[dim] != 1:
            return self
        return self.reshape(np.squeeze(self.data, dim).shape)

    def flatten(self, start_dim: int = 0, end_dim: int = -1):
        if end_dim < 0:
            end_dim += len(self.shape)
        new_shape = self.shape[:start_dim] + (-1,) + self.shape[end_dim + 1 :]
        return self.reshape(new_shape)

    def broadcast_to(self, shape: tuple[int, ...]) -> Tensor:
        if self.shape == shape:
            return self
        result = Tensor._from_op(np.broadcast_to(self.data, shape), (self,), "Broadcast")

        def backward():
            self._accumulate_grad(_unbroadcast(result._grad_data, self.shape))

        result._grad_fn = backward
        return result

    expand = broadcast_to

    def __getitem__(self, key):
        key = _unwrap_key(key)
        result = Tensor._from_op(self.data[key], (self,), "GetItem")

        def backward():
            grad_data = np.zeros_like(self.data)
            np.add.at(grad_data, key, result._grad_data)
            self._accumulate_grad(grad_data)

        result._grad_fn = backward
        return result

    def __setitem__(self, key, value):
        # in-place write, not tracked by autograd
        self.data[_unwrap_key(key)] = _ensure_tensor(value).data

    # ------------------------------------------------------------- reductions

    def sum(self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False):
        result = Tensor._from_op(np.asarray(self.data.sum(dim, keepdims=keepdim)), (self,), "Sum")

        def backward():
            rgrad = result._grad_data
            if not keepdim and dim is not None:
                rgrad = np.expand_dims(rgrad, dim)
            self._accumulate_grad(np.broadcast_to(rgrad, self.shape))

        result._grad_fn = backward
        return result

    def mean(self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False):
        total = self.sum(dim, keepdim)
        return total / (self.numel() // total.numel())

    def var(self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False, correction: int = 1):
        diff = self - self.mean(dim, keepdim=True)
        total = (diff * diff).sum(dim, keepdim)
        return total / (self.numel() // total.numel() - correction)

    def max(self, dim: int | None = None, keepdim: bool = False):
        result = Tensor._from_op(np.asarray(self.data.max(dim, keepdims=keepdim)), (self,), "Max")

        def backward():
            out = result.data if keepdim or dim is None else np.expand_dims(result.data, dim)
            grad = result._grad_data if keepdim or dim is None else np.expand_dims(result._grad_data, dim)
            mask = self.data == out
            # split the gradient evenly between ties
            self._accumulate_grad(mask * grad / mask.sum(dim, keepdims=True))

        result._grad_fn = backward
        return result

    def argmax(self, dim: int | None = None, keepdim: bool = False):
        return Tensor(np.asarray(np.argmax(self.data, axis=dim, keepdims=keepdim)))

    def all(self, dim: int | None = None):
        return Tensor(np.asarray(self.data.all(dim)))

    def any(self, dim: int | None = None):
        return Tensor(np.asarray(self.data.any(dim)))

    # ------------------------------------------------------------ comparisons

    def _compare(self, other, fn):
        other = _ensure_tensor(other, like=self)
        return Tensor(fn(self.data, other.data))

    def __eq__(self, other):
        return self._compare(other, np.equal)

    def __ne__(self, other):
        return self._compare(other, np.not_equal)

    def __lt__(self, other):
        return self._compare(other, np.less)

    def __le__(self, other):
        return self._compare(other, np.less_equal)

    def __gt__(self, other):
        return self._compare(other, np.greater)

    def __ge__(self, other):
        return self._compare(other, np.greater_equal)

    def __and__(self, other):
        return self._compare(other, np.logical_and)

    def __or__(self, other):
        return self._compare(other, np.logical_or)

    def __invert__(self):
        return Tensor(np.logical_not(self.data))

    # ------------------------------------------------------------- arithmetic

    def __add__(self, other) -> Tensor:
        other = _ensure_tensor(other, like=self)
        result = Tensor._from_op(self.data + other.data, (self, other), "+")

        def backward():
            # If self, other are not in the same shape, undo the broadcast
            self._accumulate_grad(_unbroadcast(result._grad_data, self.shape))
            other._accumulate_grad(_unbroadcast(result._grad_data, other.shape))

        result._grad_fn = backward
        return result

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = _ensure_tensor(other, like=self)
        return self + (-other)

    def __rsub__(self, other):
        other = _ensure_tensor(other, like=self)
        return other - self

    def __mul__(self, other) -> Tensor:
        other = _ensure_tensor(other, like=self)
        result = Tensor._from_op(self.data * other.data, (self, other), "*")

        def backward():
            self._accumulate_grad(_unbroadcast(other.data * result._grad_data, self.shape))
            other._accumulate_grad(_unbroadcast(self.data * result._grad_data, other.shape))

        result._grad_fn = backward
        return result

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = _ensure_tensor(other, like=self)
        if not other.requires_grad:
            # cheaper than building `other ** -1` in the graph
            return self * Tensor(1 / other.data)
        return self * (other**-1.0)

    def __rtruediv__(self, other):
        other = _ensure_tensor(other, like=self)
        return other / self

    def __pow__(self, other):
        other = _ensure_tensor(other, like=self)
        exponent = other.data
        if np.issubdtype(self.dtype, np.integer) and np.issubdtype(other.dtype, np.integer):
            # numpy refuses negative integer powers of integers
            exponent = exponent.astype(float)
        result = Tensor._from_op(self.data**exponent, (self, other), "**")

        def backward():
            g = result._grad_data
            self._accumulate_grad(_unbroadcast(g * exponent * self.data ** (exponent - 1), self.shape))
            if other.requires_grad:
                other._accumulate_grad(_unbroadcast(g * result.data * np.log(self.data), other.shape))

        result._grad_fn = backward
        return result

    def __rpow__(self, other):
        other = _ensure_tensor(other, like=self)
        return other**self

    def __matmul__(self, other: Tensor) -> Tensor:
        result = Tensor._from_op(self.data @ other.data, (self, other), "@")

        def backward():
            g = result._grad_data
            if self.requires_grad:
                self._accumulate_grad(_unbroadcast(g @ np.swapaxes(other.data, -1, -2), self.shape))
            if not other.requires_grad:
                return
            if other.ndim == 2 and self.ndim > 2:
                # (B, T, C) @ (C, H): fold the batch into rows, one big matmul
                # is much cheaper than a batched one followed by a sum
                a = self.data.reshape(-1, self.shape[-1])
                other._accumulate_grad(a.T @ g.reshape(-1, g.shape[-1]))
            else:
                # batched matmul broadcasts the leading dims, so undo that too
                other._accumulate_grad(_unbroadcast(np.swapaxes(self.data, -1, -2) @ g, other.shape))

        result._grad_fn = backward
        return result

    # --------------------------------------------------------- elementwise fns

    def log(self):
        result = Tensor._from_op(np.log(self.data), (self,), "log")

        def backward():
            self._accumulate_grad(result._grad_data / self.data)

        result._grad_fn = backward
        return result

    def exp(self):
        result = Tensor._from_op(np.exp(self.data), (self,), "exp")

        def backward():
            self._accumulate_grad(result._grad_data * result.data)

        result._grad_fn = backward
        return result

    def sqrt(self):
        return self**0.5

    def tanh(self):
        result = Tensor._from_op(np.tanh(self.data), (self,), "tanh")

        def backward():
            self._accumulate_grad(result._grad_data * (1 - result.data**2))

        result._grad_fn = backward
        return result

    def relu(self):
        result = Tensor._from_op(np.maximum(self.data, 0), (self,), "ReLU")

        def backward():
            self._accumulate_grad(result._grad_data * (self.data > 0))

        result._grad_fn = backward
        return result

    def masked_fill(self, mask: Tensor, value: builtins.float) -> Tensor:
        mask_data = np.broadcast_to(_ensure_tensor(mask).data, self.shape)
        result = Tensor._from_op(np.where(mask_data, value, self.data).astype(self.dtype), (self,), "MaskedFill")

        def backward():
            # filled positions are constants, so they get no gradient
            self._accumulate_grad(np.where(mask_data, 0, result._grad_data))

        result._grad_fn = backward
        return result

    def softmax(self, dim: int = -1) -> Tensor:
        # subtract the max for numerical stability
        e = np.exp(self.data - self.data.max(dim, keepdims=True))
        result = Tensor._from_op(e / e.sum(dim, keepdims=True), (self,), "Softmax")

        def backward():
            s, g = result.data, result._grad_data
            self._accumulate_grad(s * (g - (g * s).sum(dim, keepdims=True)))

        result._grad_fn = backward
        return result

    def log_softmax(self, dim: int = -1) -> Tensor:
        shifted = self.data - self.data.max(dim, keepdims=True)
        out = shifted - np.log(np.exp(shifted).sum(dim, keepdims=True))
        result = Tensor._from_op(out, (self,), "LogSoftmax")

        def backward():
            g = result._grad_data
            self._accumulate_grad(g - np.exp(result.data) * g.sum(dim, keepdims=True))

        result._grad_fn = backward
        return result

    # ---------------------------------------------------- in-place (no grad)

    def __iadd__(self, other):
        other = _ensure_tensor(other)
        self.data += other.data
        return self

    def __isub__(self, other):
        other = _ensure_tensor(other)
        self.data -= other.data
        return self

    def __imul__(self, other):
        other = _ensure_tensor(other)
        self.data *= other.data
        return self

    def __itruediv__(self, other):
        other = _ensure_tensor(other)
        self.data /= other.data
        return self


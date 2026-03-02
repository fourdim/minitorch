from typing import Any

import numpy as np


def compute_broadcast_dims(shape: tuple[int, ...], ndim: int):
    padded = (1,) * (ndim - len(shape)) + shape
    return tuple(i for i, s in enumerate(padded) if s == 1)


class Tensor:
    def __init__(self, data: Any):
        self.data: np.ndarray = data if isinstance(data, np.ndarray) else np.array(data)
        self.requires_grad = False
        self.grad: Tensor | None = None

        self._grad_fn = lambda: None
        self._children: tuple[Tensor, ...] = ()
        self._op = ""

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    def numpy(self):
        return self.data

    def backward(self, gradient=None):
        self.grad = gradient
        if self.grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "grad can be implicitly created only for scalar outputs"
                )
            else:
                self.grad = Tensor([1])
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
            t._grad_fn()

    def reshape(self, shape: tuple[int, ...]) -> Tensor:
        result = Tensor(self.data.reshape(shape))
        result.requires_grad = self.requires_grad
        if not result.requires_grad:
            return result
        result._children = (self,)
        result._op = "Reshape"

        def backward():
            if self.requires_grad:
                self.grad = self._zero_grad_if_none()
                self.grad += result.grad.reshape(self.shape)

        result._grad_fn = backward
        return result

    def transpose(self):
        result = Tensor(self.data.transpose())
        result.requires_grad = self.requires_grad
        if not result.requires_grad:
            return result
        result._children = (self,)
        result._op = "T"

        def backward():
            raise NotImplementedError

        result._grad_fn = backward
        return result

    def unsqueeze(self, dim: int):
        result = Tensor(np.expand_dims(self.data, dim))
        result.requires_grad = self.requires_grad
        if not result.requires_grad:
            return result
        result._children = (self,)
        result._op = "Unsqueeze"

        def backward():
            raise NotImplementedError

        result._grad_fn = backward
        return result

    def flatten(self, start_dim: int = 0, end_dim: int = -1):
        if end_dim < 0:
            end_dim += len(self.shape)
        new_shape = self.shape[:start_dim] + (-1,) + self.shape[end_dim + 1 :]
        return self.reshape(new_shape)

    def sum(self, dim: int | tuple[int, ...], keepdim: bool = False):
        result = Tensor(self.data.sum(dim, keepdims=keepdim))
        result.requires_grad = self.requires_grad
        if not result.requires_grad:
            return result
        result._children = (self,)
        result._op = "Sum"

        def backward():
            if result.grad is None:
                raise RuntimeError("result grad must be calculated before its children")
            rgrad = result.grad
            if not keepdim:
                if isinstance(dim, int):
                    rgrad = rgrad.unsqueeze(dim)
                else:
                    for d in sorted(dim):
                        rgrad = rgrad.unsqueeze(d)
            if self.requires_grad:
                self.grad = self._zero_grad_if_none()
                self.grad += rgrad.broadcast_to(self.shape)

        result._grad_fn = backward
        return result

    def broadcast_to(self, shape: tuple[int, ...]) -> Tensor:
        if self.shape == shape:
            return self
        result = Tensor(np.broadcast_to(self.data, shape))
        result.requires_grad = self.requires_grad
        if not result.requires_grad:
            return result
        result._children = (self,)
        result._op = "Broadcast"
        broadcast_dims = compute_broadcast_dims(self.shape, len(result.shape))

        def backward():
            if result.grad is None:
                raise RuntimeError("result grad must be calculated before its children")
            if self.requires_grad:
                self.grad = self._zero_grad_if_none()
                self.grad += result.grad.sum(broadcast_dims, keepdim=True)

        result._grad_fn = backward
        return result

    def __add__(self, other: Tensor) -> Tensor:
        result = Tensor(self.data + other.data)
        # If any of children requires grad, then its result will require grad.
        # This is required for loss = a + b + c case
        # where a + b creates a intermediate tensor with `require_grad` default to false
        # To tackle this, we need to have the following line.
        result.requires_grad = self.requires_grad | other.requires_grad
        if not result.requires_grad:
            # No need for computation graph if grad is not required.
            # Reference to self and other can be immediately dropped.
            return result
        # If self, other are not in the same shape, apply broadcast rules
        self = self.broadcast_to(result.shape)
        other = other.broadcast_to(result.shape)
        result._children = (self, other)
        result._op = "+"

        def backward():
            if result.grad is None:
                raise RuntimeError("result grad must be calculated before its children")
            if self.requires_grad:
                self.grad = self._zero_grad_if_none()
                self.grad += result.grad
            if other.requires_grad:
                other.grad = other._zero_grad_if_none()
                other.grad += result.grad

        result._grad_fn = backward
        return result

    def __matmul__(self, other: Tensor) -> Tensor:
        result = Tensor(self.data @ other.data)
        result.requires_grad = self.requires_grad | other.requires_grad
        if not result.requires_grad:
            return result
        result._children = (self, other)
        result._op = "@"

        def backward():
            if result.grad is None:
                raise RuntimeError(
                    "result gradient must be calculated before its children"
                )
            if self.requires_grad:
                self.grad = self._zero_grad_if_none()
                self.grad += result.grad @ other.transpose()
            if other.requires_grad:
                other.grad = other._zero_grad_if_none()
                other.grad += self.transpose() @ result.grad

        result._grad_fn = backward
        return result

    def __mul__(self, other: Tensor) -> Tensor:
        result = Tensor(self.data * other.data)
        result.requires_grad = self.requires_grad | other.requires_grad
        if not result.requires_grad:
            return result
        self = self.broadcast_to(result.shape)
        other = other.broadcast_to(result.shape)
        result._children = (self, other)
        result._op = "*"

        def backward():
            if result.grad is None:
                raise RuntimeError(
                    "result gradient must be calculated before its children"
                )
            if self.requires_grad:
                self.grad = self._zero_grad_if_none()
                self.grad.data += other.data * result.grad.data
            if other.requires_grad:
                other.grad = other._zero_grad_if_none()
                other.grad.data += self.data * result.grad.data

        result._grad_fn = backward
        return result

    def __getitem__(self, key):
        result = Tensor(self.data[key])
        result.requires_grad = self.requires_grad
        if not result.requires_grad:
            return result
        result._children = (self,)
        result._op = "GetItem"

        def backward():
            if result.grad is None:
                raise RuntimeError("result grad must be calculated before its children")
            if self.requires_grad:
                self.grad = self._zero_grad_if_none()
                grad_data = np.zeros_like(self.data)
                np.add.at(grad_data, key, result.grad.data)
                self.grad += Tensor(grad_data)

        result._grad_fn = backward
        return result

    def _zero_grad_if_none(self) -> Tensor:
        if self.grad is not None:
            return self.grad
        return Tensor(np.zeros_like(self.data))

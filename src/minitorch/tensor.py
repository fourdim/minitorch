from typing import Any

import numpy as np


class Tensor:
    def __init__(self, data: Any):
        self.data: np.ndarray = data if isinstance(data, np.ndarray) else np.array(data)
        self.requires_grad = False
        self.grad: Tensor | None = None

        self._backward = lambda: None
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
            t._backward()

    def transpose(self):
        result = Tensor(self.data.transpose())
        result.requires_grad = self.requires_grad
        if not result.requires_grad:
            return result
        result._children = (self,)
        result._op = "T"

        def backward():
            raise NotImplementedError

        result._backward = backward
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

        result._backward = backward
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

        result._backward = backward
        return result

    def _zero_grad_if_none(self) -> Tensor:
        if self.grad is not None:
            return self.grad
        return Tensor(np.zeros(self.shape, dtype=np.double))

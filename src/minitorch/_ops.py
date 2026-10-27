from collections.abc import Iterable

import numpy as np

from minitorch._tensor import Tensor


def cat(tensors: Iterable[Tensor], dim: int = 0) -> Tensor:
    tensors: tuple[Tensor, ...] = tuple(tensors)
    result = Tensor._from_op(np.concatenate([t.data for t in tensors], axis=dim), tensors, "Cat")

    def backward():
        bounds = np.cumsum([t.shape[dim] for t in tensors])[:-1]
        for t, g in zip(tensors, np.split(result._grad_data, bounds, axis=dim), strict=True):
            t._accumulate_grad(g)

    result._grad_fn = backward
    return result


def stack(tensors: Iterable[Tensor], dim: int = 0) -> Tensor:
    return cat([t.unsqueeze(dim) for t in tensors], dim)


def tril(input: Tensor, diagonal: int = 0) -> Tensor:
    result = Tensor._from_op(np.tril(input.data, diagonal), (input,), "Tril")

    def backward():
        input._accumulate_grad(np.tril(result._grad_data, diagonal))

    result._grad_fn = backward
    return result


def triu(input: Tensor, diagonal: int = 0) -> Tensor:
    result = Tensor._from_op(np.triu(input.data, diagonal), (input,), "Triu")

    def backward():
        input._accumulate_grad(np.triu(result._grad_data, diagonal))

    result._grad_fn = backward
    return result

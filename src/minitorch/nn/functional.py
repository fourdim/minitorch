import numpy as np

from minitorch._tensor import Tensor


def relu(input: Tensor) -> Tensor:
    return input.relu()


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    return input.softmax(dim)


def log_softmax(input: Tensor, dim: int = -1) -> Tensor:
    return input.log_softmax(dim)


def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training or p == 0:
        return input
    # inverted dropout: scale at train time so eval needs no change
    mask = (np.random.rand(*input.shape) >= p).astype(input.dtype) / input.dtype.type(1 - p)
    return input * Tensor(mask)


def cross_entropy(input: Tensor, target: Tensor, ignore_index: int = -100) -> Tensor:
    """Mean cross entropy over (N, C) logits and (N,) class indices."""
    logp = input.log_softmax(-1)
    keep = target.data != ignore_index
    rows = np.nonzero(keep)[0]
    picked = logp[rows, target.data[rows]]
    return -picked.sum() / max(len(rows), 1)

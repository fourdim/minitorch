import numpy as np

from minitorch.tensor import Tensor


def zeros(shape: tuple[int, ...], dtype=None) -> Tensor:
    return Tensor(np.zeros(shape, dtype))


def zeros_like(input, dtype=None) -> Tensor:
    return Tensor(np.zeros_like(input, dtype))


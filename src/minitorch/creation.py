import numpy as np

from minitorch._tensor import Tensor

long = np.int64
int64 = np.int64
float32 = np.float32
float64 = np.float64
bool = np.bool_


def _shape(size) -> tuple[int, ...]:
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        return tuple(size[0])
    return size


def tensor(data, dtype=None) -> Tensor:
    if isinstance(data, Tensor):
        data = data.data
    return Tensor(np.array(data, dtype=dtype) if dtype is not None else data)


def manual_seed(seed: int):
    np.random.seed(seed)


def zeros(*size, dtype=None) -> Tensor:
    return Tensor(np.zeros(_shape(size), dtype or np.float32))


def zeros_like(input, dtype=None) -> Tensor:
    return Tensor(np.zeros_like(input.data if isinstance(input, Tensor) else input, dtype))


def ones(*size, dtype=None) -> Tensor:
    return Tensor(np.ones(_shape(size), dtype or np.float32))


def ones_like(input, dtype=None) -> Tensor:
    return Tensor(np.ones_like(input.data if isinstance(input, Tensor) else input, dtype))


def randn(*size, dtype=None) -> Tensor:
    return Tensor(np.random.randn(*_shape(size)).astype(dtype or np.float32))


def rand(*size, dtype=None) -> Tensor:
    return Tensor(np.random.rand(*_shape(size)).astype(dtype or np.float32))


def arange(start, end=None, step=1, dtype=None) -> Tensor:
    if end is None:
        start, end = 0, start
    return Tensor(np.arange(start, end, step, dtype=dtype))


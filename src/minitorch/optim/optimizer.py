from collections.abc import Iterable

from minitorch.nn import Parameter


class Optimizer:
    def __init__(self, params: Iterable[Parameter], lr: float):
        self.params: list[Parameter] = list(params)
        self.lr: float = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            p.grad = None

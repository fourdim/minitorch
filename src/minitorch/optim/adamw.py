from collections.abc import Iterable

import numpy as np

from minitorch.nn import Parameter
from minitorch.optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params: Iterable[Parameter], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps: float = eps
        self.weight_decay: float = weight_decay
        self.t = 0
        # optimizer state is never differentiated, so plain arrays are enough
        self.m: list[np.ndarray] = [np.zeros_like(p.data) for p in self.params]
        self.v: list[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def _decay(self, p: np.ndarray, grad: np.ndarray) -> np.ndarray:
        # classic L2 penalty: folded into the gradient
        return grad + self.weight_decay * p if self.weight_decay else grad

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = self._decay(p.data, p.grad.data)

            # update biased moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2

            # bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # parameter update
            p.data -= (self.lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(p.dtype)


class AdamW(Adam):
    def __init__(self, params: Iterable[Parameter], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr, betas, eps, weight_decay)

    def _decay(self, p: np.ndarray, grad: np.ndarray) -> np.ndarray:
        # decoupled weight decay: shrink the weight directly
        p -= (self.lr * self.weight_decay * p).astype(p.dtype)
        return grad

from collections.abc import Iterable

from minitorch.nn import Parameter
from minitorch.tensor import Tensor
from minitorch.utils import zeros_like


class AdamW:
    def __init__(self, params: Iterable[Parameter], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params: list[Parameter] = list(params)
        self.lr: float = lr
        self.beta1, self.beta2 = betas
        self.eps: float = eps
        self.weight_decay: float = weight_decay
        self.t = 0
        self.m: list[Tensor] = [zeros_like(p) for p in self.params]
        self.v: list[Tensor] = [zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad

            # decoupled weight decay
            p -= self.lr * self.weight_decay * p

            # update biased moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** Tensor(2))

            # bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # parameter update
            p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

import numpy as np

from minitorch.nn.module import Module
from minitorch.nn.parameter import Parameter
from minitorch.tensor import Tensor


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        # note: kaiming init
        self.weight = Parameter(np.random.randn(in_features, out_features) / in_features**0.5)
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x: Tensor):
        self.out = x @ self.weight
        if self.bias is not None:
          self.out += self.bias
        return self.out

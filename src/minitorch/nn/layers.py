import numpy as np

from minitorch._tensor import Tensor
from minitorch.nn import functional as F
from minitorch.nn.module import Module
from minitorch.nn.parameter import Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias=True, dtype=None):
        super().__init__()
        dtype = dtype or np.float32
        # note: kaiming init
        self.weight = Parameter(np.random.randn(in_features, out_features).astype(dtype) / dtype(in_features**0.5))
        self.bias = Parameter(np.zeros(out_features, dtype)) if bias else None

    def forward(self, x: Tensor):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype=None):
        super().__init__()
        self.weight = Parameter(np.random.randn(num_embeddings, embedding_dim).astype(dtype or np.float32))

    def forward(self, x: Tensor):
        # a lookup table: each index picks a row of the weight
        return self.weight[x]


class LayerNorm(Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(np.ones(normalized_shape, dtype or np.float32))
        self.bias = Parameter(np.zeros(normalized_shape, dtype or np.float32))

    def forward(self, x: Tensor):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, correction=0)
        x_hat = (x - mean) / (var + self.eps).sqrt()
        return x_hat * self.weight + self.bias


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor):
        return F.dropout(x, self.p, self.training)


class ReLU(Module):
    def forward(self, x: Tensor):
        return x.relu()


class Tanh(Module):
    def forward(self, x: Tensor):
        return x.tanh()

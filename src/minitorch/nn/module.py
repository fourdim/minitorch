from abc import ABC, abstractmethod
from typing import Any

from .parameter import Parameter


class Module(ABC):
    def __init__(self):
        self._parameters: dict[Any, Parameter] = {}
        self._modules: dict[Any, Module] = {}

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    def parameters(self):
        yield from self._parameters.values()
        for module in self._modules.values():
            yield from module.parameters()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value

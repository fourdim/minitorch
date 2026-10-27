from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

from minitorch._tensor import Tensor
from minitorch.nn.parameter import Parameter


class Module(ABC):
    _parameters: dict[str, Parameter]
    _modules: dict[str, Module]
    _buffers: dict[str, Tensor]
    training: bool

    def __init__(self):
        # bypass our own __setattr__, it needs these dicts to exist
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_buffers", {})
        self.training = True

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    def parameters(self) -> Iterator[Parameter]:
        seen = set()
        for p in self._all_parameters():
            if p not in seen:
                seen.add(p)
                yield p

    def _all_parameters(self) -> Iterator[Parameter]:
        yield from self._parameters.values()
        for module in self._modules.values():
            yield from module._all_parameters()

    def children(self) -> Iterator[Module]:
        yield from self._modules.values()

    def register_buffer(self, name: str, tensor: Tensor):
        """Store a tensor that is part of the module state but is not trained."""
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

    def train(self, mode: bool = True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)


class ModuleList(Module):
    def __init__(self, modules: Iterable[Module] = ()):
        super().__init__()
        for module in modules:
            self.append(module)

    def append(self, module: Module):
        setattr(self, str(len(self._modules)), module)

    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx]

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleList has no forward, iterate over it instead")


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            setattr(self, str(i), module)

    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx]

    def __len__(self):
        return len(self._modules)

    def forward(self, x: Any):
        for module in self._modules.values():
            x = module(x)
        return x

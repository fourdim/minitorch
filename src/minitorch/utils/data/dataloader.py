import random
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from minitorch._ops import stack
from minitorch._tensor import Tensor
from minitorch.creation import tensor


@runtime_checkable
class Dataset[T](Protocol):
    def __getitem__(self, index: int, /) -> T: ...

    def __len__(self) -> int: ...


def default_collate(batch: list[Any]) -> Any:
    """Stack a list of samples into a batch, keeping tuples as tuples."""
    elem = batch[0]
    if isinstance(elem, Tensor):
        return stack(batch)
    if isinstance(elem, (tuple, list)):
        return type(elem)(default_collate(list(field)) for field in zip(*batch, strict=True))
    return tensor(batch)


class DataLoader:
    def __init__(
        self,
        dataset: Dataset[Any],
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn: Callable[[list[Any]], Any] = default_collate,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                break
            yield self.collate_fn([self.dataset[i] for i in batch])

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return -(-len(self.dataset) // self.batch_size)

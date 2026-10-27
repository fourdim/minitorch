from minitorch.optim.optimizer import Optimizer


class StepLR:
    """Multiply the learning rate by `gamma` every `step_size` calls to step()."""

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = optimizer.lr
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.base_lr * self.gamma ** (self.last_epoch // self.step_size)

    def get_last_lr(self) -> list[float]:
        return [self.optimizer.lr]

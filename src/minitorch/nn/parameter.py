from minitorch.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data)
        self.requires_grad = True

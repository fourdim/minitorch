import unittest

import numpy as np
import torch

import minitorch


class TestTensor(unittest.TestCase):
    def test_add(self):
        t1 = minitorch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        t2 = minitorch.Tensor([[7, 8, 9], [1, 2, 3], [4, 5, 6]])
        t1.requires_grad = True
        t2.requires_grad = True
        t3 = t1 + t2
        self.assertTrue((t3.data == t1.data + t2.data).all())
        self.assertTupleEqual(t3._children, (t1, t2))

    def test_backward(self):
        bases = (
            np.random.rand(1, 3),
            np.random.rand(3, 1),
            np.random.rand(1, 1),
        )
        ts: list[torch.Tensor] = [torch.Tensor(b) for b in bases]
        for t in ts:
            t.requires_grad = True
        mts: list[minitorch.Tensor] = [minitorch.Tensor(b) for b in bases]
        for mt in mts:
            mt.requires_grad = True

        t3 = ts[0] @ ts[1] + ts[2]
        for t in ts + [t3]:
            t.grad = None
            t.retain_grad()
        t3.backward()

        mt3 = mts[0] @ mts[1] + mts[2]
        for mt in mts + [mt3]:
            mt.grad = None
        mt3.backward()
        for t, mt in zip(ts, mts, strict=True):
            self.assertTrue(np.allclose(t.grad.numpy(), mt.grad.numpy()))


if __name__ == "__main__":
    unittest.main()

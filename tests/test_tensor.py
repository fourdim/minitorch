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

    def test_mul(self):
        t1 = minitorch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        t2 = minitorch.Tensor([[7, 8, 9], [1, 2, 3], [4, 5, 6]])
        t1.requires_grad = True
        t2.requires_grad = True
        t3 = t1 * t2
        self.assertTrue((t3.data == t1.data * t2.data).all())
        self.assertTupleEqual(t3._children, (t1, t2))

    def test_matmul_backward(self):
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

    def test_mul_backward(self):
        bases = (
            np.random.rand(2, 3),
            np.random.rand(2, 3),
        )
        ts: list[torch.Tensor] = [torch.Tensor(b) for b in bases]
        for t in ts:
            t.requires_grad = True
        mts: list[minitorch.Tensor] = [minitorch.Tensor(b) for b in bases]
        for mt in mts:
            mt.requires_grad = True

        t3 = ts[0] * ts[1]
        for t in ts + [t3]:
            t.grad = None
            t.retain_grad()
        grad = torch.ones(2, 3)
        t3.backward(grad)

        mt3 = mts[0] * mts[1]
        for mt in mts + [mt3]:
            mt.grad = None
        mt3.backward(minitorch.Tensor(np.ones((2, 3))))
        for t, mt in zip(ts, mts, strict=True):
            self.assertTrue(np.allclose(t.grad.numpy(), mt.grad.numpy()))

    def test_mul_broadcast_backward(self):
        # Test broadcasting: (2, 3) * (3,) -> (2, 3)
        base1 = np.random.rand(2, 3)
        base2 = np.random.rand(3)

        t1 = torch.Tensor(base1)
        t2 = torch.Tensor(base2)
        t1.requires_grad = True
        t2.requires_grad = True

        mt1 = minitorch.Tensor(base1)
        mt2 = minitorch.Tensor(base2)
        mt1.requires_grad = True
        mt2.requires_grad = True

        # Forward
        t3 = t1 * t2
        mt3 = mt1 * mt2

        # Backward with gradient
        grad = torch.ones(2, 3)
        t3.backward(grad)
        mt3.backward(minitorch.Tensor(np.ones((2, 3))))

        # Compare gradients
        self.assertTrue(np.allclose(t1.grad.numpy(), mt1.grad.numpy()))
        self.assertTrue(np.allclose(t2.grad.numpy(), mt2.grad.numpy()))


if __name__ == "__main__":
    unittest.main()

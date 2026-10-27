import unittest

import numpy as np
import torch
import torch.nn.functional as TF

import minitorch
import minitorch.nn.functional as F


def check_grads(test: unittest.TestCase, fn_torch, fn_mini, *arrays: np.ndarray, atol=1e-5):
    """Run the same function in torch and minitorch and compare output and input grads."""
    ts = [torch.tensor(a, requires_grad=True) for a in arrays]
    mts = [minitorch.Tensor(a) for a in arrays]
    for mt in mts:
        mt.requires_grad = True

    out = fn_torch(*ts)
    mout = fn_mini(*mts)
    test.assertTrue(np.allclose(out.detach().numpy(), mout.numpy(), atol=atol))

    grad = np.asarray(np.random.rand(*out.shape), dtype=arrays[0].dtype)
    out.backward(torch.tensor(grad))
    mout.backward(minitorch.Tensor(grad))
    for t, mt in zip(ts, mts, strict=True):
        assert t.grad is not None and mt.grad is not None
        test.assertTrue(np.allclose(t.grad.numpy(), mt.grad.numpy(), atol=atol))


def rand(*shape):
    return np.random.randn(*shape).astype(np.float32)


class TestOps(unittest.TestCase):
    def test_batched_matmul_broadcast(self):
        check_grads(self, lambda a, b: a @ b, lambda a, b: a @ b, rand(2, 3, 4), rand(4, 5))

    def test_transpose(self):
        check_grads(self, lambda a: a.transpose(-2, -1) * 2, lambda a: a.transpose(-2, -1) * 2, rand(2, 3, 4))

    def test_view(self):
        check_grads(self, lambda a: a.view(6, -1) * a.view(6, -1), lambda a: a.view(6, -1) * a.view(6, -1), rand(2, 3, 4))

    def test_softmax(self):
        check_grads(self, lambda a: a.softmax(-1), lambda a: a.softmax(-1), rand(3, 5))

    def test_masked_fill_softmax(self):
        mask = np.tril(np.ones((4, 4))) == 0
        check_grads(
            self,
            lambda a: a.masked_fill(torch.tensor(mask), float("-inf")).softmax(-1),
            lambda a: a.masked_fill(minitorch.Tensor(mask), float("-inf")).softmax(-1),
            rand(2, 4, 4),
        )

    def test_cat(self):
        check_grads(self, lambda a, b: torch.cat([a, b], dim=-1), lambda a, b: minitorch.cat([a, b], dim=-1), rand(2, 3), rand(2, 4))

    def test_tril_triu(self):
        check_grads(self, lambda a: torch.tril(a, 1), lambda a: minitorch.tril(a, 1), rand(3, 4))
        check_grads(self, lambda a: torch.triu(a, -1), lambda a: minitorch.triu(a, -1), rand(3, 4))

    def test_mean_var_sqrt(self):
        check_grads(
            self,
            lambda a: (a - a.mean(-1, keepdim=True)) / (a.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt(),
            lambda a: (a - a.mean(-1, keepdim=True)) / (a.var(-1, keepdim=True, correction=0) + 1e-5).sqrt(),
            rand(3, 6),
        )

    def test_relu_exp_log_tanh(self):
        check_grads(self, lambda a: a.relu() + a.exp() + a.tanh(), lambda a: a.relu() + a.exp() + a.tanh(), rand(3, 4))
        check_grads(self, lambda a: a.log(), lambda a: a.log(), np.random.rand(3, 4).astype(np.float32) + 0.1)

    def test_max(self):
        check_grads(self, lambda a: a.max(-1).values, lambda a: a.max(-1), rand(3, 4))

    def test_getitem(self):
        idx = np.array([[0, 2], [1, 1]])
        check_grads(self, lambda a: a[torch.tensor(idx)], lambda a: a[minitorch.Tensor(idx)], rand(3, 4))
        check_grads(self, lambda a: a[0, -1], lambda a: a[0, -1], rand(3, 4))

    def test_cross_entropy_ignore_index(self):
        target = np.array([1, -100, 0, 3, -100])
        check_grads(
            self,
            lambda a: TF.cross_entropy(a, torch.tensor(target)),
            lambda a: F.cross_entropy(a, minitorch.Tensor(target)),
            rand(5, 4),
        )

    def test_no_grad(self):
        a = minitorch.Tensor(rand(2, 2))
        a.requires_grad = True
        with minitorch.no_grad():
            self.assertFalse((a * 2).requires_grad)
        self.assertTrue((a * 2).requires_grad)

    def test_float32_preserved(self):
        a = minitorch.randn(2, 2)
        self.assertEqual((a * 0.5 + 1).sqrt().dtype, np.float32)
        self.assertEqual(minitorch.tensor([1, 2], dtype=minitorch.long).dtype, np.int64)


class TestModules(unittest.TestCase):
    def test_layernorm(self):
        x = rand(2, 3, 8)
        ln = torch.nn.LayerNorm(8)
        mln = minitorch.nn.LayerNorm(8)
        check_grads(self, ln, mln, x)

    def test_layer_dtype(self):
        x = minitorch.randn(2, 4, dtype=minitorch.float64)
        self.assertEqual(minitorch.nn.Linear(4, 3, dtype=minitorch.float64)(x).dtype, np.float64)
        self.assertEqual(minitorch.nn.LayerNorm(4, dtype=minitorch.float64)(x).dtype, np.float64)
        emb = minitorch.nn.Embedding(5, 4, dtype=minitorch.float64)
        self.assertEqual(emb(minitorch.tensor([0, 3])).dtype, np.float64)
        self.assertEqual(minitorch.nn.Linear(4, 3).weight.dtype, np.float32)

    def test_parameters_and_modes(self):
        model = minitorch.nn.Sequential(
            minitorch.nn.Linear(4, 8),
            minitorch.nn.ReLU(),
            minitorch.nn.Dropout(0.5),
            minitorch.nn.Linear(8, 2, bias=False),
        )
        self.assertEqual(len(list(model.parameters())), 3)
        model.eval()
        self.assertFalse(model[2].training)
        x = minitorch.randn(3, 4)
        self.assertTrue(np.allclose(model(x).numpy(), model(x).numpy()))

    def test_module_list_and_buffer(self):
        class M(minitorch.nn.Module):
            tril: minitorch.Tensor

            def __init__(self):
                super().__init__()
                self.layers = minitorch.nn.ModuleList([minitorch.nn.Linear(2, 2) for _ in range(3)])
                self.register_buffer("tril", minitorch.tril(minitorch.ones(2, 2)))

            def forward(self, x):
                return x

        m = M()
        self.assertEqual(len(list(m.parameters())), 6)
        self.assertEqual(m.tril.shape, (2, 2))

    def test_adam_fits_linear(self):
        minitorch.manual_seed(0)
        x = minitorch.randn(64, 3)
        y = x @ minitorch.Tensor(np.array([[1.0], [-2.0], [0.5]], np.float32))
        model = minitorch.nn.Linear(3, 1)
        opt = minitorch.optim.Adam(model.parameters(), lr=0.05)
        sched = minitorch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
        for _ in range(300):
            loss = ((model(x) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        self.assertLess(loss.item(), 1e-3)
        self.assertAlmostEqual(opt.lr, 0.05 * 0.125)


class TestData(unittest.TestCase):
    def test_dataloader(self):
        class DS(minitorch.utils.data.Dataset):
            def __getitem__(self, index):
                return minitorch.tensor([index, index]), minitorch.tensor(index)

            def __len__(self):
                return 5

        loader = minitorch.utils.data.DataLoader(DS(), batch_size=2, shuffle=True)
        batches = list(loader)
        self.assertEqual(len(loader), 3)
        self.assertEqual(batches[0][0].shape, (2, 2))
        self.assertEqual(sorted(sum((b[1].tolist() for b in batches), [])), list(range(5)))


if __name__ == "__main__":
    unittest.main()

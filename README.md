# minitorch

minitorch is a minimal reimplementation of PyTorch for educational purpose.

It follows Karpathy's [NN Zero To Hero](https://github.com/karpathy/nn-zero-to-hero) tutorial, but is implemented in Python and uses NumPy for tensor operations. Even that, the use of NumPy will be kept to a minimum, and the core of the library will be implemented from scratch.

The API is designed to be as close as possible to PyTorch, so that users can easily understand the underlying concepts.

## Installation

To install minitorch, simply clone the repository and install the required dependencies:

```bash
git clone https://github.com/fourdim/minitorch
uv sync
```

## Testing

To run the tests, we need to install the PyTorch as the ground truth:

```bash
uv sync --extra torch
```

Then, we can run the tests by specifying the test file:

```bash
python tests/test_tensor.py
```

## Acknowledgements

- [Karpathy's NN Zero To Hero Youtube](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Karpathy's NN Zero To Hero GitHub](https://github.com/karpathy/nn-zero-to-hero)
- [PyTorch](https://pytorch.org/)

## License

[MIT](./LICENSE)

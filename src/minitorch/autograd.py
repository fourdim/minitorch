import functools

_grad_enabled = True


def is_grad_enabled() -> bool:
    return _grad_enabled


def set_grad_enabled(mode: bool):
    global _grad_enabled
    _grad_enabled = mode


class no_grad:
    """Disable graph recording, usable as a context manager or a decorator."""

    def __enter__(self):
        self.prev = is_grad_enabled()
        set_grad_enabled(False)

    def __exit__(self, *exc):
        set_grad_enabled(self.prev)

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with no_grad():
                return fn(*args, **kwargs)

        return wrapper

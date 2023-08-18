import functools
import torch


def cache_tensor(func):
    cache = {}
    sentinel = object()

    @functools.wraps(func)
    def wrapper(tensorlike_arg):
        key = (*tensorlike_arg.shape, torch.sin(tensorlike_arg).sum().item(), torch.tanh(tensorlike_arg).sum().item())
        result = cache.get(key, sentinel)
        if result is sentinel:
            result = func(tensorlike_arg)
            cache[key] = result
        return result

    return wrapper
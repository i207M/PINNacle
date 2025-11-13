import torch


@torch.no_grad()
def make_lln(*components, eps=1):
    denom = eps + sum([torch.abs(c) for c in components])
    return denom

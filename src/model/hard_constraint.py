import torch


def hard_constraint_wrapper(net, data, alpha=5):
    """
    Wrapper for hard constrain.
    output = ic + t * NN
    """

    def output_transform(inputs, outputs):
        t = inputs[..., -1:]
        w = torch.exp(-t * alpha)
        outputs = outputs * (1 - w) + data.ic_func(inputs[..., :-1]) * w
        return outputs

    net._output_transform = output_transform
    return net

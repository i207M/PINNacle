import re


def parse_hidden_layers(command_args):
    layers = []
    for s in re.split(r"[,_-]", command_args.hidden_layers):
        if '*' in s:
            siz, num = s.split('*')
            layers += [int(siz)] * int(num)
        else:
            layers += [int(s)]
    return layers


def parse_loss_weight(command_args):
    if command_args.loss_weight == '': return None
    weights = []
    for s in re.split(r"[,_-]", command_args.loss_weight):
        weights.append(float(s))
    return weights

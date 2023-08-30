import torch
import torch.nn as nn
import torch.nn.init as init

'''
class MyLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        w_times_x= torch.mm(x, self.weights.t())
        return torch.add(w_times_x, self.bias)  # w times x + b
    
# Neural Network
'''

class MLP(torch.nn.Module):
    def __init__(self, layer_sizes, type='tanh'):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            # layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # add an activation function when not in the last layer
                if type=='tanh':
                    layers.append(nn.Tanh())
                if type=='relu':
                    layers.append(nn.ReLU())
                if type=='gelu':
                    layers.append(nn.GELU())
                # layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # use xavier_uniform_ initialization
                # init.xavier_uniform_(module.weight)
                # or，use xavier_normal_ initialization
                init.xavier_normal_(module.weight)
                # set all biases to 0
                init.zeros_(module.bias)
                
    def forward(self, x):
        return self.net(x)
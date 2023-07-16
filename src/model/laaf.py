from collections import OrderedDict
import torch


class LAAFlayer(torch.nn.Module):

    def __init__(self, n, a, dim_in, dim_out, activation):
        super(LAAFlayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n = n
        self.a = a
        self.activation = activation

        self.fc = torch.nn.Linear(self.dim_in, self.dim_out)

    def forward(self, x):
        x1 = self.fc(x)
        x2 = self.n * torch.mul(self.a, x1)
        out = self.activation(x2)
        return out


# n layers deep neural network with LAAF
class DNN_LAAF(torch.nn.Module):

    def __init__(self, n_layers, n_hidden, x_dim=1, u_dim=1):
        super(DNN_LAAF, self).__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.activation = torch.nn.Tanh
        self.regularizer = None

        self.a = torch.nn.Parameter(torch.empty(size=(self.n_layers, self.n_hidden)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.4)

        layer_list = list()
        layer_list.append(('layer0', LAAFlayer(10, self.a[0, :], x_dim, n_hidden, self.activation())))
        for i in range(self.n_layers - 1):
            layer_list.append(('layer%d' % (i + 1), LAAFlayer(10, self.a[i + 1, :], n_hidden, n_hidden, self.activation())))
        layer_list.append(('layer%d' % n_layers, torch.nn.Linear(self.n_hidden, self.u_dim)))
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


class DNN_GAAF(torch.nn.Module):

    def __init__(self, n_layers, n_hidden, x_dim=1, u_dim=1):
        super(DNN_GAAF, self).__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.activation = torch.nn.Tanh
        self.regularizer = None

        self.a = torch.nn.Parameter(torch.empty(size=(self.n_layers, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.4)

        layer_list = list()
        layer_list.append(('layer0', LAAFlayer(10, self.a[0], x_dim, n_hidden, self.activation())))
        for i in range(self.n_layers - 1):
            layer_list.append(('layer%d' % (i + 1), LAAFlayer(10, self.a[i + 1], n_hidden, n_hidden, self.activation())))
        layer_list.append(('layer%d' % n_layers, torch.nn.Linear(self.n_hidden, self.u_dim)))
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

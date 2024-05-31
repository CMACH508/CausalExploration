from abc import *

import torch
import torch.nn as nn


class NetworkBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        super(NetworkBase, self).__init__()
    @abstractmethod
    def forward(self, x):
        return x
    
class Network(NetworkBase):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.relu,last_activation = None):
        super(Network, self).__init__()
        self.activation = activation_function
        self.last_activation = last_activation
        layers_unit = [input_dim]+ [hidden_dim]*(layer_num-1) 
        layers = ([nn.Linear(layers_unit[idx],layers_unit[idx+1]) for idx in range(len(layers_unit)-1)])
        self.layers = nn.ModuleList(layers)
        self.last_layer = nn.Linear(layers_unit[-1],output_dim)
        self.network_init()
    def forward(self, x):
        return self._forward(x)
    def _forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.last_layer(x)
        if self.last_activation != None:
            x = self.last_activation(x)
        return x
    def network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()


class Causal_Network(NetworkBase):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.relu, last_activation = None):
        super(Causal_Network, self).__init__()
        self.activation = activation_function
        self.last_activation = last_activation
        layers_unit = [input_dim]+ [hidden_dim]*(layer_num-1) 
        layers = ([nn.Linear(layers_unit[idx],layers_unit[idx+1]) for idx in range(len(layers_unit)-1)])
        self.layers = nn.ModuleList(layers)
        self.last_layer_lst=[]
        for _ in range(output_dim):
            temp_layer = nn.Linear(layers_unit[-1], 1).cuda()
            self.last_layer_lst.append(temp_layer)
        self.network_init()

    def forward(self, x, index):
        return self._forward(x, index)

    def _forward(self, x, index):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.last_layer_lst[index](x)
        if self.last_activation != None:
            x = self.last_activation(x)
        return x

    def network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

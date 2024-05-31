import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base import Causal_Network


def init_module(m, w_init, b_init):
    if hasattr(m, 'initialized'):
        return
    if (hasattr(m, 'weight') and not hasattr(m, 'weight_initialized')
            and m.weight is not None and w_init is not None):
        w_init(m.weight)
    if (hasattr(m, 'bias') and not hasattr(m, 'bias_initialized')
            and m.bias is not None and b_init is not None):
        b_init(m.bias)


class Causal_world_net(Causal_Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation = None):
        super(Causal_world_net, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        
    def forward(self, *x, iindex):
        x = torch.cat(x,-1)
        return self._forward(x, iindex)

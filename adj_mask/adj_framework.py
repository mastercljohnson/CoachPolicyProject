import torch
from torch import nn
from adj_mask import AdjMask
from qmix import QMix

class AdjFrame(nn.Module):
    def __init__(self, agents, hidden_dim, q_hidden_dim, state_space, action_space, adj_mask=None, **kwargs):
        super().__init__()
        self.adj_mask_layer = AdjMask(hidden_dim, adj_mask=adj_mask, **kwargs)
        self.qmix = QMix(agents, state_space, action_space, q_hidden_dim, **kwargs)
    
    def forward(self, x):
        x  = self.adj_mask_layer(x)
        x = self.qmix(x)
        return self.adj_mask_layer(x)
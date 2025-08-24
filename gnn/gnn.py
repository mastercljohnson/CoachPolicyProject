from torch import nn

class GNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(self, x, edge_index):
        raise NotImplementedError
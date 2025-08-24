from torch import nn

class GNN(nn.Module):
    def __init__(self, num_vertices, hidden_dim, **kwargs):
        super().__init__()

        self.adj_mat_ff = nn.Linear(num_vertices, hidden_dim)
    
    def forward(self, x, edge_index):
        adj_enc = self.adj_mat_ff(x)
        raise NotImplementedError
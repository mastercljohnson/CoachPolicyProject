from torch import nn

#  1. Use GNN for task allocation
#  2. Create Q network for graph result 
#  Backprop q loss with actual reward from env.

#  3. Create policies for subtasks that take partial observ and zero pad full
#  4. Create Q networks for each agent, with hypernetworks to generate weights >=0 weights

class GNN(nn.Module):
    def __init__(self, num_vertices, hidden_dim, **kwargs):
        super().__init__()

        self.adj_mat_ff = nn.Linear(num_vertices, hidden_dim)
    
    def forward(self, x, edge_index):
        adj_enc = self.adj_mat_ff(x)
        raise NotImplementedError
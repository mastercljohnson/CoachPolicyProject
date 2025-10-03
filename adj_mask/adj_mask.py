import torch
from torch import nn

class AdjMask(nn.Module):
    def __init__(self, hidden_dim, adj_mask=None, **kwargs):
        super().__init__()
        self.c_attn = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.adj_mask = adj_mask
        # output projection
        self.c_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.hidden_dim = hidden_dim
        self.n_head = 1
        self.dropout = 0 #0.1
        self.resid_dropout = nn.Dropout(self.dropout)
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.hidden_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.adj_mask, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
        
    

if __name__ == "__main__":
    adj_mask = torch.tensor([[1,1,0],[1,1,1],[0,1,1]],dtype=torch.bool)  # Example adjacency mask for 3 nodes
    model = AdjMask(hidden_dim=32)
    print(model)
    x = torch.randn(3, 3, 32)  # (batch_size, seq_length, hidden_dim)
    output = model(x)
    print(output.shape)  # should be (4, 3, 32)
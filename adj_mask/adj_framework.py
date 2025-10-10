import torch
from torch import nn
from adj_mask import AdjMask
from qmix import QMix

class AdjFrame(nn.Module):
    def __init__(self, n_head ,agents, hidden_dim, q_hidden_dim, state_space, action_space, adj_mask=None, **kwargs):
        super().__init__()
        self.adj_mask_layer = AdjMask(n_head, hidden_dim, state_space, adj_mask=adj_mask, **kwargs)
        self.qmix = QMix(agents, state_space, action_space, q_hidden_dim, **kwargs)
    
    def forward(self, x):
        x = torch.stack([torch.tensor(state) for state in x.values()], dim=0).unsqueeze(0) # (1,3,31)
        x  = self.adj_mask_layer(x)
        actions, q_total = self.qmix(x)
        return self.process_actions(actions), q_total
    
    def process_actions(self, actions):
        # Convert tensor actions to dictionary format, assume batch size of 1
        action_dict = {f"walker_{i}": actions[0, i].detach().numpy() for i in range(actions.shape[1])}
        return action_dict
    
    def critic_loss(self, q_total, target_q_total):
        return nn.MSELoss()(q_total, target_q_total)
    
    def actor_loss(self, q_total, returns_to_go, log_probs):
        adv = q_total - returns_to_go
       
        return None
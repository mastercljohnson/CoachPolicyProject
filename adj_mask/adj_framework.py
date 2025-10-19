import torch
from torch import nn
from adj_mask import AdjMask
from qmix import QMix
import numpy as np

class AdjFrame(nn.Module):
    def __init__(self, n_head ,agents, hidden_dim, q_hidden_dim, state_space, action_space, adj_mask=None, **kwargs):
        super().__init__()
        self.adj_mask_layer = AdjMask(n_head, hidden_dim, state_space, adj_mask=adj_mask, **kwargs)
        self.qmix = QMix(agents, state_space, action_space, q_hidden_dim, **kwargs)
    
    def rollout(self, timesteps, env):
        rollout_states = {agent:[] for agent in env.agents}
        rollout_actions = {agent:[] for agent in env.agents}
        returns_to_go = {agent:[] for agent in env.agents}
        rtg = 0
        for t in range(timesteps):
            if t ==0 or termination_signal:
                observations, infos = env.reset()
                rtg = 0
            
            for agent in env.agents:
                rollout_states[agent].append(observations[agent])
            
            actions = self.act(observations)  # Forward pass
            actions = {agent: np.clip(actions[agent],-1.0,1.0).flatten() for agent in env.agents}  # Clip actions
            observations, rewards, terminations, truncations, infos = env.step(actions)
            rtg += sum(rewards.values()) if rewards else 0

            for agent in env.agents:
                rollout_actions[agent].append(actions[agent])
                returns_to_go[agent].append(rtg)

            termination_signal = any(terminations.values()) or any(truncations.values())
        
        return rollout_states, rollout_actions, returns_to_go


    def act(self, x):
        x = torch.stack([torch.tensor(state) for state in x.values()], dim=0).unsqueeze(0) # (1,3,31)
        x  = self.adj_mask_layer(x) # Use self attention with adjacency mask
        actions = self.qmix.act(x)
        return self.process_actions(actions)
    
    def process_actions(self, actions):
        # Convert tensor actions to dictionary format, assume batch size of 1
        action_dict = {f"walker_{i}": actions[i] for i in range(len(actions))}
        return action_dict
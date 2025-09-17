import torch
from torch import nn
class QMix(nn.Module):
    def __init__(self, num_agents, state_space, action_space, **kwargs):
        super().__init__()
        self.num_agents = num_agents
        for i in range(num_agents):
            setattr(self, f"agent_{i}_policy_network", nn.Linear(state_space, action_space))
    
    def forward(self, states):
        actions = []
        for i in range(self.num_agents):
            agent_policy_network = getattr(self, f"agent_{i}_policy_network")
            agent_action = agent_policy_network(states[:, i, :])
            actions.append(agent_action)
        actions = torch.stack(actions, dim=1)
        return actions
    
if __name__ == "__main__":
    num_agents = 3
    state_space = 10
    action_space = 4
    model = QMix(num_agents, state_space, action_space)
    print(model)
    states = torch.randn(5, num_agents, state_space)  # (batch_size, num_agents, state_space)
    output = model(states)
    print(output.shape)  # should be (5, num_agents, action_space)
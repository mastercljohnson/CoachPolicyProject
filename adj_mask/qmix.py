import torch
from torch import nn
class QMix(nn.Module):
    def __init__(self, num_agents, state_space, action_space, hidden_dim, **kwargs):
        super().__init__()
        self.num_agents = num_agents
        for i in range(num_agents):
            setattr(self, f"agent_{i}_policy_network", nn.Linear(state_space, action_space))
            setattr(self, f"agent_{i}_q_network", nn.Linear(state_space + action_space, 1))
        self.hyper_network_1 = nn.Linear(state_space, hidden_dim*num_agents) # W1 weights
        self.hyper_network_2 = nn.Linear(state_space, hidden_dim*num_agents) # bias?
        self.hyper_network_3 = nn.Linear(state_space, 1) # W2 weights
        self.hyper_network_4 = nn.Linear(state_space, 1) # bias2?
        self.hyper_network_5 = nn.Linear(state_space, 1) # bias2?
    
    def forward(self, states):
        actions = []
        q_values = []
        for i in range(self.num_agents):
            agent_policy_network = getattr(self, f"agent_{i}_policy_network")
            agent_action = agent_policy_network(states[:, i, :])
            actions.append(agent_action)
            agent_q_network = getattr(self, f"agent_{i}_q_network")
            agent_q_input = torch.cat([states[:, i, :], agent_action], dim=-1)
            agent_q_value = agent_q_network(agent_q_input)
            q_values.append(agent_q_value)
        actions = torch.stack(actions, dim=1)
        q_values = torch.stack(q_values, dim=-1)

        hyper_w1 = torch.abs(self.hyper_network_1(states).view(-1, self.num_agents, -1))  # (batch_size, num_agents, hidden_dim)
        hyper_b1 = self.hyper_network_2(states).view(-1, 1, -1)  # (batch_size, 1, hidden_dim)
        hidden = torch.relu(torch.bmm(q_values.unsqueeze(1), hyper_w1) + hyper_b1)
        hyper_w2 = torch.abs(self.hyper_network_3(states).view(-1, hidden.size(-1), 1))  # (batch_size, hidden_dim, 1)
        hyper_b2 = self.hyper_network_4(states).view(-1, 1, 1)  # (batch_size, 1, 1)
        q_total = torch.bmm(hidden, hyper_w2) + hyper_b2  # (batch_size, 1, 1)
        q_total = q_total.squeeze(-1).squeeze(-1)  # (batch_size,)


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
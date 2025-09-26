import torch
from torch import nn
class QMix(nn.Module):
    def __init__(self, agents, state_space, action_space, hidden_dim, **kwargs):
        super().__init__()
        self.num_agents = len(agents)
        self.state_space = sum([state_space[agent].shape[0] for agent in agents])
        self.hidden_dim = hidden_dim
        for i, agent in enumerate(agents):
            # print(state_space[agent].shape[0], action_space[agent].shape[0])
            setattr(self, f"agent_{i}_policy_network", nn.Linear(state_space[agent].shape[0], action_space[agent].shape[0]))
            setattr(self, f"agent_{i}_q_network", nn.Linear(state_space[agent].shape[0] + action_space[agent].shape[0], 1))
        self.hyper_network_weight_1 = nn.Linear(self.state_space, hidden_dim*self.num_agents) # W1 weights
        self.hyper_network_bias_1 = nn.Linear(self.state_space, hidden_dim) # bias
        self.hyper_network_weight_2 = nn.Linear(self.state_space, hidden_dim) # W2 weights
        self.hyper_network_bias_2_1 = nn.Linear(self.state_space, hidden_dim) # bias2
        self.hyper_network_bias_2_2 = nn.Linear(hidden_dim, 1) # bias2
    
    def forward(self, states):
        actions = []
        q_values = []
        for i in range(self.num_agents):
            agent_policy_network = getattr(self, f"agent_{i}_policy_network")
            agent_action = agent_policy_network(states[:, i, :])
            # print("agent_action", agent_action.shape)
            actions.append(agent_action)
            agent_q_network = getattr(self, f"agent_{i}_q_network")
            agent_q_input = torch.cat([states[:, i, :], agent_action], dim=-1)
            # print("agent_q_input", agent_q_input.shape)
            agent_q_value = agent_q_network(agent_q_input)
            # print("agent_q_value", agent_q_value.shape)
            q_values.append(agent_q_value)
        actions = torch.stack(actions, dim=1)
        # print(q_values)
        q_values = torch.cat(q_values, dim=-1)
        # print("q_values", q_values.shape)


        hyper_state = torch.cat([states[:, i, :] for i in range(self.num_agents)], dim=-1)
        hyper_w1 = torch.abs(self.hyper_network_weight_1(hyper_state).view(-1, self.num_agents, self.hidden_dim))  # (batch_size, self.num_agents, hidden_dim)
        hyper_b1 = self.hyper_network_bias_1(hyper_state).view(-1, 1, self.hidden_dim)  # (batch_size, 1, hidden_dim)
        # print(q_values.unsqueeze(0).shape, hyper_w1.shape, hyper_b1.shape)
        hidden = torch.relu(torch.bmm(q_values.unsqueeze(0), hyper_w1) + hyper_b1)
        hyper_w2 = torch.abs(self.hyper_network_weight_2(hyper_state).view(-1, self.hidden_dim, 1))  # (batch_size, hidden_dim, 1)
        hyper_b2_prep = torch.relu(self.hyper_network_bias_2_1(hyper_state).view(-1, self.hidden_dim))  # (batch_size, hidden dim)
        hyper_b2 = self.hyper_network_bias_2_2(hyper_b2_prep).view(-1, 1)  # (batch_size, 1, 1)
        # print("hidden", hidden.shape, hyper_w2.shape, hyper_b2.shape)
        q_total = torch.bmm(hidden, hyper_w2) + hyper_b2  # (batch_size,  1)
        # print("q_total", q_total.shape)
        q_total = q_total.squeeze(-1) # (batch_size,)
        # print("q_total_after_squeeze", q_total.shape)



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
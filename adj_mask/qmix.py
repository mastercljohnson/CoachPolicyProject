import torch
from torch import nn
from torch.distributions import MultivariateNormal


class QMix(nn.Module):
    def __init__(self, agents, state_space, action_space, hidden_dim, **kwargs):
        super().__init__()
        self.num_agents = len(agents)
        self.state_space = sum([state_space[agent].shape[0] for agent in agents])
        self.hidden_dim = hidden_dim
        # Initialize the covariance matrix used to query the actor for actions
		# self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		# self.cov_mat = torch.diag(self.cov_var)
        for i, agent in enumerate(agents):
            setattr(self, f"agent_{i}_policy_network", nn.Linear(self.hidden_dim, action_space[agent].shape[0]))
            setattr(self, f"agent_{i}_q_network", nn.Linear(self.hidden_dim, 1))
            # setattr(self, f"agent_{i}_p_optimizer", torch.optim.AdamW(getattr(self, f"agent_{i}_policy_network").parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay))
            # setattr(self, f"agent_{i}_q_optimizer", torch.optim.AdamW(getattr(self, f"agent_{i}_q_network").parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay))
        self.hyper_network_weight_1 = nn.Linear(self.hidden_dim*self.num_agents, hidden_dim*self.num_agents) # W1 weights
        self.hyper_network_bias_1 = nn.Linear(self.hidden_dim*self.num_agents, hidden_dim) # bias
        self.hyper_network_weight_2 = nn.Linear(self.hidden_dim*self.num_agents, hidden_dim) # W2 weights
        self.hyper_network_bias_2_1 = nn.Linear(self.hidden_dim*self.num_agents, hidden_dim) # bias2

        self.hyper_network_bias_2_2 = nn.Linear(hidden_dim, 1) # bias2

        # self.hyper_network_optimizer = torch.optim.AdamW(list(self.hyper_network_weight_1.parameters()) + list(self.hyper_network_bias_1.parameters()) + list(self.hyper_network_weight_2.parameters()) + list(self.hyper_network_bias_2_1.parameters()) + list(self.hyper_network_bias_2_2.parameters()), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    
    def forward(self, states):
        actions = []
        q_values = []
        for i in range(self.num_agents):
            agent_policy_network = getattr(self, f"agent_{i}_policy_network")
            agent_action, log_prob = self.get_action(states[:, i, :], i)
            actions.append(agent_action)
            agent_q_network = getattr(self, f"agent_{i}_q_network")
            agent_q_input = states[:, i, :]
            agent_q_value = agent_q_network(agent_q_input)
            q_values.append(agent_q_value)
        q_values = torch.cat(q_values, dim=-1)


        hyper_state = torch.cat([states[:, i, :] for i in range(self.num_agents)], dim=-1)
        hyper_w1 = torch.abs(self.hyper_network_weight_1(hyper_state).view(-1, self.num_agents, self.hidden_dim))  # (batch_size, self.num_agents, hidden_dim)
        hyper_b1 = self.hyper_network_bias_1(hyper_state).view(-1, 1, self.hidden_dim)  # (batch_size, 1, hidden_dim)
        hidden = torch.relu(torch.bmm(q_values.unsqueeze(0), hyper_w1) + hyper_b1)
        hyper_w2 = torch.abs(self.hyper_network_weight_2(hyper_state).view(-1, self.hidden_dim, 1))  # (batch_size, hidden_dim, 1)
        hyper_b2_prep = torch.relu(self.hyper_network_bias_2_1(hyper_state).view(-1, self.hidden_dim))  # (batch_size, hidden dim)
        hyper_b2 = self.hyper_network_bias_2_2(hyper_b2_prep).view(-1, 1)  # (batch_size, 1, 1)
        q_total = torch.bmm(hidden, hyper_w2) + hyper_b2  # (batch_size,  1)
        q_total = q_total.squeeze(-1) # (batch_size,)

        return actions, q_total

    def get_action(self, state, agent_index):
        agent_policy_network = getattr(self, f"agent_{agent_index}_policy_network")
        action_mean = agent_policy_network(state)
        action_dist = MultivariateNormal(action_mean, torch.eye(action_mean.shape[1])) # set covariance matrix later?
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()
    
    def learn(self,agent_index, reward):
        td_diff = reward + gamma * next_q_value - current_q_value
        A_t = td_diff  # Advantage estimate
        pass
    
if __name__ == "__main__":
    num_agents = 3
    state_space = 10
    action_space = 4
    model = QMix(num_agents, state_space, action_space)
    print(model)
    states = torch.randn(5, num_agents, state_space)  # (batch_size, num_agents, state_space)
    output = model(states)
    print(output.shape)  # should be (5, num_agents, action_space)
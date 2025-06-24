import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from transformer_decoder import TransformerDecoder  # Assuming you have a DecisionTransformer class defined in transformer_decoder.py
# import torch_directml

def order_comms_data_by_receiver(comms_data):
    receiver_comms = {}
    for sender, recipient_data in comms_data.items():
        for recipient, message in recipient_data.items():
            if recipient not in receiver_comms:
                receiver_comms[recipient] = {}
            receiver_comms[recipient][sender] = message
    return receiver_comms

def message_to_action_converter(message):
    # Placeholder function to convert a received communication message to an action
    # This could involve parsing the message and determining the action based on its content
    return np.zeros(4)  # Example action, replace with actual logic

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, num_layers, max_ep_len=4096,observation_spaces=None, action_spaces=None, action_tanh=True):
        super(DecisionTransformer, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size # timestep embedding size
        self.num_layers = num_layers
        self.max_ep_len = max_ep_len
        self.observation_spaces = observation_spaces  # Placeholder for observation spaces
        self.action_spaces = action_spaces        # Placeholder for action spaces
        
        # Define the embeddings for RL here, basically cast all things to embedding size
        self.embed_timestep = nn.Embedding(max_ep_len, self.hidden_size)
        self.embed_return = torch.nn.Linear(1, self.hidden_size) # lol scalar to vector
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.action_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        # tanh scales between -1 and 1, which is useful for some action spaces?
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        
        # Define the transformer layer here, assume context window is max_ep_len* 3* timesteps (R_t,s_t, a_t) is a single timestep 
        self.transformer = TransformerDecoder(num_heads=3, num_input_tokens=3*max_ep_len, encode_dim=hidden_size, key_dim=hidden_size // 3, value_dim=hidden_size // 3,num_blocks=2)

    # Just copy the implementation from the Decision Transformer paper code
    # rewards doesnt seem like its used here
    def forward(self, obs, actions, rewards,  returns_to_go, timesteps, attention_mask=None):
        # Implement the forward pass for the Decision Transformer

        batch_size, seq_length = obs.shape[0], obs.shape[1]

        # TODO: not sure what this attention mask is for?
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(obs)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # TODO: why this permutation way?
        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs) # apply layernorm over inputs

        # questions here about attention mask
        transformer_outputs = self.transformer.forward(stacked_inputs)
        
        x = transformer_outputs.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # The last hidden state is used for prediction, this is the output of the current implementation I think
        action_preds = self.predict_action(x[:,1])  # predict next action given state


        return {agent_id: action_space.sample() for agent_id, action_space in self.action_spaces.items()}

class Agent:
    def __init__(self, agent_id, observation_space, action_space, agents, disable_comms=False):
        self.agent_id = agent_id
        self.observation_space = observation_space  # Placeholder for observation space
        self.action_space = action_space        # Placeholder for action space
        self.num_agents = len(agents)  # Number of agents in the environment
        self.index_to_agent_id_map = {index: agent_id for index, agent_id in enumerate(agents)}  # Map agent IDs to indices
        self.disable_comms = disable_comms  # Flag to disable communication if needed
        
        # Initialize agent-specific parameters here

        # Initialize decision making components here
        #  Trust matrix ordered by agent_id
        self.trust_vector = torch.randn( self.num_agents + 1) # +1 for coach

    def communicate(self, obs, other_agents, coach_suggestion = None):
        # Implement the communication logic with other agents
        # This could involve sharing observations, actions, or other relevant information
        comms = {}
        for agent in other_agents:
            if agent != self.agent_id:
                comms[agent] = "sample message from " + self.agent_id  # Placeholder for communication message
        return comms

    def act(self, obs, comms, coach_suggestion, decision_transformer_overide=False):
        # Implement the action selection logic for the agent

        if decision_transformer_overide:
            # Use coach policy action.
            return coach_suggestion


        self_action = self.action_space.sample()  # Placeholder for the agent's own action

        # Here we add the comms and trust portion affecting the decision making
        
        #  Comms portion
        if self.disable_comms or comms is None:
            # If communications are disabled or no comms are provided, return the agent's own action
            return self_action
        
        comms_to_actions = {}
        for sender, message in comms.items():
            # Convert the received message to an action
            action = message_to_action_converter(message)
            comms_to_actions[sender] = action
        
        #  Trust portions
        trust_to_probability = F.softmax(self.trust_vector, dim=0)

        #  Create categorical distribution based on trust vector
        #  TODO: One direction of research is to play with the probability distribution of trust?
        #  TODO: What if multiple agent comms are taken into account for decision making, how would I do this?
        dist = Categorical(probs=trust_to_probability)
        sampled_index = dist.sample().item()  # Sample an index based on the trust vector. item() converts the tensor to a Python int
        
        if sampled_index == self.num_agents:
            # If the sampled index corresponds to the coach(index == #agents), return the coach's suggestion
            return coach_suggestion
        else:
            selected_agent_action_overide = self.index_to_agent_id_map[sampled_index]
            if selected_agent_action_overide == self.agent_id:
                # If the selected agent is this agent, return its own action
                return self_action
            else:
                return comms_to_actions[selected_agent_action_overide]

class DTCG(nn.Module):
    def __init__(self, agents, observation_spaces=None, action_spaces=None, ignore_single_agents=False, ignore_interagent_comms=False):
        super().__init__()
        # Initialize agents
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.agents = {agent_id: Agent(agent_id, self.observation_spaces[agent_id], self.action_spaces[agent_id], agents) for agent_id in agents}
        self.ignore_single_agents = ignore_single_agents
        self.ignore_interagent_comms = ignore_interagent_comms
        


        # Initialize Decision Transformer here
        self.decision_transformer = DecisionTransformer(
            state_dim=128,  # Example state dimension
            action_dim=4,   # Example action dimension
            hidden_dim=256, # Example hidden dimension
            num_layers=6,    # Example number of layers
            max_ep_len=4096,  # Example maximum episode length
            observation_spaces=self.observation_spaces,
            action_spaces=self.action_spaces
        )

    def act(self, total_obs):
        suggested_actions = self.decision_transformer.forward(total_obs)
        total_actions = {}
        
        # communication between agents
        if not self.ignore_interagent_comms:
            comm_data = {}
            for agent_id in self.agents:
                comm_data[agent_id] = self.agents[agent_id].communicate(
                    obs=total_obs[agent_id],
                    other_agents=list(self.agents.keys()),
                    coach_suggestion=suggested_actions[agent_id]
                )

            # Order communication data by recipient
            ordered_comms = order_comms_data_by_receiver(comm_data)

            for agent_id, agent in self.agents.items():
                action = agent.act(
                    obs=total_obs[agent_id],
                    comms=ordered_comms[agent_id],  # Placeholder for communication data
                    coach_suggestion=suggested_actions[agent_id],
                    decision_transformer_overide=self.ignore_single_agents
                )
                total_actions[agent_id] = action
        else:
             for agent_id, agent in self.agents.items():
                action = agent.act(
                    obs=total_obs[agent_id],
                    comms=None,  # Placeholder for communication data
                    coach_suggestion=suggested_actions[agent_id],
                    decision_transformer_overide=self.ignore_single_agents
                )
                total_actions[agent_id] = action
        return total_actions
        
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from decision_transformer.decision_transformer import DecisionTransformer
from collections import deque
from itertools import islice

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
        
        if sampled_index == self.num_agents and coach_suggestion is not None:
            # If the sampled index corresponds to the coach(index == #agents), return the coach's suggestion
            return coach_suggestion
        else:
            while sampled_index == self.num_agents:
                sampled_index = dist.sample().item()

            selected_agent_action_overide = self.index_to_agent_id_map[sampled_index]
            if selected_agent_action_overide == self.agent_id:
                # If the selected agent is this agent, return its own action
                return self_action
            else:
                return comms_to_actions[selected_agent_action_overide]

class DTCG(nn.Module):
    def __init__(self, agents, observation_spaces=None, action_spaces=None, ignore_single_agents=False, ignore_interagent_comms=False, replay_buffer_size=1000):
        super().__init__()
        # Initialize agents
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.agents = {agent_id: Agent(agent_id, self.observation_spaces[agent_id], self.action_spaces[agent_id], agents) for agent_id in agents}
        self.sample_obs_space = self.observation_spaces[agents[0]]
        self.sample_action_space = self.action_spaces[agents[0]]
        self.ignore_single_agents = ignore_single_agents
        self.ignore_interagent_comms = ignore_interagent_comms

        #  What is a good size for buffers?
        self.state_buffer = deque(maxlen=replay_buffer_size)
        self.action_buffer = deque(maxlen=replay_buffer_size)
        self.cumulative_reward_so_far = deque(maxlen=replay_buffer_size)
        self.returns_to_go_buffer = deque(maxlen=replay_buffer_size)
        self.timestep_buffer = deque(maxlen=replay_buffer_size)
        # print("State Buffer Shape:", len(self.state_buffer), "Maxlen:", self.state_buffer.maxlen)
        


        # Initialize Decision Transformer here
        self.decision_transformer = DecisionTransformer(
            state_dim=len(agents) * self.sample_obs_space.shape[0],  # State dimension is num_agents * state_dim, shape[0] is num of dims for featurespace
            act_dim=len(agents) * self.sample_action_space.shape[0],   # Example action dimension
            hidden_size=256, # Example hidden dimension
            num_heads= 4,  # Example number of heads , code later asserts hidden_size % num_heads == 0
            # num_layers=6,    # Example number of layers
            max_ep_len=4096,  # Example maximum episode length
            observation_spaces=self.observation_spaces,
            action_spaces=self.action_spaces
        )
    
    def load_state_buffer(self, states):
        dt_combined_obs = np.concatenate([states[agent_id] for agent_id in sorted(states.keys())], axis=-1)
        self.state_buffer.append(dt_combined_obs)
    
    # TODO: check if the indexes for calculating returns align for returns_to_go_buffer
    def load_returns_to_go(self, cumulative_return, env_steps):
        for timestep in range(env_steps-1, -1, -1):
            # print(timestep)
            self.returns_to_go_buffer.append(cumulative_return - self.cumulative_reward_so_far[-1 - timestep]) 
        
        # print("Returns to go buffer length:", len(self.returns_to_go_buffer))
        # print("State buffer length:", len(self.state_buffer))
        # print("Action buffer length:", len(self.action_buffer))
        
        # print("Returns to go buffer:", self.returns_to_go_buffer)
        # print("State buffer:", self.state_buffer)
        # print("Action buffer:", self.action_buffer)

    def act(self, total_obs, cumulative_rewards, env_step=0):
        # print("Total Obs Shape:", {agent: obs.shape for agent, obs in total_obs.items()})
        # print("Total Obs:", total_obs)

        self.cumulative_reward_so_far.append(cumulative_rewards)

        self.load_state_buffer(total_obs)
        self.timestep_buffer.append(env_step)
        # dt_combined_obs = np.concatenate([total_obs[agent_id] for agent_id in sorted(total_obs.keys())], axis=-1)
        # self.state_buffer.append(dt_combined_obs)


        # print("State Buffer Length:", len(self.state_buffer))
        # print("State Buffer:", self.state_buffer)

        suggested_actions = None # Set to None initially

        # Get actions from replay buffer episode
        sample_num = 30
        if len(self.returns_to_go_buffer) >= sample_num:
            start_index = np.random.randint(0, len(self.returns_to_go_buffer) - sample_num + 1)
            end_index = start_index + sample_num - 1

            states = torch.from_numpy(np.array(list(islice(self.state_buffer, start_index, end_index)))).float()
            actions = torch.from_numpy(np.array(list(islice(self.action_buffer, start_index, end_index)))).float()
            returns_to_go = torch.from_numpy(np.array(list(islice(self.returns_to_go_buffer, start_index, end_index)))).float() # need to reshape this to be (N, 1)
            returns_to_go = returns_to_go.unsqueeze(-1)
            timesteps = torch.from_numpy(np.array(list(islice(self.timestep_buffer, start_index, end_index)))).long() # long for embedding layer

            suggested_actions = self.decision_transformer.forward(states, actions, None, returns_to_go, timesteps)  # TODO: Do we need attention mask?
        
        
        total_actions = {}
        
        # communication between agents
        if not self.ignore_interagent_comms:
            comm_data = {}
            for agent_id in self.agents:
                comm_data[agent_id] = self.agents[agent_id].communicate(
                    obs=total_obs[agent_id],
                    other_agents=list(self.agents.keys()),
                    coach_suggestion=suggested_actions[agent_id] if suggested_actions else None
                )

            # Order communication data by recipient
            ordered_comms = order_comms_data_by_receiver(comm_data)

            for agent_id, agent in self.agents.items():
                action = agent.act(
                    obs=total_obs[agent_id],
                    comms=ordered_comms[agent_id],  # Placeholder for communication data
                    coach_suggestion=suggested_actions[agent_id] if suggested_actions else None,
                    decision_transformer_overide=self.ignore_single_agents
                )
                total_actions[agent_id] = action
        else:
             for agent_id, agent in self.agents.items():
                action = agent.act(
                    obs=total_obs[agent_id],
                    comms=None,  # Placeholder for communication data
                    coach_suggestion=suggested_actions[agent_id] if suggested_actions else None,
                    decision_transformer_overide=self.ignore_single_agents
                )
                total_actions[agent_id] = action
        
        # print("Total Actions:", total_actions)
        # Store actions in the replay buffer
        self.action_buffer.append(np.concatenate([total_actions[agent_id] for agent_id in sorted(total_actions.keys())], axis=-1))

        return total_actions
        
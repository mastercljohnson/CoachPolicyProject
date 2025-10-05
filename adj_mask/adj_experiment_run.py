from pettingzoo.sisl import multiwalker_v9
from adj_framework import AdjFrame
import torch
import numpy as np

# observation is a dictionary with keys as agent names and values as their respective observations
# action is a dictionary with keys as agent names and values as their respective actions
# Not sure what infos is

env = multiwalker_v9.parallel_env(render_mode="human",terminate_on_fall=True)
observations, infos = env.reset()


# hidden dim right now needs to match observation dimensions
algo = AdjFrame(3, env.agents,60,60, env.observation_spaces, env.action_spaces)

rewards = None

total_steps = 10000000

backprop_steps = 10

env_step = 0
cumulative_rewards = 0
termination_signal = False

lr = 1e-3 # learning_rate / 10 usually
beta1 = 0.9 # momentum term for Adam optimizer
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
weight_decay = 1e-2 # L2 regularization term

optimizer = torch.optim.AdamW(algo.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

# for name, param in algo.named_parameters():
#     print(f"Algo Parameter to optimize {name} with shape {param.shape}")

local_step = 0

while env_step < total_steps:

    if termination_signal:
        #  Load the terminal state, I dont think we need to do this

        # Reset the environment
        observations, infos = env.reset()
        termination_signal = False
        cumulative_rewards = 0
        rewards = None
        local_step = 0
        

    actions, q_total = algo.forward(observations)  # Forward pass
    actions = {agent: np.clip(actions[agent],-1.0,1.0) for agent in env.agents}  # Filter actions for active agents
    # print("Actions from the model:", actions)
    
    # actions = algo.act(observations, cumulative_rewards, env_step)  # Get actions from the DTCG algorithm
    # actions = {agent: env.action_space(agent).sample() for agent in env.agents} # this is a dictionary
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env_step += 1
    local_step += 1

    total_rewards = sum(rewards.values()) if rewards else 0 # place holder for total rewards 
    cumulative_rewards += total_rewards

    
    loss = algo.loss(q_total, torch.tensor([cumulative_rewards], dtype=torch.float32).unsqueeze(0))  # Compute loss
    if local_step == 50:
        print("Loss:", loss.item())
    
    algo.train()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    

    termination_signal = any(terminations.values()) or any(truncations.values())

    

env.close()
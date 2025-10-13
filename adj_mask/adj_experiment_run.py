from pettingzoo.sisl import multiwalker_v9
from adj_framework import AdjFrame
import torch
import numpy as np
import math

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./runs/exp1")

# observation is a dictionary with keys as agent names and values as their respective observations
# action is a dictionary with keys as agent names and values as their respective actions
# Not sure what infos is

env = multiwalker_v9.parallel_env(render_mode="human",terminate_on_fall=True)
observations, infos = env.reset()


# hidden dim right now needs to match observation dimensions
algo = AdjFrame(3, env.agents,60,60, env.observation_spaces, env.action_spaces)

rewards = None

total_steps = 1000000

backprop_steps = 10

env_step = 0
cumulative_rewards = 0
termination_signal = False

learning_rate = 5e-4 # learning_rate / 10 usually
beta1 = 0.9 # momentum term for Adam optimizer
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
weight_decay = 1e-2 # L2 regularization term

lr_decay_iters = 1000000 # make equal to max_iters usually
warmup_iters = 10000 # usually 0.1 * lr_decay_iters
min_lr = 5e-5 # learning_rate / 10 usually

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

optimizer = torch.optim.AdamW(algo.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

# for name, param in algo.named_parameters():
#     print(f"Algo Parameter to optimize {name} with shape {param.shape}")

local_step = 0
episode = 0
q_acc = 0

while env_step < total_steps:

    if termination_signal:
        #  Log to TensorBoard when an episode ends
        writer.add_scalar("Scaled Reward per timestep/episode", cumulative_rewards/local_step, episode)
        writer.add_scalar("Loss/episode", q_acc, episode)

        # Reset the environment
        episode += 1
        observations, infos = env.reset()
        termination_signal = False
        cumulative_rewards = 0
        rewards = None
        local_step = 0
        q_acc = 0
        

    actions, q_total = algo.forward(observations)  # Forward pass
    actions = {agent: np.clip(actions[agent],-1.0,1.0).flatten() for agent in env.agents}  # Clip actions
    
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env_step += 1
    local_step += 1

    total_rewards = sum(rewards.values()) if rewards else 0 # place holder for total rewards 
    cumulative_rewards += total_rewards

    
    loss = algo.loss(q_total, torch.tensor([cumulative_rewards], dtype=torch.float32).unsqueeze(0))  # Compute loss
    q_acc += loss.item()
    
    algo.train()
    for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(env_step)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    

    termination_signal = any(terminations.values()) or any(truncations.values())

    

env.close()
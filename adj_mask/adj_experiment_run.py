from pettingzoo.sisl import multiwalker_v9
from adj_framework import AdjFrame
import torch
import numpy as np
import math

#  Boiler plate code for tensorboard logging, keeping here for intended future usage

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir="./runs/exp1")
#  Log to TensorBoard when an episode ends
# writer.add_scalar("Scaled Reward per timestep/episode", cumulative_rewards/local_step, episode)
# writer.add_scalar("Loss/episode", q_acc, episode)

# observation is a dictionary with keys as agent names and values as their respective observations
# action is a dictionary with keys as agent names and values as their respective actions
env = multiwalker_v9.parallel_env(render_mode="human",terminate_on_fall=True)
observations, infos = env.reset()


# hidden dim right now needs to match observation dimensions
algo = AdjFrame(3, env.agents,60,60, env.observation_spaces, env.action_spaces)

#  Test rollout
algo.rollout(300, env)
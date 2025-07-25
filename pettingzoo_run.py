from pettingzoo.sisl import multiwalker_v9
from dtcg import DTCG

# observation is a dictionary with keys as agent names and values as their respective observations
# action is a dictionary with keys as agent names and values as their respective actions
# Not sure what infos is

env = multiwalker_v9.parallel_env(render_mode="human")
observations, infos = env.reset()

# print(env.agents)  # List of agents in the environment
# print(env.action_spaces)
# print(env.observation_spaces)



algo = DTCG(env.agents, observation_spaces=env.observation_spaces, action_spaces=env.action_spaces)

while env.agents:
    # print("Observations:", observations)
    # this is where you would insert your policy
    actions = algo.act(observations)  # Get actions from the DTCG algorithm
    actions = {agent: env.action_space(agent).sample() for agent in env.agents} # this is a dictionary
    # print("Actions:", actions)

    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
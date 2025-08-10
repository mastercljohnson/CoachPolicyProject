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

rewards = None

total_steps = 200

env_step = 0
cumulative_rewards = 0
termination_signal = False
while env.agents and env_step < total_steps:
    # print("Observations:", observations)
    # print(f"Rewards {rewards}")
    # this is where you would insert your policy
    total_rewards = sum(rewards.values()) if rewards else 0 # place holder for total rewards 
    cumulative_rewards += total_rewards

    actions = algo.act(observations, cumulative_rewards, env_step)  # Get actions from the DTCG algorithm
    actions = {agent: env.action_space(agent).sample() for agent in env.agents} # this is a dictionary
    # print("Actions:", actions)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env_step += 1
    termination_signal = any(terminations.values()) or any(truncations.values())

    if termination_signal:
        print("Terminates or truncates detected.")
        #  Load the terminal state, I dont think we need to do this
        # algo.load_state_buffer(observations)
        # Calculate returns to go
        algo.load_returns_to_go(cumulative_rewards, env_step)

        # Reset the environment
        observations, infos = env.reset()
        termination_signal = False
        cumulative_rewards = 0
        env_step = 0

env.close()
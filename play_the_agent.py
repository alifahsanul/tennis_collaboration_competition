from unityagents import UnityEnvironment
import numpy as np
from agent import Agent

env = UnityEnvironment(file_name="/home/alif/Documents/deeprl-udacity/deep-reinforcement-learning/p3_collab-compet/Tennis_Linux/Tennis")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

agent = Agent(device = 'cpu',
                        state_size = state_size,
                        n_agents = num_agents,
                        action_size = action_size
                        )

for episode in range(3):
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    score = np.zeros(num_agents)
    while True:
        actions = agent.act(states, add_noise=False)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        score += rewards
        states = next_states
        if np.any(dones): break
    print('Episode: {} Score: {:.2f}'.format(episode, np.mean(score)))

env.close()

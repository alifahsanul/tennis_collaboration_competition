from unityagents import UnityEnvironment
import numpy as np
import torch
from agent import Agent
from collections import deque
import os
import pandas as pd
import csv


env = UnityEnvironment(file_name="/home/alif/Documents/deeprl-udacity/deep-reinforcement-learning/p3_collab-compet/Tennis_Linux/Tennis")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)


# size of each action
action_size = brain.vector_action_space_size


# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]

folder_path = './'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = Agent(device, state_size, num_agents, action_size, folder_path)
def print_info():
    print('Number of agents:', num_agents)
    print('Size of each action:', action_size)
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])
    print('run on {}'.format(device))
# print_info()
if os.path.isfile('score/scores.csv'):
    with open('score/scores.csv', 'r') as f:
        reader = csv.reader(f)
        scores = list(reader)
        # print(reader)
        # raise ValueError
        # if np.isscalar(scores[0]) == True: raise ValueError
        # print(scores)
        scores = [float(x[0]) for x in scores]
    start_eps = len(scores)
else:
    start_eps = 0
    scores = []
print('episode start at {}'.format(start_eps))
scores_window = deque([0] * 100, maxlen=100)
for i in range(min(100-1, len(scores)-1), 0-1, -1):
    scores_window[i] = scores[i]
delta_eps = 424
if delta_eps <= 100: raise ValueError
input('press enter continue ...')
n_episodes = start_eps + delta_eps
for episode in range(start_eps, n_episodes):
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    agent.reset()
    score = np.zeros(num_agents)
    play_counter = 0
    while True:
        actions = agent.act(states)
        actions = np.clip(actions, -1, 1)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        agent.step(states, actions, rewards, next_states, dones)
        score += rewards
        states = next_states
        play_counter += 1
        if np.any(dones):
            break
    if episode%5==0: agent.checkpoint()
    scores.append(np.mean(score))
    if episode%20 == 0: np.savetxt('score/scores.csv', scores, delimiter = ',')
    scores_window.append(np.mean(score))
    print('\rEps {} Score: {:.3f} Avg Score: {:.3f} Count: {:d}\n'.format(episode, np.mean(score), np.mean(scores_window), play_counter), end="")
    if np.mean(scores_window) >= 0.5:
        np.savetxt('score/scores.csv', scores, delimiter=',')
        np.savetxt('scores_window.csv', scores_window, delimiter=',')
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}\n'.format(episode, np.mean(scores_window)))
        if np.mean(scores_window) > 0.6:
            break
np.savetxt('score/scores.csv', scores, delimiter = ',')












































































# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 10:26:30 2019

@author: alifahsanul
"""

from unityagents import UnityEnvironment
import numpy as np
import pandas as pd
# select this option to load version 1 (with a single agent) of the environment
# env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')

# select this option to load version 2 (with 20 agents) of the environment
env = UnityEnvironment(file_name='/home/alif/Documents/deeprl-udacity/deep-reinforcement-learning/p2_continuous-control/Reacher_Linux/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

def print_info():
    print('Number of agents:', num_agents)
    print('Size of each action:', action_size)
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])
#print_info()

import torch
from agent import Agent
from collections import deque

folder_path = './'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('run on {}'.format(device))
agent = Agent(device, state_size, num_agents, action_size, folder_path)

scores = []
scores_window = deque([0] * 100, maxlen=100)
n_episodes = 3000

for episode in range(n_episodes):
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    agent.reset()
    score = np.zeros(num_agents)
    while True:
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        agent.step(states, actions, rewards, next_states, dones)
        score += rewards
        states = next_states
        if np.any(dones):
            break
    agent.checkpoint()
    scores.append(np.mean(score))
    scores_window.append(np.mean(score))
    # print(scores_window)
    # raise ValueError
    print('\rEpisode {} Score: {:.2f} Average Score: {:.2f}\n'.format(episode, np.mean(score), np.mean(scores_window)), end="")
    if np.mean(scores_window) >= 30.0:
        np.savetxt('scores.csv', scores, delimiter=',')
        np.savetxt('scores_window.csv', scores_window, delimiter=',')
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\n'.format(episode, np.mean(scores_window)))
        if np.mean(scores_window) > 50.0:
            break

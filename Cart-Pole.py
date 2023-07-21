import gymnasium as gym
from Qalgo import *
import numpy as np
import os
import random

env = gym.make('CartPole-v1', render_mode="human")
#observation, info = env.reset(seed=42)

alpha = 0.15
gamma = 0.95

num_explorations = 10000
num_exploitations = 1000

#num_bins = [10] * env.observation_space.shape[0]
num_bins = [20,10,20,10]

state_bins = get_state_bins(env, num_bins)

observation, info = env.reset()

initial_state = discretize_state(observation, state_bins)

# Initialize the Q-table with zeros (numbins and numactions)
q_table = np.zeros((*num_bins, env.action_space.n))

if not os.path.exists("CartPole_q_table.npy"):
    explore(num_explorations, alpha, gamma, q_table, state_bins, env, initial_state)

exploit(num_exploitations, env, state_bins)

'''for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
'''
env.close()
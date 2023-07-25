import gymnasium as gym
from Qalgo import *
import numpy as np
import os
import random

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
#observation, info = env.reset(seed=42)

name = "FrozenLake"

alpha = 0.15
gamma = 0.95

num_explorations = 1000
num_exploitations = 1000

observation, info = env.reset()

initial_state = observation

# Initialize the Q-table with zeros (numbins and numactions)
q_table = np.zeros((env.observation_space.n, env.action_space.n))

if not os.path.exists(name+"_q_table.npy"):
    explore_discrete_observation(num_explorations, alpha, gamma, q_table, env, initial_state, name)

exploit_discrete_observation(num_exploitations, env, name)

env.close()
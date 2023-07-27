import gymnasium as gym
from Qalgo import *
import numpy as np
import os
import random

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")#, render_mode="human"
#observation, info = env.reset(seed=42)

name = "FrozenLake"

alpha = 0.15
gamma = 0.95
epsilon = 1
epsilon_decay = 0.999999
epsilon_min = 0.15

num_explorations = 1000000
num_exploitations = 100

observation, info = env.reset()

initial_state = observation

# Initialize the Q-table with zeros (numbins and numactions)
q_table = {}

if not os.path.exists(name+"_q_table.json"):
    explore_discrete_observation(num_explorations, alpha, gamma, epsilon, epsilon_decay, epsilon_min, q_table, env, initial_state, name)

exploit_discrete_observation(num_exploitations, env, name)

env.close()
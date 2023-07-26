import gymnasium as gym
from Qalgo import *
import numpy as np
import os
import random

env = gym.make('CartPole-v1', render_mode="human")#, render_mode="human"
#observation, info = env.reset(seed=42)

name = "CartPole"

alpha = 0.15
gamma = 0.95
epsilon = 1
epsilon_decay = 0.999999
epsilon_min = 0.15

num_explorations = 4000000
num_exploitations = 1000

num_bins = [50] * env.observation_space.shape[0]

state_bins = get_state_bins(env, num_bins)

observation, info = env.reset()

initial_state = discretize_state(observation, state_bins)

# Initialize the Q-table with zeros (numbins and numactions)
#q_table = np.zeros((*num_bins, env.action_space.n))
q_table = {}

if not os.path.exists(name+"_q_table.json"):
    explore(num_explorations, alpha, gamma, epsilon, epsilon_decay, epsilon_min, q_table, state_bins, env, initial_state, name)

exploit(num_exploitations, env, state_bins, name)

env.close()
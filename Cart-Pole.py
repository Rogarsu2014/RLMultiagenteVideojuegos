import gymnasium as gym
from Qalgo import *
import numpy as np

env = gym.make('CartPole-v1', render_mode="human")
observation, info = env.reset(seed=42)
print(env.observation_space)

num_bins = [10] * env.observation_space.shape.count()




for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
from libs.pyRIL.RL_Agent import dqn_agent
from libs.pyRIL.RL_Problem.base.ValueBased import dqn_problem
from libs.pyRIL.RL_Agent.base.utils import agent_saver, history_utils

import gymnasium as gym
import os
import tensorflow as tf

# Get the list of available GPU devices
gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print("TensorFlow is using GPU.")
else:
    print("TensorFlow is using CPU.")


environment = gym.make("CartPole-v1", render_mode="human")#, render_mode="human"

name = "CartPole-v1"

show_results = False

agent = dqn_agent.Agent(learning_rate=1e-3, batch_size=128, epsilon=0.4, epsilon_decay=0.999, epsilon_min=0.15)

if show_results:
    if os.path.exists(name + '_agent.json'):
        agent = agent_saver.load(name + '_agent.json', agent)
    else:
        show_results = False

problem = dqn_problem.DQNProblem(environment, agent)

if not show_results:
    problem.solve(100, verbose=1)
    hist = problem.get_histogram_metrics()
    history_utils.plot_reward_hist(hist, n_moving_average=10)

problem.test(n_iter=10, verbose=1)

if not show_results:
    agent_saver.save(agent, name + '_agent.json')
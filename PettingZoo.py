from pettingzoo.butterfly import pistonball_v6

from libs.pyRIL.RL_Agent import dqn_agent
#from libs.pyRIL.RL_Problem.base.ValueBased import dqn_problem
from libs.pyRIL.RL_Agent.base.utils import agent_saver, history_utils

import dqn_problem_multiagent

import os
import tensorflow as tf

# Get the list of available GPU devices
gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print("TensorFlow is using GPU.")
else:
    print("TensorFlow is using CPU.")


environment = pistonball_v6.parallel_env(render_mode="human", continuous=False)#, render_mode="human"
environment.reset()
name = "Pistonball_v6"

show_results = False

agents = []
for agent in environment.agents:
    agents.append(dqn_agent.Agent(learning_rate=1e-3, batch_size=128, epsilon=0.4, epsilon_decay=0.999, epsilon_min=0.15, img_input=True))

if show_results:
    if os.path.exists(name + '_agent.json'):
        agent = agent_saver.load(name + '_agent.json', agent)
    else:
        show_results = False

problem = dqn_problem_multiagent.DQNProblemMultiAgent(environment, agents)

if not show_results:
    problem.solve(10, verbose=1)
    hist = problem.get_histogram_metrics()
    history_utils.plot_reward_hist(hist, n_moving_average=10)

#problem.test(n_iter=10, verbose=1)

#if not show_results:
    #agent_saver.save(agent, name + '_agent.json')
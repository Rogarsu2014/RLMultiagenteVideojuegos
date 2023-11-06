from pettingzoo.sisl import pursuit_v4

from libs.pyRIL.RL_Agent import dqn_agent
#from libs.pyRIL.RL_Problem.base.ValueBased import dqn_problem
from libs.pyRIL.RL_Agent.base.utils import agent_saver, history_utils

from RL_Agent.base.utils.networks import networks

import dqn_problem_multiagent

import os
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

import dqn_agent_coop_brain

def plot_reward_hist(hist, n_moving_average, agent, n_catch, shared_reward):
    x = hist[:, 0]
    y = hist[:, 1]

    y_mean = moving_average(y, n_moving_average)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    fig.suptitle('Reward history ' + agent, fontsize=16)

    ax.plot(x, y, 'r', label="reward")
    ax.plot(x, y_mean, 'b', label="average " + str(n_moving_average) + " rewards")

    ax.legend(loc="upper left")

    ax.set_title("Usando n_catch = "+n_catch+" y shared_reward = "+shared_reward)

    ax.set_ylabel('Reward')
    ax.set_xlabel('Episodes')

    plt.show()


def moving_average(values, window):
    values = np.array(values)
    if window % 2 != 0:
        expand_dims_f = int(window / 2)
        expand_dims_l = int(window / 2)
    else:
        expand_dims_f = int(window / 2) - 1
        expand_dims_l = int(window / 2)

    for i in range(expand_dims_f):
        values = np.insert(values, 0, values[0])
    for i in range(expand_dims_l):
        values = np.append(values, values[-1])
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma

def greedy_multi_action(act_pred, n_actions, i, epsilon=0., n_env=1, exploration_noise=1.0):
    actions = []
    if np.random.rand() <= epsilon:
        # TODO: al utilizar algoritmos normales puede fallar
        for j in range(i):
            action = np.random.rand(n_actions)
            actions.append(np.argmax(action, axis=-1))
    else:
        for j in range(i):
            actions.append(np.argmax(act_pred[0][j*n_actions:(j+1)*n_actions], axis=-1))
    return actions

import cProfile
import re
import pstats

# Get the list of available GPU devices
gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print("TensorFlow is using GPU.")
else:
    print("TensorFlow is using CPU.")


environment = pursuit_v4.parallel_env(n_evaders=30, n_pursuers=8, n_catch=2, surround=False, shared_reward=True)#, render_mode="human"
environment.reset()
env_name = "Pursuit_v4"
num_agents_dqn = 1

name = env_name + "_" + str(num_agents_dqn) + "_"

show_results = False
'''
net_architecture = networks.dqn_net(dense_layers=2,
                                    n_neurons=[100, 100],
                                    dense_activation=['relu', 'relu'],)
'''
'''
agents = []
for i in range(num_agents_dqn):#environment.agents:
    agents.append(dqn_agent.Agent(learning_rate=1e-3, batch_size=128, epsilon=0.4, epsilon_decay=0.999, epsilon_min=0.15, img_input=False, state_size=147, train_steps=10))
'''
agents = []
agents.append(dqn_agent_coop_brain.BrainAgent(action_space_size=environment.action_space(environment.agents[0]).n, num_agents=8,learning_rate=1e-3, batch_size=128, epsilon=0.4, epsilon_decay=0.999, epsilon_min=0.15, img_input=False, state_size=147*8, train_steps=10, train_action_selection_options=greedy_multi_action))

if show_results:
    for i, agent in enumerate(agents):
        if os.path.exists(name + "agent_" + str(i) + '_agent.json'):
            agents[i] = agent_saver.load(name + "agent_" + str(i) + '_agent.json', agent)
        else:
            show_results = False


problem = dqn_problem_multiagent.DQNProblemMultiAgent(environment, agents)

if not show_results:
    problem.solve(1, verbose=1, coop=True)
    for i in range(num_agents_dqn+1):
        hist = problem.get_histogram_metrics(i)

        num = "of agent " + str(i)
        if i == num_agents_dqn:
            num = "of all the agents"

        #Los detalles del plot hay que rellenarlos a mano
        plot_reward_hist(hist, n_moving_average=10, agent=num, n_catch="1", shared_reward="False")

problem.test(n_iter=1, verbose=1)
'''
#cProfile.run('re.compile(problem.test(n_iter=1, verbose=1))', sort='tottime')

pr = cProfile.Profile()                # create a cProfiler object
pr.enable()                            # turn profiling on
pr.runcall(problem.test)  # profile my function
pr.disable()
p = pstats.Stats(pr)           # create pstats obj based on profiler above.
p.print_callers('numpy.array')  # find all the callers of isinstance.
'''
'''
if not show_results:
    for i, agent in enumerate(agents):
        agent_saver.save(agent, name + "agent_" + str(i) + '_agent.json')

'''

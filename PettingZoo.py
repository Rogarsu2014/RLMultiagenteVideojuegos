from pettingzoo.butterfly import pistonball_v6

from libs.pyRIL.RL_Agent import dqn_agent
#from libs.pyRIL.RL_Problem.base.ValueBased import dqn_problem
from libs.pyRIL.RL_Agent.base.utils import agent_saver, history_utils

from RL_Agent.base.utils.networks import networks

import dqn_problem_multiagent

import os
import tensorflow as tf

import cProfile
import re
import pstats

# Get the list of available GPU devices
gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print("TensorFlow is using GPU.")
else:
    print("TensorFlow is using CPU.")


environment = pistonball_v6.parallel_env(continuous=False, render_mode="human")#, render_mode="human"
environment.reset()
env_name = "Pistonball_v6"
num_agents_dqn = 4

name = env_name + "_" + str(num_agents_dqn) + "_"

show_results = True
'''
net_architecture = networks.dqn_net(dense_layers=2,
                                    n_neurons=[100, 100],
                                    dense_activation=['relu', 'relu'],)
'''
agents = []
for i in range(num_agents_dqn):#environment.agents:
    agents.append(dqn_agent.Agent(learning_rate=1e-3, batch_size=128, epsilon=0.4, epsilon_decay=0.999, epsilon_min=0.15, img_input=False, state_size=164520, train_steps=10))

if show_results:
    for i, agent in enumerate(agents):
        if os.path.exists(name + "agent_" + str(i) + '_agent.json'):
            agents[i] = agent_saver.load(name + "agent_" + str(i) + '_agent.json', agent)
        else:
            show_results = False


problem = dqn_problem_multiagent.DQNProblemMultiAgent(environment, agents)

if not show_results:
    problem.solve(300, verbose=1)
    hist = problem.get_histogram_metrics()
    history_utils.plot_reward_hist(hist, n_moving_average=10)

problem.test(n_iter=3, verbose=1)
'''
#cProfile.run('re.compile(problem.test(n_iter=1, verbose=1))', sort='tottime')

pr = cProfile.Profile()                # create a cProfiler object
pr.enable()                            # turn profiling on
pr.runcall(problem.test)  # profile my function
pr.disable()
p = pstats.Stats(pr)           # create pstats obj based on profiler above.
p.print_callers('numpy.array')  # find all the callers of isinstance.
'''

if not show_results:
    for i, agent in enumerate(agents):
        agent_saver.save(agent, name + "agent_" + str(i) + '_agent.json')
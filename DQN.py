from libs.pyRIL.RL_Agent import dqn_agent
from libs.pyRIL.RL_Problem.base.ValueBased import dqn_problem
from libs.pyRIL.RL_Agent.base.utils import agent_saver, history_utils

import gymnasium as gym

environment = gym.make("CartPole-v1", render_mode="human")

agent = dqn_agent.Agent(learning_rate=1e-3, batch_size=128, epsilon=0.4, epsilon_decay=0.999, epsilon_min=0.15)

problem = dqn_problem.DQNProblem(environment, agent)

problem.solve(30, render=True, verbose=1)

problem.test(n_iter=10, verbose=1)

hist = problem.get_histogram_metrics()
history_utils.plot_reward_hist(hist, n_moving_average=10)

agent_saver.save(agent, 'agent_dqn_lunar.json')
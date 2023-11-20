import copy
import numpy as np
import cv2

from collections import deque
from abc import ABCMeta, abstractmethod
from RL_Agent.base.utils.history_utils import write_history

class RLProblemMultiAgentSuper(object, metaclass=ABCMeta):
    """ Reinforcement Learning Problem.

    This class represent the RL problem to solve formed by an agent and an environment. The RL problem controls and
    defines how, when and where the agent and the environment send information to each other or in other words, the
    events flow between agent and environment.
    Specific problem classes for each agent should extend this class in order to adjust and introduce specific features
    in the events flow.
    """
    def __init__(self, environment, agents):
        """
        Attributes:
        :param environment:    (EnvInterface) Environment selected for this RL problem.
        :param agent:          (AgentInterface) Selected agent for solving the environment.
        """
        self.env = environment
        self.agents = agents
        self.num_agents = len(agents)
        self.n_stack = []
        self.img_input = []
        self.state_size = []
        self.n_actions = []
        for i, agent in enumerate(self.agents):
            self.n_stack.append(agent.n_stack)
            self.img_input.append(agent.img_input)

            # Set state_size depending on the input type
            self.state_size.append(agent.env_state_size)

            if self.state_size[i] is None:
                if self.img_input[i]:
                    self.state_size[i] = self.env.observation_space(self.env.agents[i]).shape#(50, 20, 1)
                else:
                    self.state_size[i] = self.env.observation_space(self.env.agents[i]).shape[0]
                agent.env_state_size = self.state_size[i]

            # Set n_actions depending on the environment format
            try:
                self.n_actions.append(self.env.action_space(self.env.agents[i]).n)
            except AttributeError:
                self.n_actions.append(self.env.action_space(self.env.agents[i]).shape[0])

        # Setting default preprocess and clip_norm_reward functions
        self.preprocess = self._preprocess  # Preprocessing function for observations
        self.clip_norm_reward = self._clip_norm_reward  # Clipping reward

        # Total number of steps processed
        self.global_steps = 0
        self.total_episodes = 0

        self.max_rew_mean = -2**1000  # Store the maximum value for reward mean

        self.histogram_metrics = []

    @abstractmethod
    def _build_agent(self):
        """
        This method should call the agent build_agent method. This method needs to be called when a new Rl problem is
        defined or where an already trained agent will be used in a new RL problem.
        """
        pass

    def compile(self):
        for agent in self.agents:
            agent.compile()


    def solve(self, episodes, render=True, render_after=None, max_step_epi=None, skip_states=1, verbose=1,
              discriminator=None, save_live_histogram=False, smooth_rewards=10, comp = False, coop = False, comp_harcore = False):
        """ Method for training the agent to solve the environment problem. The reinforcement learning loop is
        implemented here.

        :param episodes: (int) >= 1. Number of episodes to train.
        :param render: (bool) If True, the environment will show the user interface during the training process.
        :param render_after: (int) >=1 or None. Star rendering the environment after this number of episodes.
        :param max_step_epi: (int) >=1 or None. Maximum number of epochs per episode. Mainly for problems where the
            environment doesn't have a maximum number of epochs specified.
        :param skip_states: (int) >= 1. Frame skipping technique applied in Playing Atari With Deep Reinforcement paper.
            If 1, this technique won't be applied.
        :param verbose: (int) in range [0, 3]. If 0 no training information will be displayed, if 1 lots of
           information will be displayed, if 2 fewer information will be displayed and 3 a minimum of information will
           be displayed.
        :param save_live_histogram: (bool or str) Path for recording live evaluation params. If is set to False, no
            information will be recorded.
        :return:
        """
        if comp or comp_harcore:
            self.histogram_metrics = [[] for i in range(self.num_agents + 1)]

        self.compile()
        # Inicializar iteraciones globales
        if discriminator is None:
            self.global_steps = 0
            self.total_episodes = 0

        # List of 100 last rewards
        rew_mean_list = deque(maxlen=smooth_rewards)

        # Stacking inputs
        obs_queue = []
        obs_next_queue = []
        for i, agent in enumerate(self.agents):
            if self.n_stack[i] is not None and self.n_stack[i] > 1:#Esto no entra (hay que retocarlo si entra)
                obs_queue.append(deque(maxlen=self.n_stack[i]))
                obs_next_queue.append(deque(maxlen=self.n_stack[i]))
            else:
                obs_queue.append(None)
                obs_next_queue.append(None)

        # For each episode do
        for e in range(episodes):

            # Init episode parameters
            if comp_harcore:
                observations, infos, observations_evaders = self.env.reset()
            else:
                observations, infos = self.env.reset()


            if comp or comp_harcore:
                episodic_reward = [0] * (self.num_agents+1)
            else:
                episodic_reward = 0
            epochs = 0
            done = False

            # Reading initial state
            observations = self.preprocess(observations, False)
            if comp_harcore:
                observations_evaders = self.preprocess(observations_evaders, False)
            # obs = np.zeros((300, 300))

            # Stacking inputs
            for j, agent in enumerate(self.agents):
                if self.n_stack[j] is not None and self.n_stack[j] > 1:#Esto no entra (hay que retocarlo si entra)
                    for i in range(self.n_stack[j]):
                        obs_queue[j].append(np.zeros(observations[j].shape))
                        obs_next_queue[j].append(np.zeros(observations[j].shape))
                    obs_queue[j].append(observations[j])
                    obs_next_queue[j].append(observations[j])

            actions = {}
            if comp_harcore:
                actions_evaders = {}

            # While the episode doesn't reach a final state
            while self.env.agents:
                if render or ((render_after is not None) and e > render_after):
                    self.env.render()
                # this is where you would insert your policy
                if coop:
                    all_actions, observations = self.act_train_all(observations)
                    for i, agent in enumerate(self.env.agents):
                        actions[agent] = all_actions[i]
                elif comp_harcore:
                    for i, agent in enumerate(self.env.agents):
                        actions[agent] = self.act_train(observations[agent], obs_queue[0], 0, True)
                    for i, evader in enumerate(self.env.evader):
                        actions_evaders[evader] = self.act_train(observations[evader], obs_queue[0], 1, True)
                else:
                    for i, agent in enumerate(self.env.agents):
                        actions[agent] = self.act_train(observations[agent], obs_queue[i%self.num_agents], i, False)

                if comp_harcore:
                    next_observations, rewards, terminations, truncations, infos, next_observations_evaders, rewards_evaders = self.env.step(actions)
                else:
                    next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
                # Agent act in the environment
                #next_obs, reward, terminated, truncated, _ = self.env.step(action)
                #done = any(terminations.values()) or any(truncations.values())
                if discriminator is not None:#Esto no entra (hay que retocarlo si entra)
                    if discriminator.stack:
                        reward = discriminator.get_reward(obs_queue, actions)[0]
                    else:
                        reward = discriminator.get_reward(observations, actions)[0]
                # next_obs = np.zeros((300, 300))
                # next_obs = self.preprocess(next_obs)  # Is made in store_experience now
                next_observations = self.preprocess(next_observations, False)
                # Store the experience in memory

                if coop:
                    if len(self.env.agents) != 0:
                        next_observations, obs_next_queue, rewards, done, epochs = self.store_experience(actions.copy(), done, next_observations, observations, obs_next_queue, obs_queue, rewards, skip_states, epochs, 0, coop)
                elif comp_harcore:
                    for i, agent in enumerate(self.env.agents):
                        next_observations[agent], obs_next_queue[0], rewards[agent], done, epochs = self.store_experience(actions[agent], done, next_observations[agent], observations[agent], obs_next_queue[0],
                                                                            obs_queue[0], rewards[agent], skip_states, epochs, 0, False, True)
                    for i, evader in enumerate(self.env.evader):
                        next_observations_evaders[evader], obs_next_queue[1], rewards_evaders[evader], done, epochs = self.store_experience(actions_evaders[evader], done, next_observations_evaders[evader], observations_evaders[evader], obs_next_queue[0],
                                                                            obs_queue[0], rewards_evaders[evader], skip_states, epochs, 1, False, True)
                else:
                    for i, agent in enumerate(self.env.agents):
                        next_observations[agent], obs_next_queue[i%self.num_agents], rewards[agent], done, epochs = self.store_experience(actions[agent], done, next_observations[agent], observations[agent], obs_next_queue[i%self.num_agents],
                                                                            obs_queue[i%self.num_agents], rewards[agent], skip_states, epochs, i, False)
                if epochs % 25 == 0:
                    for agent in self.agents:
                        # Replay some memories and training the agent
                        agent.replay()

                # copy next_obs to obs
                observations, obs_queue = self.copy_next_obs(next_observations, observations, obs_next_queue, obs_queue,0)#i = 0 por conveniencia

                if comp_harcore:
                    observations_evaders, obs_queue = self.copy_next_obs(next_observations_evaders, observations_evaders, obs_next_queue,
                                                                 obs_queue, 0)  # i = 0 por conveniencia

                # If max steps value is reached the episode is finished
                done = self._max_steps(done, epochs, max_step_epi)

                values = rewards.values()
                if comp:
                    for i, agent in enumerate(self.env.agents):
                        episodic_reward[i] += rewards[agent]
                    episodic_reward[self.num_agents] += sum(values) / len(values)
                    # Add reward to the list
                    rew_mean_list.append(episodic_reward[self.num_agents])
                elif comp_harcore:
                    values_evaders = rewards_evaders.values()
                    for agent in range(self.env.agents):
                        episodic_reward[0] += rewards[agent]
                    for evader in range(self.env.evaders):
                        episodic_reward[1] += rewards_evaders[evader]
                    episodic_reward[self.num_agents] += (sum(values) / len(values)) + (sum(values_evaders) / len(values_evaders))
                    # Add reward to the list
                    rew_mean_list.append(episodic_reward[self.num_agents])
                else:
                    episodic_reward += sum(values) / len(values)
                    # Add reward to the list
                    rew_mean_list.append(episodic_reward)
                epochs += 1
                self.global_steps += 1

            if comp or comp_harcore:
                for i in range(self.num_agents+1):
                    self.histogram_metrics[i].append([self.total_episodes, episodic_reward[i], epochs, self.agents[i%self.num_agents].epsilon, self.global_steps])
            else:
                self.histogram_metrics.append([self.total_episodes, episodic_reward, epochs, self.agents[0].epsilon, self.global_steps])

            if save_live_histogram:
                if isinstance(save_live_histogram, str):
                    write_history(rl_hist=self.histogram_metrics, monitor_path=save_live_histogram)
                else:
                    raise Exception('Type of parameter save_live_histories must be string but ' +
                                    str(type(save_live_histogram)) + ' has been received')

            # Copy main model to target model
            for i, agent in enumerate(self.agents):
                self.agents[i].copy_model_to_target()

            # Print log on scream
            self._feedback_print(self.total_episodes, episodic_reward, epochs, verbose, rew_mean_list, comp)
            self.total_episodes += 1

            for i, agent in enumerate(self.agents):
                if comp or comp_harcore:
                    self.agents[i].save_tensorboar_rl_histogram(self.histogram_metrics[i])
                else:
                    self.agents[i].save_tensorboar_rl_histogram(self.histogram_metrics)
        return

    def copy_next_obs(self, next_obs, obs, obs_next_queue, obs_queue, i):
        """
        Make a copy of the current observation ensuring the is no conflicts of two variables pointing common values.
        """
        if self.n_stack[i%self.num_agents] is not None and self.n_stack[i%self.num_agents] > 1:
            obs_queue = copy.copy(obs_next_queue)
        else:
            obs = next_obs
        return obs, obs_queue

    def act_train(self, obs, obs_queue, i, comp_hardcore):
        """
        Make the agent select an action in training mode given an observation. Use an input depending if the
        observations are stacked in
        time or not.
        :param i:
        :param obs: (numpy nd array) observation (state).
        :param obs_queue: (numpy nd array) List of observation (states) in sequential time steps.
        :return: (int or [floats]) int if actions are discrete or numpy array of float of action shape if actions are
            continuous)
        """
        if self.n_stack[i%self.num_agents] is not None and self.n_stack[i%self.num_agents] > 1:
            action = self.agents[i%self.num_agents].act_train(np.array(obs_queue))
        else:
            if comp_hardcore:
                action = self.agents[i].act_train(obs)
            else:
                action = self.agents[i%self.num_agents].act_train(obs)
        return action

    def act(self, obs, obs_queue, i, comp_hardcore):
        """
        Make the agent select an action in exploitation mode given an observation. Use an input depending if the
        observations are stacked.
        time or not.
        :param obs: (numpy nd array) observation (states).
        :param obs_queue: (numpy nd array) List of observation (states) in sequential time steps.
        :return: (int or [floats]) int if actions are discrete or numpy array of float of action shape if actions are
            continuous)
        """
        if self.n_stack[i%self.num_agents] is not None and self.n_stack[i%self.num_agents] > 1:
            action = self.agents[i%self.num_agents].act(np.array(obs_queue))
        else:
            if comp_hardcore:
                action = self.agents[i].act_train(obs)
            else:
                action = self.agents[i%self.num_agents].act(obs)
        return action

    def store_experience(self, action, done, next_obs, obs, obs_next_queue, obs_queue, reward, skip_states, epochs, i, coop, comp_hardcore):
        """
        Method for store a experience in the agent memory. A standard experience consist of a tuple (observation,
        action, reward, next observation, done flag).
        :param action: (int, [int] or [float]) Action selected by the agent. Type depend on the action type (discrete or
            continuous) and the agent needs.
        :param done: (bool) Episode is finished flag. True denotes that next_obs or obs_next_queue represent a final
            state.
        :param next_obs: (numpy nd array) Next observation to obs (state).
        :param obs: (numpy nd array) Observation (state).
        :param obs_next_queue: (numpy nd array) List of Next observations to obs_queue (states) in sequential time steps.
        :param obs_queue: (numpy nd array) List of observation (states) in sequential time steps.
        :param reward: (float) Regard value obtained by the agent in the current experience.
        :param skip_states: (int) >= 0. Select the states to skip with the frame skipping technique (explained in
            Playing Atari With Deep Reinforcement paper).
        :param epochs: (int) Episode epochs counter.
        """
        # Execute the frame skipping technique explained in Playing Atari With Deep Reinforcement paper.
        done, next_obs, reward, epochs = self.frame_skipping(action, done, next_obs, reward, skip_states, epochs)

        # Store the experience in memory depending on stacked inputs and observations type
        if self.n_stack[i%self.num_agents] is not None and self.n_stack[i%self.num_agents] > 1:
            obs_next_queue.append(next_obs)

            if self.img_input:
                obs_satck = np.dstack(obs_queue)
                obs_next_stack = np.dstack(obs_next_queue)
            else:
                obs_satck = np.array(obs_queue)
                obs_next_stack = np.array(obs_next_queue)

            self.agents[i%self.num_agents].remember(obs_satck, action, self.clip_norm_reward(reward), obs_next_stack, done)
        else:
            if coop:
                self.agents[i%self.num_agents].remember(obs, action, self.clip_norm_reward(reward), self.preprocess(next_obs, coop), done)
            elif comp_hardcore:
                self.agents[i].remember(obs, action, self.clip_norm_reward(reward), next_obs, done)
            else:
                self.agents[i % self.num_agents].remember(obs, action, self.clip_norm_reward(reward), next_obs, done)
        return next_obs, obs_next_queue, reward, done, epochs

    def frame_skipping(self, action, done, next_obs, reward, skip_states, epochs):
        """
        This method execute the frame skipping technique explained in Playing Atari With Deep Reinforcement paper. It
        consist on repeating the last selected action n times whit the objective of explore a bigger space.
        :param action: (int, [int] or [float]) Action selected by the agent. Type depend on the action type (discrete or
            continuous) and the agent needs.
        :param done: (bool) Episode is finished flag. True denotes that next_obs or obs_next_queue represent a final
            state.
        :param next_obs: (numpy nd array) Next observation to obs (state).
        :param reward: (float) Regard value obtained by the agent in the current experience.
        :param skip_states: (int) >= 0. Select the number of states to skip.
        :param epochs: (int) Spisode epochs counter.
        """
        if skip_states > 1 and not done:#Esto no entra (hay que retocarlo si entra)
            for i in range(skip_states - 2):
                next_obs_aux1, reward_aux, terminated_aux, truncated_aux, _ = self.env.step(action)
                done_aux = terminated_aux or truncated_aux
                epochs += 1
                reward += reward_aux
                if done_aux:
                    next_obs_aux2 = next_obs_aux1
                    done = done_aux
                    break

            if not done:
                next_obs_aux2, reward_aux, terminated_aux, truncated_aux, _ = self.env.step(action)
                done_aux = terminated_aux or truncated_aux
                epochs += 1
                reward += reward_aux
                done = done_aux

            if self.img_input:
                next_obs_aux2 = self.preprocess(next_obs_aux2, False)
                if skip_states > 2:
                    next_obs_aux1 = self.preprocess(next_obs_aux1, False)
                    # TODO: esto no se debería hacer con todas las imágenes intermedias? consultar en paper atari dqn
                    next_obs = np.maximum(next_obs_aux2, next_obs_aux1)

                else:
                    next_obs = self.preprocess(next_obs, False)
                    next_obs = np.maximum(next_obs_aux2, next_obs)
            else:
                next_obs = self.preprocess(next_obs_aux2, False)
        else:
            next_obs = next_obs
        return done, next_obs, reward, epochs

    def test(self, n_iter=1, render=True, verbose=1, callback=None, smooth_rewards=10, discriminator=None, coop = False):
        """ Test a trained agent using only exploitation mode on the environment.

        :param n_iter: (int) number of test iterations.
        :param render: (bool) If True, the environment will show the user interface during the training process.
        :param verbose: (int) in range [0, 3]. If 0 no training information will be displayed, if 1 lots of
           information will be displayed, if 2 fewer information will be displayed and 3 a minimum of information will
           be displayed.
        :param callback: A extern function that receives a tuple (prev_obs, obs, action, reward, done, info)
        """
        epi_rew_mean = 0
        rew_mean_list = deque(maxlen=smooth_rewards)

        # Stacking inputs
        obs_queue = []
        obs_next_queue = []
        for i, agent in enumerate(self.agents):
            if self.n_stack[i] is not None and self.n_stack[i] > 1:
                obs_queue.append(deque(maxlen=self.n_stack))
            else:
                obs_queue.append(None)

        # For each episode do
        for e in range(n_iter):
            done = False
            episodic_reward = 0
            epochs = 0
            observations, info = self.env.reset()
            observations = self.preprocess(observations, False)

            # stacking inputs
            for i, agent in enumerate(self.env.agents):
                if self.n_stack[i%self.num_agents] is not None and self.n_stack[i%self.num_agents] > 1:#Esto no entra (hay que retocarlo si entra)
                    for j in range(self.n_stack[i%self.num_agents]):
                        obs_queue[i%self.num_agents].append(np.zeros(observations[agent].shape))
                    obs_queue.append(observations)

            actions = {}
            while self.env.agents:
                if render:
                    self.env.render()

                # Select action
                # TODO: poner bien
                # action = self.act(obs, obs_queue)
                if coop:
                    all_actions, observations = self.act_train_all(observations)
                    for i, agent in enumerate(self.env.agents):
                        actions[agent] = all_actions[i]
                else:
                    for i, agent in enumerate(self.env.agents):
                        actions[agent] = self.act(observations[agent], obs_queue[i % self.num_agents], i, False)
                #action = self.act(obs, obs_queue)
                prev_observations = observations

                observations, rewards, terminations, truncations, infos = self.env.step(actions)
                #obs, reward, terminated, truncated, info = self.env.step(action)

                observations = self.preprocess(observations, False)

                if discriminator is not None:#Esto no entra (hay que retocarlo si entra)
                    if discriminator.stack:
                        reward = discriminator.get_reward(obs_queue, actions, multithread=False)[0]
                    else:
                        reward = discriminator.get_reward(observations, actions, multithread=False)[0]

                if callback is not None:
                    callback(prev_observations, observations, actions, reward, done, info)#Esto no entra (hay que retocarlo si entra)

                values = rewards.values()
                episodic_reward += sum(values) / len(values)
                epochs += 1

                for i, agent in enumerate(self.env.agents):#Esto no entra (hay que retocarlo si entra)
                    if self.n_stack[i%self.num_agents] is not None and self.n_stack[i%self.num_agents] > 1:
                        obs_queue.append(observations[agent])
            rew_mean_list.append(episodic_reward)

            self._feedback_print(e, episodic_reward, epochs, verbose, rew_mean_list, comp=False, test=True)

        # print('Mean Reward ', epi_rew_mean / n_iter)
        self.env.close()

    def _preprocess(self, obs, coop):
        """
        Preprocessing function by default does nothing to the observation.
        :param obs: (numpy nd array) Observation (state).
        """
        if coop:
            obs_aux = []
            for agent in self.env.agents:
                obs_aux.append(obs[agent])
            return obs_aux
        else:
            for agent in self.env.agents:
                #cv2.imshow("Ventana", obs[agent])
                #cv2.waitKey(0)
                obs[agent] = np.ravel(obs[agent])
            return obs

    def _clip_norm_reward(self, rew):
        """
        Clip and/or normalize the reward. By default does nothing to the reward value.
        :param rew: (float) Regard value obtained by the agent in a experience.
        """
        return rew

    def _max_steps(self, done, epochs, max_steps):
        """
        Return True if number of epochs pass a selected number of steps. This allow to set a maximum number of
        iterations for each RL epoch.
        :param done: (bool) Episode is finished flag. True if the episode has finished.
        :param epochs: (int) Episode epochs counter.
        :param max_steps: (int) Maximum number of episode epochs. When it is reached param done is set to True.
        """
        if max_steps is not None:
            return epochs >= max_steps or done
        return done

    def _feedback_print(self, episode, episodic_reward, epochs, verbose, epi_rew_list, comp, test=False):
        """
        Print on terminal information about the training process.
        :param episode: (int) Current episode.
        :param episodic_reward: (float) Cumulative reward of the last episode.
        :param epochs: (int) Episode epochs counter.
        :param verbose: (int) in range [0, 3]. If 0 no training information will be displayed, if 1 lots of
           information will be displayed, if 2 fewer information will be displayed and 3 a minimum of information will
           be displayed.
        :param epi_rew_list: ([float]) List of reward of the last episode.
        :param test: (bool) Flag for select test mode. True = test mode, False = train mode.
        """
        rew_mean = np.sum(epi_rew_list) / len(epi_rew_list)

        if test:
            episode_str = 'Test episode: '
        else:
            episode_str = 'Episode: '
        if verbose == 1:
            if (episode + 1) % 1 == 0:
                if comp:
                    print(episode_str, episode + 1, 'Epochs: ', epochs, ' Reward: {:.1f}'.format(episodic_reward[self.num_agents]),
                          'Smooth Reward: {:.1f}'.format(rew_mean), ' Epsilon: {:.4f}'.format(self.agents[0].epsilon))
                else:
                    print(episode_str, episode + 1, 'Epochs: ', epochs, ' Reward: {:.1f}'.format(episodic_reward),
                          'Smooth Reward: {:.1f}'.format(rew_mean), ' Epsilon: {:.4f}'.format(self.agents[0].epsilon))

        if verbose == 2:
            print(episode_str, episode + 1, 'Mean Reward: ', rew_mean)
        if verbose == 3:
            print(episode_str, episode + 1)

    # def load_model(self, dir_load="", name_loaded=""):
    #     self.agent._load(dir_load)
    #
    # def save_agent(self, agent_name=None):
    #     if agent_name is None:
    #         agent_name = self.agent.agent_name
    #     with open(agent_name, 'wb') as f:
    #         dill.dump(self.agent, f)

    def get_histogram_metrics(self, i, comp):
        """
        Return the history of metrics consisting on a array with rows:  [episode number, episode reward, episode epochs,
        epsilon value, global steps]
        return: (2D array)
        """
        if comp:
            return np.array(self.histogram_metrics[i])
        else:
            return np.array(self.histogram_metrics)

    def parseAgent(self, i):
        if i == 0:
            return 1
        else:
            return 0


    def act_train_all(self, obs):
        """
        Make the agent select an action in training mode given an observation. Use an input depending if the
        observations are stacked in
        time or not.
        :param obs: (numpy nd array) observation (state).
        :param obs_queue: (numpy nd array) List of observation (states) in sequential time steps.
        :return: (int or [floats]) int if actions are discrete or numpy array of float of action shape if actions are
            continuous)
        """
        obs_aux = self.preprocess(obs, True)
        action_aux = self.agents[0].act_train(obs_aux, len(self.env.agents))
        return action_aux, obs_aux
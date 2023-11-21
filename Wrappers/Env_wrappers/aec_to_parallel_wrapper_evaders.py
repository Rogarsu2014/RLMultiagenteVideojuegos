from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from collections import defaultdict

class aec_to_parallel_wrapper_evaders(aec_to_parallel_wrapper):
    def __init__(self, aec_env):
        super().__init__(aec_env)


    def reset(self, seed=None, options=None):
        self.aec_env.reset(seed=seed, options=options)
        self.agents = self.aec_env.agents[:]
        observations = {
            agent: self.aec_env.observe(agent)
            for agent in self.aec_env.agents
            if not (self.aec_env.terminations[agent] or self.aec_env.truncations[agent])
        }
        observations_evaders = {
            evader: self.aec_env.observe(evader)
            for evader in self.aec_env.env.env.evaders
            #if not (self.aec_env.terminations[agent] or self.aec_env.truncations[agent])
        }

        infos = dict(**self.aec_env.infos)
        return observations, infos, observations_evaders

    def step(self, actions):
        rewards = defaultdict(int)
        terminations = {}
        truncations = {}
        infos = {}
        observations = {}
        for agent in self.aec_env.agents:
            if agent != self.aec_env.agent_selection:
                if self.aec_env.terminations[agent] or self.aec_env.truncations[agent]:
                    raise AssertionError(
                        f"expected agent {agent} got termination or truncation agent {self.aec_env.agent_selection}. Parallel environment wrapper expects all agent death (setting an agent's self.terminations or self.truncations entry to True) to happen only at the end of a cycle."
                    )
                else:
                    raise AssertionError(
                        f"expected agent {agent} got agent {self.aec_env.agent_selection}, Parallel environment wrapper expects agents to step in a cycle."
                    )
            obs, rew, termination, truncation, info = self.aec_env.last()
            self.aec_env.step(actions[agent])
            for agent in self.aec_env.agents:
                rewards[agent] += self.aec_env.rewards[agent]

        terminations = dict(**self.aec_env.terminations)
        truncations = dict(**self.aec_env.truncations)
        infos = dict(**self.aec_env.infos)
        observations = {
            agent: self.aec_env.observe(agent) for agent in self.aec_env.agents
        }
        while self.aec_env.agents and (
            self.aec_env.terminations[self.aec_env.agent_selection]
            or self.aec_env.truncations[self.aec_env.agent_selection]
        ):
            self.aec_env.step(None)

        self.agents = self.aec_env.agents


        return observations, rewards, terminations, truncations, infos,


    def step_evaders(self, actions_evaders):
        rewards_evaders = defaultdict(int)
        observations_evaders = {}

        for evader in self.aec_env.env.env.evaders:
            if evader != self.aec_env.env.env.evader_selection:
                raise AssertionError(
                    f"expected agent {evader} got agent {self.aec_env.env.env.evader_selection}, Parallel environment wrapper expects agents to step in a cycle. Error evader manual"
                )
            # No sabemos si esto se usa o no #obs, rew, termination, truncation, info = self.aec_env.last()
            self.aec_env.env.env.step_evader(actions_evaders[evader])
            for evader in self.aec_env.env.env.evaders:
                rewards_evaders[evader] += self.aec_env.env.env.rewards_evaders[evader]

        observations_evaders = {
            evader: self.aec_env.observe(evader) for evader in self.aec_env.env.env.evaders
        }
        #Ya veremos si esto es necesario #self.evaders = self.aec_env.evaders

        return observations_evaders, rewards_evaders
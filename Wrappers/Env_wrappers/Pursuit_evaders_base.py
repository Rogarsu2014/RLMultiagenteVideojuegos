from pettingzoo.sisl.pursuit.pursuit_base import Pursuit
import numpy as np
from typing import Optional
from pettingzoo.sisl.pursuit.utils.controllers import (
    PursuitPolicy,
    RandomPolicy,
    SingleActionPolicy,
)

class PursuitEvaders(Pursuit):
    def __init__(self, x_size: int = 16, y_size: int = 16, max_cycles: int = 500, shared_reward: bool = True,
                 n_evaders: int = 30, n_pursuers: int = 8, obs_range: int = 7, n_catch: int = 2,
                 freeze_evaders: bool = False, evader_controller: Optional[PursuitPolicy] = None,
                 pursuer_controller: Optional[PursuitPolicy] = None, tag_reward: float = 0.01,
                 catch_reward: float = 5.0, urgency_reward: float = -0.1, surround: bool = True,
                 render_mode=None, constraint_window: float = 1.0,):

        self.num_evaders = n_evaders
        self.latest_reward_evader_state = [0 for _ in range(self.num_evaders)]

        super().__init__(x_size=x_size, y_size=y_size, max_cycles=max_cycles, shared_reward=shared_reward,
                         n_evaders=n_evaders, n_pursuers=n_pursuers, obs_range=obs_range, n_catch=n_catch,
                         freeze_evaders=freeze_evaders, evader_controller=evader_controller,
                         pursuer_controller=pursuer_controller, tag_reward=tag_reward, catch_reward=catch_reward,
                         urgency_reward=urgency_reward, surround=surround, render_mode=render_mode,
                         constraint_window=constraint_window,)


    def agents_evaders(self):
        return self.evaders

    def n_agents_evaders(self):
        return self.evader_layer.n_agents()

    def reset(self):
        self.latest_reward_evader_state = [0 for _ in range(self.num_evaders)]
        super().reset()

    def step(self, action, agent_id, is_last):
        agent_layer = self.pursuer_layer

        # actual action application, change the pursuer layer
        agent_layer.move_agent(agent_id, action)

        # Update only the pursuer layer
        self.model_state[1] = self.pursuer_layer.get_state_matrix()

        self.latest_reward_state = self.reward() / self.num_agents

        if is_last:
            # Possibly change the evader layer
            ev_remove, pr_remove, pursuers_who_remove = self.remove_agents()

            self.latest_reward_state += self.catch_reward * pursuers_who_remove
            self.latest_reward_state += self.urgency_reward
            self.frames = self.frames + 1

        # Update the remaining layers
        self.model_state[0] = self.map_matrix
        self.model_state[2] = self.evader_layer.get_state_matrix()

        global_val = self.latest_reward_state.mean()
        local_val = self.latest_reward_state
        self.latest_reward_state = (
                self.local_ratio * local_val + (1 - self.local_ratio) * global_val
        )

        if self.render_mode == "human":
            self.render()

    def step_evader(self, action, evader_id):
        opponent_layer = self.evader_layer

        opponent_layer.move_agent(evader_id, action)

        self.latest_reward_evader_state = self.reward_evader() / self.num_evaders

        self.model_state[2] = self.evader_layer.get_state_matrix()
        # Update the remaining layers
        self.model_state[0] = self.map_matrix
        self.model_state[1] = self.pursuer_layer.get_state_matrix()

        global_val = self.latest_reward_evader_state.mean()
        local_val = self.latest_reward_evader_state
        self.latest_reward_evader_state = (
                self.local_ratio * local_val + (1 - self.local_ratio) * global_val
        )

        if self.render_mode == "human":
            self.render()

    def reward_evader(self):
        rewards = [self.tag_reward for i in range(self.n_evaders)]
        return np.array(rewards)

    def safely_observe_layer(self, layer_number, i):
        if layer_number == 0:
            agent_layer = self.pursuer_layer
        else:
            agent_layer = self.evader_layer
        obs = self.collect_obs_layer(agent_layer, layer_number, i)
        return obs

    def collect_obs_layer(self, agent_layer, layer_number, i):
        num = self.n_agents() if layer_number == 0 else self.n_agents_evaders()
        for j in range(num):
            if i == j:
                return self.collect_obs_by_idx(agent_layer, i)
        assert False, "bad index"

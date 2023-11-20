from pettingzoo.sisl.pursuit.pursuit_base import Pursuit
import numpy as np

class Pursuit_evaders(Pursuit):
    def __init__(self):
        super().__init__()
        self.num_evaders = self.n_evaders
        self.latest_reward_evader_state = [0 for _ in range(self.num_evaders)]

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

    def step_evader(self,  action, evader_id):
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
        obs = self.collect_obs(agent_layer, layer_number, i)
        return obs

    def collect_obs_layer(self, agent_layer, layer_number, i):
        num = self.n_agents() if layer_number == 0 else self.n_agents_evaders()
        for j in range(num):
            if i == j:
                return self.collect_obs_by_idx(agent_layer, i)
        assert False, "bad index"

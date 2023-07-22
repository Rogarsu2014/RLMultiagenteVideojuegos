import numpy as np
import random

def calculateQ(actualQ, alpha, reward, gamma, max_next_state):
    return (1 - alpha) * actualQ + alpha * (reward + gamma * max_next_state)


def random_argmax_list(arr):
    max_val = max(arr)
    max_indices = [i for i, val in enumerate(arr) if val == max_val]
    return random.choice(max_indices)


def get_state_bins(env, num_bins):
    state_bins = []
    for i in range(env.observation_space.shape[0]):
        if env.observation_space.low[i] < -1e6 or env.observation_space.high[i] > 1e6:
            state_low = -5.0  # Set a fixed lower bound for infinite values
            state_high = 5.0  # Set a fixed upper bound for infinite values
        else:
            state_low = env.observation_space.low[i]
            state_high = env.observation_space.high[i]
        #state_low, state_high = env.observation_space.low[i], env.observation_space.high[i]
        state_bins.append(np.linspace(state_low, state_high, num_bins[i] + 1)[1:-1])
    return state_bins

def discretize_state(state, state_bins):
    # Discretize each state variable based on the bin values
    discrete_state = []
    for i in range(len(state)):
        discrete_state.append(np.digitize(state[i], state_bins[i]))
    return tuple(discrete_state)


def explore(num_cycles, alpha, gamma, q_table, state_bins, env, initial_state):

    discrete_state = initial_state

    learning_curve = (num_cycles-num_cycles/3)

    for i in range(num_cycles):

        if random.randint(i, num_cycles) < learning_curve:
            action = env.action_space.sample() #accion aleatoria
        else:
            action = random_argmax_list(q_table[discrete_state])#mejor accion

        q_value = q_table[discrete_state + (action,)]

        observation, reward, terminated, truncated, info = env.step(action)

        next_discrete_state = discretize_state(observation, state_bins)
        next_action = random_argmax_list(q_table[next_discrete_state])
        new_q_value = calculateQ(q_value, alpha, reward, gamma, q_table[next_discrete_state + (next_action,)])

        # Update the Q-table with the new Q-value
        q_table[discrete_state + (action,)] = new_q_value

        #pasar al siguiente
        discrete_state = next_discrete_state

        if terminated or truncated:
            observation, info = env.reset()
            initial_state = discretize_state(observation, state_bins)
            discrete_state = initial_state

    print("End Exploration")
    np.save('CartPole_q_table.npy', q_table)

def exploit(num_cycles, env, state_bins):

    count = 0
    max_count = 0

    q_table = np.load('CartPole_q_table.npy')
    observation, info = env.reset()
    initial_state = discretize_state(observation, state_bins)
    discrete_state = initial_state

    for _ in range(num_cycles):
        action = random_argmax_list(q_table[discrete_state])
        observation, reward, terminated, truncated, info = env.step(action)
        discrete_state = discretize_state(observation, state_bins)

        count += 1

        if terminated or truncated:
            observation, info = env.reset()
            initial_state = discretize_state(observation, state_bins)
            discrete_state = initial_state

            print(count)
            if count > max_count:
                max_count = count
            count = 0

    print("End Exploitation")
    print(max_count)
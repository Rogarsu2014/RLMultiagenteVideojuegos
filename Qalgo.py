import numpy as np
import random
import json


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
            state_low = -10.0  # Set a fixed lower bound for infinite values
            state_high = 10.0  # Set a fixed upper bound for infinite values
        else:
            state_low = env.observation_space.low[i]
            state_high = env.observation_space.high[i]
        # state_low, state_high = env.observation_space.low[i], env.observation_space.high[i]
        state_bins.append(np.linspace(state_low, state_high, num_bins[i] + 1)[1:-1])
    return state_bins


def discretize_state(state, state_bins):
    # Discretize each state variable based on the bin values
    discrete_state = []
    for i in range(len(state)):
        discrete_state.append(np.digitize(state[i], state_bins[i]))
    return tuple(discrete_state)


def get_action_bins(env, num_bins):
    action_bins = []
    for i in range(env.action_space.shape[0]):
        if env.action_space.low[i] < -1e6 or env.action_space.high[i] > 1e6:
            action_low = -10.0  # Set a fixed lower bound for infinite values
            action_high = 10.0  # Set a fixed upper bound for infinite values
        else:
            action_low = env.action_space.low[i]
            action_high = env.action_space.high[i]
        # action_low, action_high = env.action_space.low[i], env.action_space.high[i]
        action_bins.append(np.linspace(action_low, action_high, num_bins[i] + 1)[1:-1])
    return action_bins


def discretize_action(action, action_bins):
    # Discretize each action variable based on the bin values
    discrete_action = []
    for i in range(len(action)):
        discrete_action.append(np.digitize(action[i], action_bins[i]))
    return tuple(discrete_action)


def hash_state(discrete_state):
    aux_array = [str(num) for num in discrete_state]
    result = ''.join(aux_array)
    return result

def hash_discrete_state(discrete_state):
    return str(discrete_state)


def explore(num_cycles, alpha, gamma, epsilon, epsilon_decay, epsilon_min, q_table, state_bins, env, initial_state, name):

    num_actions = env.action_space.n
    aux_epsilon = epsilon

    discrete_state = hash_state(initial_state)

    for i in range(num_cycles):

        if discrete_state not in q_table:
            q_table[discrete_state] = [0] * num_actions

        if random.random() < aux_epsilon:
            action = env.action_space.sample()  # accion aleatoria
        else:
            action = random_argmax_list(q_table[discrete_state])  # mejor accion

        q_value = q_table[discrete_state][action]

        observation, reward, terminated, truncated, info = env.step(action)

        aux_discrete_state = discretize_state(observation, state_bins)
        next_discrete_state = hash_state(aux_discrete_state)
        if next_discrete_state not in q_table:
            q_table[next_discrete_state] = [0] * num_actions
        next_action = random_argmax_list(q_table[next_discrete_state])
        new_q_value = calculateQ(q_value, alpha, reward, gamma, q_table[next_discrete_state][next_action])

        # Update the Q-table with the new Q-value
        q_table[discrete_state][action] = new_q_value

        # pasar al siguiente
        discrete_state = next_discrete_state

        if aux_epsilon >= epsilon_min:
            aux_epsilon = aux_epsilon * epsilon_decay

        if terminated or truncated:
            observation, info = env.reset()
            initial_state = discretize_state(observation, state_bins)
            discrete_state = hash_state(initial_state)

    print("End Exploration")
    # np.save(name +'_q_table.npy', q_table)
    with open(name + "_q_table.json", "w") as f:
        json.dump(q_table, f)


def explore_discrete_observation(num_cycles, alpha, gamma, epsilon, epsilon_decay, epsilon_min, q_table, env, initial_state, name):
    num_actions = env.action_space.n
    discrete_state = hash_discrete_state(initial_state)

    aux_epsilon = epsilon

    for i in range(num_cycles):

        if discrete_state not in q_table:
            q_table[discrete_state] = [0] * num_actions

        if random.random() < aux_epsilon:
            action = env.action_space.sample()  # accion aleatoria
        else:
            action = random_argmax_list(q_table[discrete_state])  # mejor accion

        q_value = q_table[discrete_state][action]

        observation, reward, terminated, truncated, info = env.step(action)

        next_discrete_state = hash_discrete_state(observation)
        if next_discrete_state not in q_table:
            q_table[next_discrete_state] = [0] * num_actions
        next_action = random_argmax_list(q_table[next_discrete_state])
        new_q_value = calculateQ(q_value, alpha, reward, gamma, q_table[next_discrete_state][next_action])

        # Update the Q-table with the new Q-value
        q_table[discrete_state][action] = new_q_value

        # pasar al siguiente
        discrete_state = next_discrete_state

        if aux_epsilon >= epsilon_min:
            aux_epsilon = aux_epsilon * epsilon_decay

        if terminated or truncated:
            observation, info = env.reset()
            initial_state = observation
            discrete_state = hash_discrete_state(initial_state)

    print("End Exploration")
    #np.save(name + '_q_table.npy', q_table)
    with open(name + "_q_table.json", "w") as f:
        json.dump(q_table, f)


def exploit(num_cycles, env, state_bins, name):
    episode_number = 0
    count = 0
    max_count = -99999
    aux_reward = 0
    sum_reward = 0
    max_reward = -99999

    num_actions = env.action_space.n
    # q_table = np.load(name+'_q_table.npy')
    with open(name + "_q_table.json") as f:
        q_table = json.load(f)
    observation, info = env.reset()
    initial_state = discretize_state(observation, state_bins)
    discrete_state = hash_state(initial_state)

    for _ in range(num_cycles):

        if discrete_state not in q_table:
            q_table[discrete_state] = [0] * num_actions

        action = random_argmax_list(q_table[discrete_state])
        observation, reward, terminated, truncated, info = env.step(action)
        discrete_state = hash_state(discretize_state(observation, state_bins))

        count += 1
        aux_reward += reward

        if terminated or truncated:
            observation, info = env.reset()
            initial_state = discretize_state(observation, state_bins)
            discrete_state = hash_state(initial_state)

            episode_number += 1
            print("episode count: " + str(count))
            print("episode reward: " + str(aux_reward))
            sum_reward += aux_reward
            if count > max_count:
                max_count = count
            if aux_reward > max_reward:
                max_reward = aux_reward
            count = 0
            aux_reward = 0

    print("End Exploitation")
    print("Maximum episode count: " + str(max_count))
    print("Maximum episode reward: " + str(max_reward))
    print("average reward: " + str(sum_reward / episode_number))


def exploit_discrete_observation(num_cycles, env, name):
    episode_number = 0
    count = 0
    max_count = -99999
    aux_reward = 0
    sum_reward = 0
    max_reward = -99999

    num_actions = env.action_space.n
    # q_table = np.load(name+'_q_table.npy')
    with open(name + "_q_table.json") as f:
        q_table = json.load(f)
    observation, info = env.reset()
    discrete_state = hash_discrete_state(observation)

    for _ in range(num_cycles):

        if discrete_state not in q_table:
            q_table[discrete_state] = [0] * num_actions

        action = random_argmax_list(q_table[discrete_state])
        observation, reward, terminated, truncated, info = env.step(action)
        discrete_state = hash_discrete_state(observation)

        count += 1
        aux_reward += reward

        if terminated or truncated:
            observation, info = env.reset()
            initial_state = observation
            discrete_state = hash_discrete_state(initial_state)

            episode_number += 1
            print("episode count: " + str(count))
            print("episode reward: " + str(aux_reward))
            sum_reward += aux_reward
            if count > max_count:
                max_count = count
            if aux_reward > max_reward:
                max_reward = aux_reward
            count = 0
            aux_reward = 0

    print("End Exploitation")
    print("Maximum episode count: " + str(max_count))
    print("Maximum episode reward: " + str(max_reward))
    print("average reward: " + str(sum_reward / episode_number))

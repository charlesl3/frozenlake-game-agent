import numpy as np
import gymnasium as gym
import random

# -----------------------------
# 1. Create the environment
# -----------------------------
# Default 4x4 deterministic FrozenLake
#
# desc = [
#     "SFFFFFFFFF",
#     "FFFFFFFFFF",
#     "FHFHFFHFFF",
#     "FFFFFFFFFF",
#     "FFFHFFHFFF",
#     "FFFFFFFFFF",
#     "FFHFFHFFFF",
#     "FFHFFHFFHF",
#     "FFHFFFFFFF",
#     "FFFHFFFFFG",
# ]
#
# envs = gym.make("FrozenLake-v1", desc=desc, is_slippery=False) #, render_mode="human"
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
# Or, if you want a custom map:
# desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
# envs = gym.make("FrozenLake-v1", desc=desc, is_slippery=True)

print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

state_space = env.observation_space.n
action_space = env.action_space.n

print("There are", state_space, "possible states")
print("There are", action_space, "possible actions")

# -----------------------------
# 2. Q-table initialization
# -----------------------------
def initialize_q_table(state_space, action_space):
    return np.zeros((state_space, action_space))

Qtable_frozenlake = initialize_q_table(state_space, action_space)

# -----------------------------
# 3. Policies
# -----------------------------
def epsilon_greedy_policy(Qtable, state, epsilon):
    if random.uniform(0, 1) > epsilon:
        return np.argmax(Qtable[state])   # exploit
    else:
        return env.action_space.sample()  # explore

def greedy_policy(Qtable, state):
    return np.argmax(Qtable[state])

# -----------------------------
# 4. Training
# -----------------------------
def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    learning_rate = 0.1
    gamma = 0.99

    for episode in range(n_training_episodes):
        state, _ = env.reset()
        state = int(state)

        # Update epsilon using exponential decay
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)

            new_state, reward, terminated, truncated, info = env.step(action)
            new_state = int(new_state)
            done = terminated or truncated

            # Q-learning update
            Qtable[state, action] = Qtable[state, action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state, action]
            )

            state = new_state

            if done:
                break

    return Qtable

# -----------------------------
# 5. Evaluation
# -----------------------------
def evaluate(Qtable, env, n_eval_episodes, max_steps):
    total_rewards = 0

    for episode in range(n_eval_episodes):
        state, _ = env.reset()
        state = int(state)

        for step in range(max_steps):
            action = greedy_policy(Qtable, state)  # purely greedy
            new_state, reward, terminated, truncated, info = env.step(action)
            new_state = int(new_state)

            total_rewards += reward
            state = new_state
            if terminated or truncated:
                break

    avg_reward = total_rewards / n_eval_episodes
    print(f"Average reward over {n_eval_episodes} episodes: {avg_reward:.2f}")
    return avg_reward

# -----------------------------
# Run everything
# -----------------------------
n_training_episodes = 20000
n_eval_episodes = 100
max_steps = 99
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.00005

Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon,
                          decay_rate, env, max_steps, Qtable_frozenlake)

print("Trained Q-table:")
print(Qtable_frozenlake)

# Evaluate performance
evaluate(Qtable_frozenlake, env, n_eval_episodes, max_steps)

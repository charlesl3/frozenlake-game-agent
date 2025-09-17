import numpy as np
from policies import epsilon_greedy_policy, greedy_policy

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    learning_rate = 0.1
    gamma = 0.99

    for episode in range(n_training_episodes):
        state, _ = env.reset()
        state = int(state)

        # Update epsilon using exponential decay
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon, env)

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

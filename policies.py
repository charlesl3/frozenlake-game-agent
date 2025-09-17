import numpy as np
import random

def epsilon_greedy_policy(Qtable, state, epsilon, env):
    """
    Chooses an action using epsilon-greedy:
      - exploit with prob (1 - epsilon)
      - explore (random action) with prob epsilon
    """
    if random.uniform(0, 1) > epsilon:
        return int(np.argmax(Qtable[state]))   # exploit
    else:
        return env.action_space.sample()       # explore

def greedy_policy(Qtable, state):
    """Purely greedy action (used for evaluation)."""
    return int(np.argmax(Qtable[state]))

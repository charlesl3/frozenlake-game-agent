# test/test_qlearning.py

# Trivial Q-table tests for CI

import numpy as np

def test_qtable_creation():
    """A Q-table can be created with the right shape."""
    n_states, n_actions = 5, 3
    Q = np.zeros((n_states, n_actions))
    assert Q.shape == (n_states, n_actions)

def test_qtable_update():
    """Updating a Q-value should change that entry only."""
    n_states, n_actions = 4, 2
    Q = np.zeros((n_states, n_actions))
    Q[1, 0] = 0.5
    assert Q[1, 0] == 0.5
    assert np.count_nonzero(Q) == 1

"""
Epsilon greedy and softmax policies for action selection have been created.
The poliies are implemented such that the number of states and actions are not hardcoded in order to run both CartPole and MountainCar environments.
"""

import numpy as np
from scipy.special import softmax

seed = 42
rg = np.random.RandomState(seed)

# Epsilon greedy
def choose_action_epsilon(Q, state, epsilon, rg=rg):
    q_values = Q[tuple(state)]
    # If all Q-values are zero or with probability epsilon, choose a random action
    if not q_values.any() or rg.rand()<epsilon:
        return rg.choice(Q.shape[-1])
    else:
        return np.argmax(q_values)

# Softmax
def choose_action_softmax(Q, state, tau, rg=rg):
    q_values = Q[tuple(state)]
    probs = softmax(q_values - np.max(q_values)) ** (1/tau)
    probs = probs / np.sum(probs)
    return rg.choice(Q.shape[-1], p=probs)

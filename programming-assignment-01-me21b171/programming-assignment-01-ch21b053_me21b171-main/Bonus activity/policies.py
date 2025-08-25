import numpy as np
from scipy.special import softmax

def choose_action_epsilon(Q, state, epsilon, rg=None):
    """
    Epsilon-greedy policy for action selection.
    """
    if rg is None:
        rg = np.random.default_rng()
    
    # Handle different state types
    if isinstance(state, (int, np.integer)):
        q_values = Q[state]
    else:
        q_values = Q[tuple(state)]
    
    n_actions = len(q_values)
    
    # With probability epsilon, choose a random action
    if rg.random() < epsilon:
        return rg.integers(n_actions)
    else:
        # Choose the greedy action (with random tie-breaking)
        max_q = np.max(q_values)
        greedy_actions = np.where(q_values == max_q)[0]
        return rg.choice(greedy_actions)

def choose_action_softmax(Q, state, tau, rg=None):
    """
    Softmax policy for action selection.
    """
    if rg is None:
        rg = np.random.default_rng()
    
    # Handle different state types
    if isinstance(state, (int, np.integer)):
        q_values = Q[state]
    else:
        q_values = Q[tuple(state)]
    
    # Handle numerical stability by subtracting the maximum Q-value
    q_values = q_values - np.max(q_values)
    
    # Calculate probabilities using softmax with temperature
    probs = softmax(q_values / tau)
    
    # Choose action based on calculated probabilities
    return rg.choice(len(probs), p=probs)

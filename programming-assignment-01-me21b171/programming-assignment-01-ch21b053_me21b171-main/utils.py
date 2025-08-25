import numpy as np

def create_bins(env, n_bins):
    """
    Create bins for each state variable. This is required for discretizing the state for sarsa and q-learning.
    """
    if env.spec.id == "CartPole-v1":
        bins = []
        bins.append(np.linspace(-4.8, 4.8, n_bins))    # Cart position
        bins.append(np.linspace(-3.0, 3.0, n_bins))    # Cart velocity
        bins.append(np.linspace(-0.418, 0.418, n_bins))# Pole angle
        bins.append(np.linspace(-3.5, 3.5, n_bins))    # Pole angular velocity
        return bins
    else:
        obs_low = env.observation_space.low
        obs_high = env.observation_space.high
        bins = []
        for i in range(len(obs_low)):
            bins.append(np.linspace(obs_low[i], obs_high[i], n_bins))
        return bins

def discretize_state(state, bins):
    """
    Discretize the state.
    """
    indices=[]
    for i, val in enumerate(state):
        # Subtract 1 as digitize is 1-indexed
        index = np.digitize(val, bins[i]) - 1
        # Ensure the index is within the bounds
        index = min(max(index, 0), len(bins[i]) - 2)
        indices.append(index)
    return tuple(indices)
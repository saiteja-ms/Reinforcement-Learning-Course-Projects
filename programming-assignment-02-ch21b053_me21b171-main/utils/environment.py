import gymnasium as gym
import numpy as np
import torch

def make_env(env_name, seed=None):
    """
    Create and configure a Gymnasium environment.
    
    Args:
        env_name (str): Name of the environment (Acrobot-v1 or CartPole-v1)
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        gym.Env: The configured environment
    """
    env =gym.make(env_name)
    if seed is not None:   # Set the seed if provided
        env.reset(seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Get the environment properties
    state_dim = env.observation_space.shape[0]

    # Check if the action space is discrete or continuous
    # Discrete action space: action_dim = number of actions
    if isinstance(env.action_space, gym.spaces.Discrete): 
        action_dim = env.action_space.n
        discrete = True 
    # Continuous action space: action_dim = number of dimensions
    else:
        action_dim = env.action_space.shape[0] 
        discrete = False

    return env, state_dim, action_dim, discrete
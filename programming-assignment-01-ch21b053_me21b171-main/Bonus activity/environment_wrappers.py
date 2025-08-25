import gymnasium as gym
import numpy as np
from utils import create_bins, discretize_state

class CartPoleWrapper:
    """
    Wrapper for the CartPole-v1 environment.
    """
    def __init__(self, seed=None, n_bins=20):
        self.env = gym.make('CartPole-v1')
        if seed is not None:
            self.env.reset(seed=seed)
            
        self.bins = create_bins(self.env, n_bins)
        self.n_actions = self.env.action_space.n
        
        # Create state shape for Q-table
        self.state_shape = tuple([n_bins-1] * len(self.env.observation_space.low))
        
    def reset(self):
        return self.discretize_state(self.env.reset()[0])
    
    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return self.discretize_state(obs), reward, done
    
    def discretize_state(self, state):
        return discretize_state(state, self.bins)
    
    def create_q_table(self):
        return np.zeros(self.state_shape + (self.n_actions,))

class MountainCarWrapper:
    """
    Wrapper for the MountainCar-v0 environment.
    """
    def __init__(self, seed=None, n_bins=20):
        self.env = gym.make('MountainCar-v0')
        if seed is not None:
            self.env.reset(seed=seed)
            
        self.bins = create_bins(self.env, n_bins)
        self.n_actions = self.env.action_space.n
        
        # Create state shape for Q-table
        self.state_shape = tuple([n_bins-1] * len(self.env.observation_space.low))
        
    def reset(self):
        return self.discretize_state(self.env.reset()[0])
    
    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return self.discretize_state(obs), reward, done
    
    def discretize_state(self, state):
        return discretize_state(state, self.bins)
    
    def create_q_table(self):
        return np.zeros(self.state_shape + (self.n_actions,))

class MiniGridWrapper:
    """
    Wrapper for the MiniGrid-Dynamic-Obstacles-5x5-v0 environment.
    """
    def __init__(self, seed=None):
        try:
            import minigrid
            from minigrid.wrappers import ImgObsWrapper
        except ImportError:
            raise ImportError("Please install minigrid: pip install minigrid")
            
        self.env = gym.make('MiniGrid-Dynamic-Obstacles-5x5-v0')
        self.env = ImgObsWrapper(self.env)  # Use image observations
        
        if seed is not None:
            self.env.reset(seed=seed)
            
        self.n_actions = self.env.action_space.n
        
        # For MiniGrid, we'll use a different Q-table structure
        # Instead of a multi-dimensional array, use a dictionary-based approach
        self.state_dim = 5  # Number of features we'll extract
        
    def reset(self):
        obs, _ = self.env.reset()
        return self.discretize_state(obs)
    
    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return self.discretize_state(obs), reward, done
    
    def discretize_state(self, img_obs):
        """
        Extract key features from the image observation and return a simple state index.
        """
        # Extract simple features and convert to a single integer index
        flat_obs = img_obs.flatten()
        
        # Create a simple hash of the observation to use as state index
        # This avoids the dimensionality issues
        state_hash = hash(flat_obs.tobytes()) % 10000  # Limit to reasonable size
        return state_hash
    
    def create_q_table(self):
        # Create a dictionary-based Q-table for MiniGrid
        # This will dynamically grow as new states are encountered
        return np.zeros((10000, self.n_actions))  # Pre-allocate for 10000 possible states

import numpy as np
import matplotlib.pyplot as plt
import os

def create_bins(env, n_bins):
    """    
    Create bins for each state variable for discretizing continuous state spaces.
    """ 
    obs_low = env.observation_space.low
    obs_high = env.observation_space.high
    
    # Handle infinite bounds (common in CartPole)
    obs_low = np.where(np.isinf(obs_low), -10.0, obs_low)
    obs_high = np.where(np.isinf(obs_high), 10.0, obs_high)
    
    bins = []
    for i in range(len(obs_low)):
        bins.append(np.linspace(obs_low[i], obs_high[i], n_bins))
    return bins

def discretize_state(state, bins):
    """
    Discretize a continuous state using the provided bins.
    """
    indices = []
    for i, val in enumerate(state):
        # Clip the value to be within the bin range
        clipped_val = np.clip(val, bins[i][0], bins[i][-1])
        # Subtract 1 as digitize is 1-indexed
        index = np.digitize(clipped_val, bins[i]) - 1
        # Ensure the index is within the bounds
        index = min(max(index, 0), len(bins[i]) - 2)
        indices.append(index)
    return tuple(indices)

def plot_comparison(sarsa_rewards, q_rewards, env_name, save_dir='plots'):
    """
    Plot comparison between SARSA and Q-learning results.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Calculate mean and std for SARSA
    sarsa_mean = np.mean(sarsa_rewards, axis=0)
    sarsa_std = np.std(sarsa_rewards, axis=0)
    
    # Calculate mean and std for Q-learning
    q_mean = np.mean(q_rewards, axis=0)
    q_std = np.std(q_rewards, axis=0)
    
    episodes = np.arange(len(sarsa_mean))
    
    # Plot SARSA
    plt.plot(episodes, sarsa_mean, color='blue', label='SARSA (ε-greedy)')
    plt.fill_between(episodes, sarsa_mean - sarsa_std, sarsa_mean + sarsa_std, 
                     color='blue', alpha=0.2)
    
    # Plot Q-learning
    plt.plot(episodes, q_mean, color='red', label='Q-Learning (Softmax)')
    plt.fill_between(episodes, q_mean - q_std, q_mean + q_std, 
                     color='red', alpha=0.2)
    
    plt.xlabel('Episode Number', fontsize=14)
    plt.ylabel('Episodic Return', fontsize=14)
    plt.title(f'SARSA vs Q-Learning on {env_name}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{env_name}_comparison.png", dpi=300)
    plt.close()
    
    print(f"Comparison plot saved to {save_dir}/{env_name}_comparison.png")

def save_hyperparameters(env_name, algo_name, hyperparams, performance, save_dir='results'):
    """
    Save hyperparameters and their performance for later analysis.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"{save_dir}/{env_name}_{algo_name}_hyperparams.txt"
    
    with open(filename, 'a') as f:
        f.write(f"Performance: {performance}\n")
        for param, value in hyperparams.items():
            f.write(f"{param}: {value}\n")
        f.write("-" * 50 + "\n")
    
    print(f"Hyperparameters saved to {filename}")

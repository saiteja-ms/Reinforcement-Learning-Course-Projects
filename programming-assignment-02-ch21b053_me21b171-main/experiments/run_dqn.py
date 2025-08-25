import os
import numpy as np
import torch
from utils.environment import make_env
from models.dueling_dqn import DuelingDQNAgent
from utils.plotting import plot_results

def run_dueling_dqn_experiments(env_name, num_seeds=5, num_episodes=200, 
                               lr=0.001, gamma=0.99, save_dir='results'):
    """Run experiments for both types of Dueling DQN."""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Lists to store returns for both variants
    type1_returns = []
    type2_returns = []
    
    for seed in range(num_seeds):
        print(f"\n--- Running with seed {seed} ---")
        
        # Set up environment
        env, state_dim, action_dim, _ = make_env(env_name, seed)
        
        # Type-1: Average advantage
        print("\n--- Training Type-1 Dueling DQN (Average Advantage) ---")
        agent1 = DuelingDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            aggregation_type=1,  # Type-1: Average advantage
            lr=lr,
            gamma=gamma
        )
        
        returns1 = agent1.train(env, num_episodes)
        type1_returns.append(returns1)
        
        # Type-2: Max advantage
        print("\n--- Training Type-2 Dueling DQN (Max Advantage) ---")
        agent2 = DuelingDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            aggregation_type=2,  # Type-2: Max advantage
            lr=lr,
            gamma=gamma
        )
        
        returns2 = agent2.train(env, num_episodes)
        type2_returns.append(returns2)
        
        # Close environment
        env.close()
    
    # Plot results
    plt = plot_results(
        [type1_returns, type2_returns],
        title=f"Dueling DQN in {env_name}",
        variant_names=["Type-1 (Average)", "Type-2 (Max)"]
    )
    
    # Save plot
    plt.savefig(os.path.join(save_dir, f"{env_name}_dueling_dqn_comparison.png"))
    
    return type1_returns, type2_returns

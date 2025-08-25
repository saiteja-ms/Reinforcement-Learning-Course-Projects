import os
import numpy as np
import torch

from utils.environment import make_env
from models.reinforce import REINFORCEAgent
from utils.plotting import plot_results

def run_reinforce_experiments(env_name, num_seeds=5, num_episodes=200, 
                              lr_policy=1e-3, lr_value=1e-3, gamma=0.99,
                              save_dir='results'):
    """
    Run experiments for REINFORCE algorithm with and without baseline.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Lists to store returns for each variant
    no_baseline_returns = []
    with_baseline_returns = []

    for seed in range(num_seeds):
        print(f"\n--- Running with seed {seed} ---")

        # Setup the environment
        env, state_dim, action_dim, _ = make_env(env_name, seed)

        # Without baseline
        print("\n--- Training REINFORCE without Baseline ---")
        agent1 = REINFORCEAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            use_baseline=False,
            lr_policy=lr_policy,
            gamma=gamma
        )

        returns1 = agent1.train(env, num_episodes)
        no_baseline_returns.append(returns1)

        # With baseline
        print("\n--- Training REINFORCE with Baseline ---")
        agent2 = REINFORCEAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            use_baseline=True,
            lr_policy=lr_policy,
            lr_value=lr_value,
            gamma=gamma
        )

        returns2 = agent2.train(env, num_episodes)
        with_baseline_returns.append(returns2)

        # Close the environment
        env.close()

    # Plot the results
    plt = plot_results(
        [no_baseline_returns, with_baseline_returns],
        title=f"REINFORCE in {env_name}",
        variant_names=["Without Baseline", "With Baseline"]
    )

    # Save the plot
    plt.savefig(os.path.join(save_dir, f"{env_name}_reinforce_comparison.png"))

    return no_baseline_returns, with_baseline_returns
 
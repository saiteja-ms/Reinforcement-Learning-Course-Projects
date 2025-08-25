# run_experiments.py

import gym
import numpy as np
import matplotlib.pyplot as plt
from smdp_q_learning import smdp_q_learning, plot_rewards, plot_q_values
from intra_option_q_learning import intra_option_q_learning
from tqdm import tqdm  # Import tqdm for progress bar

# Run the experiment for a given algorithm
def run_algorithm(algorithm, env, num_episodes, alpha, gamma, epsilon, algo_name):
    print(f"Running {algo_name}...")
    
    # Execute the algorithm
    if algo_name == "SMDP Q-Learning":
        Q, rewards = smdp_q_learning(env, num_episodes, alpha, gamma, epsilon)
    elif algo_name == "Intra-Option Q-Learning":
        Q, rewards = intra_option_q_learning(env, num_episodes, alpha, gamma, epsilon)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    # Plot rewards per episode
    plot_rewards(rewards, title=f"{algo_name} - Reward Plot for Original Option Set")
    
    # Plot the learned Q-values
    plot_q_values(Q, title=f"{algo_name} - Q-values Visualization")

    return rewards

def main():
    # Create the taxi environment
    env = gym.make("Taxi-v3")
    
    # Hyperparameters
    num_episodes = 1000
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1

    # Run the SMDP Q-Learning experiment
    smdp_rewards = run_algorithm(smdp_q_learning, env, num_episodes, alpha, gamma, epsilon, "SMDP Q-Learning")
    
    # Run the Intra-Option Q-Learning experiment
    intra_rewards = run_algorithm(intra_option_q_learning, env, num_episodes, alpha, gamma, epsilon, "Intra-Option Q-Learning")
    
    # Compare the reward curves for both algorithms
    plt.plot(smdp_rewards, label="SMDP Q-Learning")
    plt.plot(intra_rewards, label="Intra-Option Q-Learning")
    plt.title("Reward Comparison: SMDP vs Intra-Option Q-Learning for Original Option Set")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from utils import create_bins, discretize_state
from q_learning import q_learning
from sarsa import sarsa
from policies import choose_action_softmax, choose_action_epsilon

# Let's create a Discretized environment wrapperusing the utils functions.

class DiscretizedEnv:
    def __init__(self, env, bins):
        self.env = env
        self.bins = bins

    def reset(self):
        state, _ = self.env.reset()
        return discretize_state(state, self.bins)
    
    def step(self, action):
        state, reward, done, truncated, _ = self.env.step(action)
        terminal = done or truncated
        return discretize_state(state, self.bins), reward, terminal

def run_experiment(env_name, algorithm, policy_fn, episodes, seeds, gamma, alpha, 
                   epsilon, epsilon_decay=None, epsilon_min=None, 
                   tau=None, tau_decay=None, tau_min=None, n_bins=20):
    all_rewards = np.zeros((len(seeds), episodes))
    print_freq = 100


    for i, seed in enumerate(seeds):
        env = gym.make(env_name)
        env.reset(seed=seed)

        bins = create_bins(env, n_bins)
        # Discretize the environment
        d_env = DiscretizedEnv(env, bins)

        n_actions = env.action_space.n
        # Initialize Q-table
        state_shape = (n_bins-1,)*len(env.observation_space.low)
        Q = np.zeros(state_shape + (n_actions,))

        if policy_fn == choose_action_epsilon:
            rewards, _ = algorithm(d_env, Q, episodes, gamma=gamma, alpha=alpha, epsilon=epsilon, 
                                   epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, 
                                   print_freq=print_freq, choose_action=policy_fn)
        else:
            rewards, _ = algorithm(d_env, Q, episodes, gamma=gamma, alpha=alpha, tau=tau, 
                                   tau_decay=tau_decay, tau_min=tau_min, 
                                   print_freq=print_freq, choose_action=policy_fn)
        all_rewards[i, :] = rewards

    return all_rewards

def plot_results(all_rewards, title):
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    episodes_range = np.arange(len(mean_rewards))
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, mean_rewards, color='green', label="Mean Return")
    plt.fill_between(episodes_range, mean_rewards - std_rewards, mean_rewards + std_rewards, color='blue', alpha=0.2,
                     label="Std Dev")
    plt.xlabel("Episode")
    plt.ylabel("Episodic Return")
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run RL algorithms on a Gym environment.")
    parser.add_argument("--env", type=str, default="MountainCar-v0",
                        help="Gym environment to run (e.g., MountainCar-v0, CartPole-v1)")
    parser.add_argument("--algorithm", type=str, default="sarsa",
                        choices=["sarsa", "qlearning", "both"],
                        help="Algorithm to run: sarsa, qlearning, or both")
    parser.add_argument("--episodes", type=int, default=12000, help="Number of episodes to run")
    parser.add_argument("--n_bins", type=int, default=20, help="Number of bins per state dimension")
    parser.add_argument("--tau_decay", type=float, default=0.999, help="Tau_decay")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon_decay")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate")
    args = parser.parse_args()

    seeds = [42, 43, 44, 45, 46]
    gamma = 0.99

    epsilon = 1.0
    
    epsilon_min = 0.05

    # For softmax exploration (Q-learning), use tau (temperature)
    tau = 2.0
    tau_min = 0.1
    
    if args.algorithm in ["sarsa", "both"]:
        print("Running SARSA with epsilon-greedy exploration on", args.env)
        sarsa_rewards = run_experiment(args.env, sarsa, choose_action_epsilon, args.episodes, seeds, 
                               gamma, args.alpha, epsilon, epsilon_decay=args.epsilon_decay, 
                               epsilon_min=epsilon_min, n_bins=args.n_bins)
        plot_results(sarsa_rewards, f"SARSA on {args.env}")

    if args.algorithm in ["qlearning", "both"]:
        print("Running Q-Learning with softmax exploration on", args.env)
        q_rewards = run_experiment(args.env, q_learning, choose_action_softmax, args.episodes, seeds, 
                           gamma, args.alpha, epsilon, tau=tau, tau_decay=args.tau_decay, tau_min=tau_min,
                           n_bins=args.n_bins)
        plot_results(q_rewards, f"Q-Learning on {args.env}")
        
if __name__ == '__main__':
    main()
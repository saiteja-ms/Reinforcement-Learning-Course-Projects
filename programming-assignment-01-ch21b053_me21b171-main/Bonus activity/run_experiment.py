import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import wandb
import os
from minigrid.wrappers import ImgObsWrapper

from utils import create_bins, discretize_state
from q_learning import q_learning
from sarsa import sarsa
from policies import choose_action_softmax, choose_action_epsilon

# MiniGrid wrapper for the dynamic obstacles environment
class MiniGridEnv:
    def __init__(self, seed=None):
        self.env = gym.make('MiniGrid-Dynamic-Obstacles-5x5-v0')
        self.env = ImgObsWrapper(self.env)  # Use image observations
        
        if seed is not None:
            self.env.reset(seed=seed)
            
        self.n_actions = self.env.action_space.n
        # For MiniGrid, we'll use a simplified state representation
        self.state_shape = (2, 2, 2, 2, 2)  # 5 binary features
        
    def reset(self):
        obs, _ = self.env.reset()
        return self.discretize_state(obs)
    
    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return self.discretize_state(obs), reward, done
    
    def discretize_state(self, img_obs):
        """
        Extract and discretize key features from the image observation.
        """
        flat_obs = img_obs.flatten()
        
        # Extract 5 binary features
        features = []
        # Check for agent position (blue pixel)
        features.append(1 if np.any(flat_obs[2::3] > 0.5) else 0)
        # Check for goal (green pixel)
        features.append(1 if np.any(flat_obs[1::3] > 0.5) else 0)
        # Check for obstacles in different quadrants (red pixels)
        features.append(1 if np.any(flat_obs[:len(flat_obs)//3:3] > 0.5) else 0)
        features.append(1 if np.any(flat_obs[len(flat_obs)//3:2*len(flat_obs)//3:3] > 0.5) else 0)
        features.append(1 if np.any(flat_obs[2*len(flat_obs)//3::3] > 0.5) else 0)
        
        return tuple(features)
    
    def create_q_table(self):
        return np.zeros(self.state_shape + (self.n_actions,))

def run_experiment(env_type, algorithm, policy_fn, episodes, seeds, gamma, alpha, 
                   epsilon=None, epsilon_decay=None, epsilon_min=None, 
                   tau=None, tau_decay=None, tau_min=None, n_bins=5, use_wandb=True):
    all_rewards = np.zeros((len(seeds), episodes))
    print_freq = 100

    # Create descriptive name for wandb run
    if algorithm.__name__ == 'sarsa':
        algo_name = f"sarsa_alpha{alpha}_epsilon{epsilon}_decay{epsilon_decay}_min{epsilon_min}"
    else:
        algo_name = f"qlearning_alpha{alpha}_tau{tau}_decay{tau_decay}_min{tau_min}"

    for i, seed in enumerate(seeds):
        # Initialize wandb for this seed
        if use_wandb:
            run_name = f"{algo_name}_seed{seed}"
            wandb.init(project="RL_Assignment", name=run_name, reinit=True)
            wandb.config.update({
                "algorithm": algorithm.__name__,
                "environment": env_type,
                "gamma": gamma,
                "alpha": alpha,
                "seed": seed,
                "episodes": episodes
            })
            if algorithm.__name__ == 'sarsa':
                wandb.config.update({
                    "epsilon": epsilon,
                    "epsilon_decay": epsilon_decay,
                    "epsilon_min": epsilon_min
                })
            else:
                wandb.config.update({
                    "tau": tau,
                    "tau_decay": tau_decay,
                    "tau_min": tau_min
                })

        # Create environment based on type
        if env_type == "MiniGrid-Dynamic-Obstacles-5x5-v0":
            env = MiniGridEnv(seed=seed)
            Q = env.create_q_table()
        else:
            env_gym = gym.make(env_type)
            env_gym.reset(seed=seed)
            bins = create_bins(env_gym, n_bins)
            env = DiscretizedEnv(env_gym, bins)
            n_actions = env_gym.action_space.n
            state_shape = (n_bins-1,)*len(env_gym.observation_space.low)
            Q = np.zeros(state_shape + (n_actions,))

        # Run algorithm
        if algorithm.__name__ == 'sarsa':
            rewards, _ = algorithm(env, Q, episodes, gamma=gamma, alpha=alpha, epsilon=epsilon, 
                                   epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, 
                                   print_freq=print_freq, choose_action=policy_fn)
        else:
            rewards, _ = algorithm(env, Q, episodes, gamma=gamma, alpha=alpha, tau=tau, 
                                   tau_decay=tau_decay, tau_min=tau_min, 
                                   print_freq=print_freq, choose_action=policy_fn)
        
        all_rewards[i, :] = rewards
        
        # Log rewards to wandb
        if use_wandb:
            for ep, reward in enumerate(rewards):
                wandb.log({"episode": ep, "reward": reward})
            wandb.finish()
    
    return all_rewards  # Added the missing return statement

def plot_results(all_rewards, title, save_path=None):
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    episodes_range = np.arange(len(mean_rewards))
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_range, mean_rewards, label="Mean Return")
    plt.fill_between(episodes_range, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2,
                     label="Std Dev")
    plt.xlabel("Episode Number")
    plt.ylabel("Episodic Return")
    plt.title(title)
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def setup_wandb_sweep(env_type, algorithm_name):
    if algorithm_name == "sarsa":
        sweep_config = {
            'method': 'bayes',
            'metric': {'name': 'final_reward', 'goal': 'maximize'},
            'parameters': {
                'alpha': {'min': 0.001, 'max': 0.5},
                'epsilon': {'min': 0.1, 'max': 1.0},
                'epsilon_decay': {'min': 0.9, 'max': 0.999},
                'epsilon_min': {'min': 0.01, 'max': 0.2}
            }
        }
    else:  # q_learning
        sweep_config = {
            'method': 'bayes',
            'metric': {'name': 'final_reward', 'goal': 'maximize'},
            'parameters': {
                'alpha': {'min': 0.001, 'max': 0.5},
                'tau': {'min': 0.1, 'max': 2.0},
                'tau_decay': {'min': 0.9, 'max': 0.999},
                'tau_min': {'min': 0.01, 'max': 0.5}
            }
        }
    
    sweep_id = wandb.sweep(sweep_config, project="RL_Assignment_minigrid")
    return sweep_id

def sweep_agent(env_type, algorithm_name, episodes=1000, seeds=[42, 43, 44, 45, 46], gamma=0.99):
    def train():
        # Initialize wandb
        wandb.init()
        
        # Get hyperparameters from wandb
        config = wandb.config
        
        if algorithm_name == "sarsa":
            algorithm = sarsa
            policy_fn = choose_action_epsilon
            rewards = run_experiment(
                env_type, algorithm, policy_fn, episodes, seeds, gamma,
                alpha=config.alpha, epsilon=config.epsilon, 
                epsilon_decay=config.epsilon_decay, epsilon_min=config.epsilon_min
            )
        else:  # q_learning
            algorithm = q_learning
            policy_fn = choose_action_softmax
            rewards = run_experiment(
                env_type, algorithm, policy_fn, episodes, seeds, gamma,
                alpha=config.alpha, tau=config.tau, 
                tau_decay=config.tau_decay, tau_min=config.tau_min
            )
        
        # Calculate final performance (average of last 100 episodes across all seeds)
        mean_rewards = np.mean(rewards, axis=0)
        final_reward = np.mean(mean_rewards[-100:])
        
        # Log final performance
        wandb.log({"final_reward": final_reward})
    
    return train

def main():
    parser = argparse.ArgumentParser(description="Run RL algorithms on a Gym environment.")
    parser.add_argument("--env", type=str, default="MiniGrid-Dynamic-Obstacles-5x5-v0",
                        help="Gym environment to run (e.g., MountainCar-v0, CartPole-v1, MiniGrid-Dynamic-Obstacles-5x5-v0)")
    parser.add_argument("--algorithm", type=str, default="both",  # Changed to run both by default
                        choices=["sarsa", "qlearning", "both"],
                        help="Algorithm to run: sarsa, qlearning, or both")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to run")
    parser.add_argument("--n_bins", type=int, default=5, help="Number of bins per state dimension")
    parser.add_argument("--sweep", action="store_true", help="Run wandb Bayesian sweep")
    parser.add_argument("--sweep_count", type=int, default=5, help="Number of sweep runs")
    args = parser.parse_args()

    seeds = [42, 43, 44, 45, 46]
    gamma = 0.99
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Run with sweep by default for MiniGrid
    if args.env == "MiniGrid-Dynamic-Obstacles-5x5-v0":
        args.sweep = True
    
    if args.sweep:
        # Run Bayesian hyperparameter sweep with wandb
        if args.algorithm in ["sarsa", "both"]:
            print(f"Running SARSA sweep with {args.sweep_count} iterations...")
            sweep_id = setup_wandb_sweep(args.env, "sarsa")
            wandb.agent(sweep_id, sweep_agent(args.env, "sarsa", args.episodes, seeds, gamma), count=args.sweep_count)
            
        if args.algorithm in ["qlearning", "both"]:
            print(f"Running Q-Learning sweep with {args.sweep_count} iterations...")
            sweep_id = setup_wandb_sweep(args.env, "qlearning")
            wandb.agent(sweep_id, sweep_agent(args.env, "qlearning", args.episodes, seeds, gamma), count=args.sweep_count)
    else:
        # Run with default hyperparameters
        if args.algorithm in ["sarsa", "both"]:
            print(f"Running SARSA with epsilon-greedy exploration on {args.env}")
            sarsa_rewards = run_experiment(
                args.env, sarsa, choose_action_epsilon, args.episodes, seeds, 
                gamma, alpha=0.1, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05,
                n_bins=args.n_bins
            )
            plot_results(sarsa_rewards, f"SARSA on {args.env}", f"plots/sarsa_{args.env.replace('-', '_')}.png")

        if args.algorithm in ["qlearning", "both"]:
            print(f"Running Q-Learning with softmax exploration on {args.env}")
            q_rewards = run_experiment(
                args.env, q_learning, choose_action_softmax, args.episodes, seeds, 
                gamma, alpha=0.1, tau=1.0, tau_decay=0.995, tau_min=0.1,
                n_bins=args.n_bins
            )
            plot_results(q_rewards, f"Q-Learning on {args.env}", f"plots/qlearning_{args.env.replace('-', '_')}.png")
        
if __name__ == '__main__':
    main()

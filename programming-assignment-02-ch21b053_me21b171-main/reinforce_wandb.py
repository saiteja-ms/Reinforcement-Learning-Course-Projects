import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import matplotlib.pyplot as plt
import wandb
import os
from typing import Tuple, List, Dict, Optional
import json

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99  # Fixed gamma as per assignment
NUM_SEEDS_FINAL = 5
NUM_EPISODES_FINAL = 500
NUM_EPISODES_SWEEP = 200  # Fewer episodes for tuning
SWEEP_SEED = 42          # Fixed seed for tuning runs
SWEEP_COUNT = 20         # Number of runs per sweep

print(f"Using device: {DEVICE}")

# --- Utility Functions ---
def set_seeds(seed: int):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- Neural Network Models ---
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1)
        )
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

# --- REINFORCE Agent ---
class REINFORCEAgent:
    def __init__(self, state_dim: int, action_dim: int, use_baseline: bool = False,
                 policy_lr: float = 1e-3, value_lr: float = 1e-3, gamma: float = GAMMA,
                 hidden_dim: int = 128):
        self.gamma = gamma
        self.use_baseline = use_baseline

        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        if use_baseline:
            self.value_net = ValueNetwork(state_dim, hidden_dim).to(DEVICE)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)

        self.trajectory_log_probs = []
        self.trajectory_rewards = []
        self.trajectory_states = []

    def select_action(self, state: np.ndarray) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        self.trajectory_states.append(state_tensor) # Store tensor for baseline calculation

        probs = self.policy_net(state_tensor)
        m = Categorical(probs)
        action = m.sample()
        self.trajectory_log_probs.append(m.log_prob(action))
        return action.item()

    def store_reward(self, reward: float):
        self.trajectory_rewards.append(reward)

    def finish_episode(self) -> Tuple[float, float]:
        returns = []
        discounted_reward = 0
        for reward in reversed(self.trajectory_rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns).to(DEVICE)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9) # Added epsilon for stability

        policy_loss_terms = []
        value_loss_terms = []

        # Prepare states tensor if using baseline
        states_tensor = None
        if self.use_baseline:
           states_tensor = torch.cat(self.trajectory_states)

        baselines = torch.zeros_like(returns)
        if self.use_baseline:
             with torch.no_grad(): # Get baselines without tracking gradients for advantage calculation
                 baselines = self.value_net(states_tensor).squeeze()

        for i, log_prob in enumerate(self.trajectory_log_probs):
            advantage = returns[i] - baselines[i] # Advantage calculation
            policy_loss_terms.append(-log_prob * advantage.detach()) # Detach advantage for policy update

            if self.use_baseline:
                 # Recalculate baseline with gradients for value loss
                 current_baseline = self.value_net(self.trajectory_states[i]).squeeze()
                 value_loss_terms.append(F.mse_loss(current_baseline, returns[i]))

        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss_terms).sum()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update value network if using baseline
        value_loss_val = 0.0
        if self.use_baseline and value_loss_terms:
            self.value_optimizer.zero_grad()
            value_loss = torch.stack(value_loss_terms).mean() # Use mean MSE loss
            value_loss.backward()
            self.value_optimizer.step()
            value_loss_val = value_loss.item()

        # Clear trajectory data
        self.trajectory_log_probs = []
        self.trajectory_rewards = []
        self.trajectory_states = []

        return policy_loss.item(), value_loss_val

# --- Training Functions ---

def run_single_episode(env, agent, max_steps=1000):
    """Runs a single episode and returns the total reward."""
    state, _ = env.reset()
    episode_reward = 0
    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.store_reward(reward)
        state = next_state
        episode_reward += reward
        if done or truncated:
            break
    return episode_reward

def train_reinforce_run(is_sweep_run: bool = False):
    """
    Performs a single training run (either for sweep or final evaluation).
    Logs metrics to wandb.
    Returns:
        - final_avg_reward (float) if is_sweep_run is True.
        - episode_rewards (List[float]) if is_sweep_run is False.
    """
    run = wandb.init() # Initialize run (will inherit config from sweep agent if applicable)
    config = wandb.config

    env = gym.make(config.env_name)
    env.reset(seed=config.seed)
    set_seeds(config.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        use_baseline=config.use_baseline,
        policy_lr=config.policy_lr,
        value_lr=config.value_lr if config.use_baseline else 1e-3, # Default value_lr if not in config
        gamma=GAMMA,
        hidden_dim=config.hidden_dim
    )

    episode_rewards = []
    total_steps = 0

    for episode in range(config.num_episodes):
        episode_reward = run_single_episode(env, agent)
        policy_loss, value_loss = agent.finish_episode()
        episode_rewards.append(episode_reward)
        total_steps += len(agent.trajectory_rewards) # Track steps if needed

        # Log to wandb
        log_dict = {
            "episode": episode + 1,
            "episode_reward": episode_reward,
            "policy_loss": policy_loss,
            "avg_reward_last_100": np.mean(episode_rewards[-100:]) if episode >= 99 else np.mean(episode_rewards)
        }
        if config.use_baseline:
            log_dict["value_loss"] = value_loss
        wandb.log(log_dict)

        if (episode + 1) % 50 == 0:
            print(f"Run: {run.name}, Ep: {episode+1}/{config.num_episodes}, Reward: {episode_reward:.2f}, AvgR(100): {log_dict['avg_reward_last_100']:.2f}")

    # --- Final Logging ---
    final_avg_reward = np.mean(episode_rewards[-20:]) # Metric for sweep optimization
    wandb.log({"final_avg_reward": final_avg_reward})
    wandb.run.summary["final_avg_reward_all_eps"] = np.mean(episode_rewards)
    wandb.run.summary["final_avg_reward_last_100"] = np.mean(episode_rewards[-100:])
    wandb.run.summary["algorithm_variant"] = config.algorithm_variant

    env.close()

    if is_sweep_run:
        return final_avg_reward # Return metric for sweep agent
    else:
        return episode_rewards # Return full history for final plotting

# --- Sweep Configuration and Execution ---

def setup_sweep_config(env_name: str, use_baseline: bool) -> Dict:
    """Creates the sweep configuration dictionary."""
    baseline_str = "WithBaseline" if use_baseline else "WithoutBaseline"
    sweep_name = f"SWEEP_REINFORCE_{baseline_str}_{env_name}"
    algorithm_variant = f"REINFORCE_{baseline_str}"

    sweep_config = {
        "name": sweep_name, # Add sweep name here
        "method": "bayes",
        "metric": {"name": "final_avg_reward", "goal": "maximize"},
        "parameters": {
            "env_name": {"value": env_name},
            "use_baseline": {"value": use_baseline},
            "algorithm_variant": {"value": algorithm_variant},
            "policy_lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 5e-3}, # Adjusted range
            "hidden_dim": {"values": [64, 128, 256]},
            "num_episodes": {"value": NUM_EPISODES_SWEEP},
            "seed": {"value": SWEEP_SEED} # Fixed seed for tuning
        }
    }
    if use_baseline:
        sweep_config["parameters"]["value_lr"] = {"distribution": "log_uniform_values", "min": 1e-5, "max": 5e-3} # Adjusted range

    return sweep_config

def run_sweep(env_name: str, use_baseline: bool, sweep_count: int = SWEEP_COUNT) -> str:
    """Initializes and runs a wandb sweep."""
    project_name = f"REINFORCE_{env_name.replace('-', '_')}"
    sweep_config = setup_sweep_config(env_name, use_baseline)
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"--- Starting Sweep {sweep_id} for {env_name} ({'With' if use_baseline else 'Without'} Baseline) ---")
    # Use a lambda to pass the is_sweep_run=True argument
    wandb.agent(sweep_id, function=lambda: train_reinforce_run(is_sweep_run=True), count=sweep_count)
    print(f"--- Finished Sweep {sweep_id} ---")
    return sweep_id

def get_best_hyperparams(sweep_id: str, project_name: str, wandb_entity: str) -> Optional[Dict]:
    """Retrieves the best hyperparameters from a completed sweep."""
    try:
        api = wandb.Api()
        sweep_path = f"{wandb_entity}/{project_name}/{sweep_id}"
        print(f"Fetching best run from sweep: {sweep_path}")
        sweep = api.sweep(sweep_path)
        best_run = sweep.best_run(order="final_avg_reward") # Ensure metric name matches

        if best_run:
            print(f"Best run found: {best_run.name} with score {best_run.summary.get('final_avg_reward', 'N/A')}")
            params = {
                "policy_lr": best_run.config["policy_lr"],
                "hidden_dim": best_run.config["hidden_dim"]
            }
            if best_run.config.get("use_baseline", False):
                 # Check if value_lr exists in config (it should for baseline runs)
                 if "value_lr" in best_run.config:
                    params["value_lr"] = best_run.config["value_lr"]
                 else:
                    print("Warning: value_lr not found in best run config for baseline=True. Using default.")
                    params["value_lr"] = 1e-3 # Fallback default
            return params
        else:
            print(f"Warning: No finished runs found for sweep {sweep_id}. Using default parameters.")
            return None
    except Exception as e:
        print(f"Error retrieving best hyperparameters for sweep {sweep_id}: {e}")
        print("Using default parameters.")
        return None # Fallback to default if API call fails

# --- Final Experiment Execution ---

def run_final_experiments(env_name: str, use_baseline: bool, best_params: Dict,
                          num_episodes: int = NUM_EPISODES_FINAL, num_seeds: int = NUM_SEEDS_FINAL) -> List[List[float]]:
    """Runs the final experiments using the best hyperparameters across multiple seeds."""
    all_seed_rewards = []
    baseline_str = "WithBaseline" if use_baseline else "WithoutBaseline"
    project_name = f"REINFORCE_{env_name.replace('-', '_')}"
    algorithm_variant = f"REINFORCE_{baseline_str}"

    # Use default params if best_params is None
    if best_params is None:
         print("Using default parameters for final runs.")
         best_params = {
             "policy_lr": 1e-3,
             "hidden_dim": 128
         }
         if use_baseline:
             best_params["value_lr"] = 1e-3

    for seed in range(num_seeds):
        print(f"--- Running Final: REINFORCE {baseline_str} on {env_name} with Seed {seed} ---")

        config = {
            "env_name": env_name,
            "use_baseline": use_baseline,
            "algorithm_variant": algorithm_variant,
            "num_episodes": num_episodes,
            "seed": seed,
            **best_params # Unpack best hyperparameters
        }

        # Manually create wandb run for final evaluation
        run = wandb.init(
            project=project_name,
            name=f"FINAL_{algorithm_variant}_seed{seed}",
            config=config,
            group=f"{baseline_str}_FinalRuns",
            tags=["final_run", baseline_str, env_name],
            job_type="final_evaluation",
            reinit=True # Allows multiple wandb.init calls in one script
        )
        with run:
            # Call the training function, ensuring it returns the full list
            episode_rewards = train_reinforce_run(is_sweep_run=False)
            all_seed_rewards.append(episode_rewards)

    print(f"--- Finished Final Runs for REINFORCE {baseline_str} on {env_name} ---")
    return all_seed_rewards

# --- Plotting ---

def plot_comparison(env_name: str, without_baseline_rewards: List[List[float]], with_baseline_rewards: List[List[float]]):
    """Plots the comparison between variants and logs to wandb."""
    if not without_baseline_rewards or not with_baseline_rewards:
        print(f"Warning: Missing reward data for {env_name}. Skipping plot.")
        return

    # Pad shorter reward lists if necessary (e.g., if some runs had different episode counts)
    max_len = max(len(r) for r in without_baseline_rewards + with_baseline_rewards)
    padded_without = [r + [r[-1]] * (max_len - len(r)) for r in without_baseline_rewards]
    padded_with = [r + [r[-1]] * (max_len - len(r)) for r in with_baseline_rewards]

    without_mean = np.mean(padded_without, axis=0)
    without_std = np.std(padded_without, axis=0)

    with_mean = np.mean(padded_with, axis=0)
    with_std = np.std(padded_with, axis=0)

    episodes = range(1, max_len + 1)

    plt.figure(figsize=(12, 7)) # Adjusted size

    # Plot without baseline (Type 1 - Red)
    plt.plot(episodes, without_mean, color='red', label='REINFORCE (Without Baseline)')
    plt.fill_between(episodes, without_mean - without_std, without_mean + without_std, color='red', alpha=0.2)

    # Plot with baseline (Type 2 - Blue)
    plt.plot(episodes, with_mean, color='blue', label='REINFORCE (With Baseline)')
    plt.fill_between(episodes, with_mean - with_std, with_mean + with_std, color='blue', alpha=0.2)

    plt.xlabel('Episode Number', fontsize=12)
    plt.ylabel('Episodic Return', fontsize=12)
    plt.title(f'REINFORCE Comparison on {env_name}', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    # Save locally
    plot_filename = f'reinforce_{env_name.lower()}_comparison.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved comparison plot to {plot_filename}")

    # Log comparison plot and data to wandb
    project_name = f"REINFORCE_{env_name.replace('-', '_')}"
    run = wandb.init(project=project_name, name=f"Comparison_{env_name}", job_type="comparison_plot", reinit=True)
    with run:
        wandb.log({"comparison_plot": wandb.Image(plt)})

        # Log data for interactive plots
        log_data = []
        for i, episode in enumerate(episodes):
             log_data.append({
                 "episode": episode,
                 "without_baseline_mean": without_mean[i],
                 "without_baseline_std": without_std[i],
                 "with_baseline_mean": with_mean[i],
                 "with_baseline_std": with_std[i]
             })
        # Log as a table for better visualization
        wandb.log({"comparison_data": wandb.Table(data=log_data, columns=["episode", "without_baseline_mean", "without_baseline_std", "with_baseline_mean", "with_baseline_std"])})

    plt.close()

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- !! IMPORTANT !! ---
    # Replace 'your-wandb-username' with your actual wandb username or entity name
    WANDB_ENTITY = "your-wandb-username"
    # ---

    wandb.login()

    environments = ["CartPole-v1", "Acrobot-v1"]
    baseline_options = [False, True] # Without and With baseline

    all_best_params = {} # Store best params for all envs/variants

    # --- Hyperparameter Tuning Phase ---
    print("\n=== STARTING HYPERPARAMETER TUNING ===")
    for env_name in environments:
        all_best_params[env_name] = {}
        for use_baseline in baseline_options:
            baseline_str = "WithBaseline" if use_baseline else "WithoutBaseline"
            sweep_id = run_sweep(env_name, use_baseline, sweep_count=SWEEP_COUNT)
            project_name = f"REINFORCE_{env_name.replace('-', '_')}"
            all_best_params[env_name][baseline_str] = get_best_hyperparams(sweep_id, project_name, WANDB_ENTITY)

    # Save best parameters
    params_filename = 'reinforce_best_params.json'
    try:
        with open(params_filename, 'w') as f:
            json.dump(all_best_params, f, indent=4)
        print(f"\nBest hyperparameters saved to {params_filename}")
    except Exception as e:
        print(f"Error saving hyperparameters: {e}")

    # --- Final Evaluation Phase ---
    print("\n=== STARTING FINAL EVALUATION RUNS ===")
    all_final_rewards = {} # Store final rewards for plotting

    for env_name in environments:
        all_final_rewards[env_name] = {}
        for use_baseline in baseline_options:
            baseline_str = "WithBaseline" if use_baseline else "WithoutBaseline"
            best_params = all_best_params.get(env_name, {}).get(baseline_str, None) # Safely get params

            final_rewards = run_final_experiments(
                env_name=env_name,
                use_baseline=use_baseline,
                best_params=best_params, # Pass potentially None params
                num_episodes=NUM_EPISODES_FINAL,
                num_seeds=NUM_SEEDS_FINAL
            )
            all_final_rewards[env_name][baseline_str] = final_rewards

            # Save raw reward data
            rewards_filename = f'reinforce_{"with" if use_baseline else "without"}_baseline_{env_name.lower()}.npy'
            try:
                np.save(rewards_filename, np.array(final_rewards))
                print(f"Saved final rewards to {rewards_filename}")
            except Exception as e:
                print(f"Error saving final rewards: {e}")

    # --- Comparison Plotting Phase ---
    print("\n=== GENERATING COMPARISON PLOTS ===")
    for env_name in environments:
        rewards_without = all_final_rewards.get(env_name, {}).get("WithoutBaseline", [])
        rewards_with = all_final_rewards.get(env_name, {}).get("WithBaseline", [])

        if rewards_without and rewards_with:
            plot_comparison(env_name, rewards_without, rewards_with)
        else:
            print(f"Skipping comparison plot for {env_name} due to missing data.")

    print("\n=== All REINFORCE experiments completed! ===")

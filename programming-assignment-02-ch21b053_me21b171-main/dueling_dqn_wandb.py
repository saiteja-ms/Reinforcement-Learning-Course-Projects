# -*- coding: utf-8 -*-
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import wandb
import os
from typing import Tuple, List, Dict, Optional
import json
import time # Added for potential delays if needed

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
    # Ensure reproducibility for certain operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- Neural Network Models ---
class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, dueling_type: str = "avg"):
        super(DuelingDQN, self).__init__()
        self.dueling_type = dueling_type
        # Shared feature layer
        self.feature_layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        # State value stream
        self.value_stream = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        # Action advantage stream
        self.advantage_stream = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine value and advantage streams based on dueling type
        if self.dueling_type == "avg": # Type-1
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else: # Type-2 (max)
            q_values = value + advantage - advantage.max(dim=1, keepdim=True)[0]
        return q_values

# --- Replay Buffer ---
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
        self.Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Save an experience."""
        self.memory.append(self.Experience(np.array(state), action, reward, np.array(next_state), done))

    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Randomly sample a batch of experiences from memory."""
        if len(self.memory) < batch_size:
            return None # Not enough memory yet
        experiences = random.sample(self.memory, batch_size)
        # Convert batch of experiences to tensors
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(DEVICE)
        actions = torch.LongTensor(np.array([e.action for e in experiences])).to(DEVICE)
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).to(DEVICE)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(DEVICE)
        dones = torch.FloatTensor(np.array([e.done for e in experiences])).to(DEVICE)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)

# --- Dueling DQN Agent ---
class DuelingDQNAgent:
    """Agent implementing the Dueling DQN algorithm."""
    def __init__(self, state_dim: int, action_dim: int, dueling_type: str = "avg", lr: float = 1e-3, gamma: float = GAMMA,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
                 buffer_size: int = 10000, batch_size: int = 64, target_update: int = 10, hidden_dim: int = 128):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # Initialize policy and target networks
        self.policy_net = DuelingDQN(state_dim, action_dim, hidden_dim, dueling_type).to(DEVICE)
        self.target_net = DuelingDQN(state_dim, action_dim, hidden_dim, dueling_type).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only for inference

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.update_count = 0

    def select_action(self, state: np.ndarray) -> int:
        """Selects an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim) # Explore
        else: # Exploit
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item() # Return action with highest Q-value

    def update_epsilon(self):
        """Decays epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def learn(self) -> float:
        """Update policy network using a batch from replay buffer."""
        sample = self.memory.sample(self.batch_size)
        if sample is None:
            return 0.0 # Return 0 loss if buffer not full enough

        states, actions, rewards, next_states, dones = sample

        # Get Q-values for current states from policy network
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Get max Q-values for next states from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)

        # Compute target Q-values: R + gamma * max_a' Q_target(s', a')
        target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (1 - dones.unsqueeze(1)))

        # Compute loss (Smooth L1 loss is common for DQN)
        loss = F.smooth_l1_loss(current_q_values, target_q_values) # Target is implicitly detached

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

# --- Training Functions ---

def run_single_episode(env, agent, max_steps=1000) -> Tuple[float, float]:
    """Runs a single episode, trains the agent, and returns total reward and average loss."""
    state, _ = env.reset()
    episode_reward = 0.0
    total_loss = 0.0
    steps = 0
    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.memory.push(state, action, reward, next_state, done)
        loss = agent.learn() # Learn at each step
        total_loss += loss
        steps += 1
        state = next_state
        episode_reward += reward
        if done:
            break
    agent.update_epsilon() # Update epsilon at the end of the episode
    avg_loss = total_loss / steps if steps > 0 else 0.0
    return episode_reward, avg_loss

def train_dueling_dqn_run(is_sweep_run: bool = False):
    """
    Performs a single training run (either for sweep or final evaluation).
    Logs metrics to wandb.
    Returns:
        - final_avg_reward_last_20 (float) if is_sweep_run is True.
        - episode_rewards (List[float]) if is_sweep_run is False.
    """
    # Initialize run (will inherit config from sweep agent if applicable)
    # Use reinit=True for final runs called within the same script execution
    run = wandb.init(reinit=True)
    config = wandb.config

    env = gym.make(config.env_name)
    # Important: Use the specific seed from the config for this run
    env.reset(seed=config.seed)
    set_seeds(config.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DuelingDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        dueling_type=config.dueling_type,
        lr=config.learning_rate,
        gamma=GAMMA, # Use fixed gamma
        epsilon_decay=config.epsilon_decay,
        batch_size=config.batch_size,
        target_update=config.target_update,
        hidden_dim=config.hidden_dim
    )

    episode_rewards = []
    print_interval = max(10, config.num_episodes // 20) # Print progress periodically

    for episode in range(config.num_episodes):
        episode_reward, avg_loss = run_single_episode(env, agent)
        episode_rewards.append(episode_reward)

        # Log to wandb
        avg_reward_last_100 = np.mean(episode_rewards[-100:]) if episode >= 99 else np.mean(episode_rewards)
        wandb.log({
            "episode": episode + 1,
            "episode_reward": episode_reward,
            "epsilon": agent.epsilon,
            "avg_episode_loss": avg_loss,
            "avg_reward_last_100": avg_reward_last_100
        })

        if (episode + 1) % print_interval == 0:
             run_name = run.name if run else "sweep_run" # Handle potential None run object
             print(f"Run: {run_name}, Env: {config.env_name}, Type: {config.dueling_type.upper()}, Seed: {config.seed}, Ep: {episode+1}/{config.num_episodes}, Reward: {episode_reward:.2f}, AvgR(100): {avg_reward_last_100:.2f}, Eps: {agent.epsilon:.3f}")

    # --- Final Logging ---
    final_avg_reward_last_20 = np.mean(episode_rewards[-20:]) # Metric for sweep optimization
    wandb.log({"final_avg_reward_last_20": final_avg_reward_last_20})
    # Ensure summary is updated even if called via wandb.agent
    if wandb.run:
        wandb.run.summary["final_avg_reward_all_eps"] = np.mean(episode_rewards)
        wandb.run.summary["final_avg_reward_last_100"] = np.mean(episode_rewards[-100:])
        wandb.run.summary["algorithm_variant"] = config.algorithm_variant # Log variant

    env.close()

    if is_sweep_run:
        return final_avg_reward_last_20 # Return metric for sweep agent
    else:
        return episode_rewards # Return full history for final plotting

# --- Sweep Configuration and Execution ---

def setup_sweep_config(env_name: str, dueling_type: str) -> Dict:
    """Creates the sweep configuration dictionary."""
    # Clearer Sweep Name inside the config
    sweep_name = f"SWEEP_{dueling_type.upper()}_DuelingDQN_{env_name}"
    algorithm_variant = f"DuelingDQN_{dueling_type.upper()}"

    sweep_config = {
        "name": sweep_name, # Sweep name inside config
        "method": "bayes",
        "metric": {"name": "final_avg_reward_last_20", "goal": "maximize"}, # Target last 20 eps avg
        "parameters": {
            "env_name": {"value": env_name},
            "dueling_type": {"value": dueling_type},
            "algorithm_variant": {"value": algorithm_variant}, # Add variant info
            "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 5e-3}, # Adjusted hyperparameter ranges
            "epsilon_decay": {"distribution": "uniform", "min": 0.985, "max": 0.999},
            "batch_size": {"values": [32, 64, 128, 256]},
            "target_update": {"values": [5, 10, 20, 50]},
            "hidden_dim": {"values": [64, 128, 256, 512]},
            "num_episodes": {"value": NUM_EPISODES_SWEEP},
            "seed": {"value": SWEEP_SEED} # Fixed seed for tuning
        }
    }
    return sweep_config

def run_sweep(env_name: str, dueling_type: str, sweep_count: int = SWEEP_COUNT) -> Optional[str]:
    """Initializes and runs a wandb sweep. Returns sweep_id or None on failure."""
    # Project specific to the environment
    project_name = f"RL_DuelingDQN_{env_name.replace('-', '_')}"
    sweep_config = setup_sweep_config(env_name, dueling_type)
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"--- Starting Sweep {sweep_id} for {env_name} ({dueling_type.upper()}) ---")
    # Use a lambda to pass the is_sweep_run=True argument
    wandb.agent(sweep_id, function=lambda: train_dueling_dqn_run(is_sweep_run=True), count=sweep_count)
    print(f"--- Finished Sweep {sweep_id} ---")
    return sweep_id

def get_best_hyperparams(sweep_id: str, project_name: str, wandb_entity: str) -> Optional[Dict]:
    """Retrieves the best hyperparameters from a completed sweep."""
    try:
        # Short delay to allow sweep data to sync
        time.sleep(5)
        api = wandb.Api(timeout=19) # Increased timeout
        sweep_path = f"{wandb_entity}/{project_name}/{sweep_id}"
        print(f"Fetching best run from sweep: {sweep_path}")
        sweep = api.sweep(sweep_path)
        # Order by the metric defined in the sweep config
        # Filter runs by state 'finished' before getting best_run
        runs = [run for run in sweep.runs if run.state == "finished"]
        if not runs:
             print(f"Warning: No finished runs found for sweep {sweep_id}. Using default parameters.")
             return None

        # Sort runs manually based on the summary metric
        best_run = max(runs, key=lambda run: run.summary.get("final_avg_reward_last_20", float('-inf')))

        if best_run:
            print(f"Best run found: {best_run.name} with score {best_run.summary.get('final_avg_reward_last_20', 'N/A')}")
            # Extract only the tunable parameters
            params = {
                "learning_rate": best_run.config["learning_rate"],
                "epsilon_decay": best_run.config["epsilon_decay"],
                "batch_size": best_run.config["batch_size"],
                "target_update": best_run.config["target_update"],
                "hidden_dim": best_run.config["hidden_dim"]
            }
            return params
        else:
            print(f"Warning: Could not determine best run for sweep {sweep_id}. Using default parameters.")
            return None
    except Exception as e:
        print(f"Error retrieving best hyperparameters for sweep {sweep_id}: {e}")
        print("Using default parameters.")
        return None

# --- Final Experiment Execution ---

def run_final_experiments(env_name: str, dueling_type: str, best_params: Optional[Dict],
                          num_episodes: int = NUM_EPISODES_FINAL, num_seeds: int = NUM_SEEDS_FINAL) -> List[List[float]]:
    """Runs the final experiments using the best hyperparameters across multiple seeds."""
    all_seed_rewards = []
    project_name = f"RL_DuelingDQN_{env_name.replace('-', '_')}" # Project per environment
    algorithm_variant = f"DuelingDQN_{dueling_type.upper()}"

    # Use default params if best_params is None
    if best_params is None:
         print(f"Using default parameters for final runs ({env_name}, {dueling_type}).")
         best_params = {
             "learning_rate": 1e-3, "epsilon_decay": 0.995, "batch_size": 64,
             "target_update": 10, "hidden_dim": 128
         }

    for seed in range(num_seeds):
        print(f"--- Running Final: {algorithm_variant} on {env_name} with Seed {seed} ---")

        config = {
            "env_name": env_name,
            "dueling_type": dueling_type,
            "algorithm_variant": algorithm_variant, # Log variant
            "num_episodes": num_episodes,
            "seed": seed, # Use current seed
            **best_params # Unpack best hyperparameters
        }

        # Manually create wandb run for final evaluation
        # Pass entity explicitly here as well
        run = wandb.init(
            project=project_name,
            name=f"FINAL_{algorithm_variant}_seed{seed}", # Clear run name
            config=config,
            group=f"{dueling_type.upper()}_FinalRuns", # Group final runs by type
            tags=["final_run", dueling_type, env_name], # Add tags
            job_type="final_evaluation",
            reinit=True # Allows multiple wandb.init calls
        )
        with run:
            # Call the training function, ensuring it returns the full list
            episode_rewards = train_dueling_dqn_run(is_sweep_run=False)
            all_seed_rewards.append(episode_rewards)

    print(f"--- Finished Final Runs for {algorithm_variant} on {env_name} ---")
    return all_seed_rewards

# --- Plotting ---

def plot_comparison(env_name: str, type1_rewards: List[List[float]], type2_rewards: List[List[float]]):
    """Plots the comparison between variants and logs to wandb."""
    # Check if lists contain valid reward data
    valid_type1 = [r for r in type1_rewards if r and isinstance(r, list)]
    valid_type2 = [r for r in type2_rewards if r and isinstance(r, list)]

    if not valid_type1 or not valid_type2:
        print(f"Warning: Missing valid reward data for {env_name}. Skipping comparison plot.")
        return

    # Pad shorter reward lists if necessary
    max_len = 0
    valid_reward_lists = valid_type1 + valid_type2
    if not valid_reward_lists:
         print(f"Warning: All reward lists empty for {env_name}. Skipping plot.")
         return
    max_len = max(len(r) for r in valid_reward_lists)


    # Pad with the last value, handle empty lists by padding with NaN
    padded_type1 = [r + [r[-1]] * (max_len - len(r)) if r else [np.nan] * max_len for r in type1_rewards]
    padded_type2 = [r + [r[-1]] * (max_len - len(r)) if r else [np.nan] * max_len for r in type2_rewards]

    # Use nanmean/nanstd to ignore potential NaNs from padding empty/failed runs
    type1_mean = np.nanmean(padded_type1, axis=0)
    type1_std = np.nanstd(padded_type1, axis=0)

    type2_mean = np.nanmean(padded_type2, axis=0)
    type2_std = np.nanstd(padded_type2, axis=0)

    episodes = range(1, max_len + 1)

    plt.figure(figsize=(12, 7))

    # Plot Type 1 (AVG - Red)
    plt.plot(episodes, type1_mean, color='red', label='Type 1 (AVG Advantage)')
    plt.fill_between(episodes, type1_mean - type1_std, type1_mean + type1_std, color='red', alpha=0.2)

    # Plot Type 2 (MAX - Blue)
    plt.plot(episodes, type2_mean, color='blue', label='Type 2 (MAX Advantage)')
    plt.fill_between(episodes, type2_mean - type2_std, type2_mean + type2_std, color='blue', alpha=0.2)

    plt.xlabel('Episode Number', fontsize=12)
    plt.ylabel('Episodic Return', fontsize=12)
    plt.title(f'Dueling-DQN Comparison on {env_name}', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    # Save locally
    plot_filename = f'dueling_dqn_{env_name.lower()}_comparison.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved comparison plot to {plot_filename}")

    # Log comparison plot and data to wandb
    project_name = f"RL_DuelingDQN_{env_name.replace('-', '_')}" # Project per environment
    # Use a separate run for the final comparison plot, specifying entity
    run = wandb.init(project=project_name, name=f"ComparisonPlot_{env_name}", job_type="comparison_plot", reinit=True)
    with run:
        wandb.log({"comparison_plot": wandb.Image(plt)})

        # Log data for interactive plots
        log_data = []
        for i, episode in enumerate(episodes):
             # Check for NaN before logging, log 0 if NaN (or handle as needed)
             log_data.append({
                 "episode": episode,
                 "type1_mean": type1_mean[i] if not np.isnan(type1_mean[i]) else 0,
                 "type1_std": type1_std[i] if not np.isnan(type1_std[i]) else 0,
                 "type2_mean": type2_mean[i] if not np.isnan(type2_mean[i]) else 0,
                 "type2_std": type2_std[i] if not np.isnan(type2_std[i]) else 0
             })
        wandb.log({"comparison_data": wandb.Table(data=log_data, columns=["episode", "type1_mean", "type1_std", "type2_mean", "type2_std"])})

    plt.close() # Close plot figure

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Check WANDB_ENTITY ---
    WANDB_ENTITY = "your-wandb-username" # Replace with your wandb username or entity name

    wandb.login()

    # --- Configuration ---
    environments = ["CartPole-v1", "Acrobot-v1"]
    dueling_types = ["avg", "max"] # Type-1, Type-2

    all_best_params = {} # Dictionary to store best params

    # --- Hyperparameter Tuning Phase ---
    print("\n=== STARTING HYPERPARAMETER TUNING ===")
    for env_name in environments:
        all_best_params[env_name] = {}
        for dueling_type in dueling_types:
            try:
                sweep_id = run_sweep(env_name, dueling_type, sweep_count=SWEEP_COUNT)
                if sweep_id: # Proceed only if sweep started successfully
                    project_name = f"RL_DuelingDQN_{env_name.replace('-', '_')}"
                    all_best_params[env_name][dueling_type] = get_best_hyperparams(sweep_id, project_name)
                else:
                    all_best_params[env_name][dueling_type] = None # Mark as failed
            except Exception as e:
                 print(f"ERROR during sweep/param retrieval for {env_name} ({dueling_type}): {e}")
                 all_best_params[env_name][dueling_type] = None # Mark as failed

    # Save best parameters
    params_filename = 'dueling_dqn_best_params.json'
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
        for dueling_type in dueling_types:
            # Load best params, use defaults if None or tuning failed
            best_params = all_best_params.get(env_name, {}).get(dueling_type, None)

            try:
                final_rewards = run_final_experiments(
                    env_name=env_name,
                    dueling_type=dueling_type,
                    best_params=best_params, # Pass potentially None params
                    num_episodes=NUM_EPISODES_FINAL,
                    num_seeds=NUM_SEEDS_FINAL
                )
                all_final_rewards[env_name][dueling_type] = final_rewards

                # Save raw reward data
                rewards_filename = f'dueling_dqn_type{"1" if dueling_type=="avg" else "2"}_{env_name.lower()}.npy'
                try:
                    # Ensure data is a numpy array before saving
                    # Pad inner lists to the max length found across all seeds for this variant/env
                    max_len_final = 0
                    if final_rewards:
                        max_len_final = max(len(r) for r in final_rewards if r)

                    if max_len_final > 0:
                        padded_rewards = [r + [r[-1]] * (max_len_final - len(r)) if r else [np.nan] * max_len_final for r in final_rewards]
                        np_rewards = np.array(padded_rewards)
                        np.save(rewards_filename, np_rewards)
                        print(f"Saved final rewards to {rewards_filename}")
                    else:
                        print(f"Skipping save for {rewards_filename} as no reward data was generated.")

                except Exception as e_save:
                    print(f"Error saving final rewards for {rewards_filename}: {e_save}")

            except Exception as e_run:
                 print(f"ERROR during final run for {env_name} ({dueling_type}): {e_run}")
                 all_final_rewards[env_name][dueling_type] = [] # Store empty list if run failed


    # --- Comparison Plotting Phase ---
    print("\n=== GENERATING COMPARISON PLOTS ===")
    for env_name in environments:
        rewards_type1 = all_final_rewards.get(env_name, {}).get("avg", [])
        rewards_type2 = all_final_rewards.get(env_name, {}).get("max", [])

        plot_comparison(env_name, rewards_type1, rewards_type2)

    print("\n=== All Dueling-DQN experiments completed! Check wandb projects. ===")
    # Explicitly finish the last wandb run if any is active
    if wandb.run is not None:
        wandb.finish()

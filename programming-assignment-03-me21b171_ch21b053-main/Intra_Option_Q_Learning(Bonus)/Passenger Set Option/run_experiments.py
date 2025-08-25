import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from smdp_q_learning import SMDPQLearner, PassengerOption
from intra_option_q_learning import IntraOptionQLearner
from tqdm import tqdm

# Define constants
NUM_EPISODES = 1000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

# Define visualization functions
def plot_rewards(rewards, title="Reward Plot"):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def plot_q_values(Q, title="Q-values"):
    plt.figure(figsize=(12, 10))
    plt.imshow(Q, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def compare_methods(smdp_rewards, intra_rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(smdp_rewards, label="SMDP Q-Learning")
    plt.plot(intra_rewards, label="Intra-Option Q-Learning")
    plt.title("Reward Comparison: SMDP vs Intra-Option Q-Learning for Passenger Option Set")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("reward_comparison.png")
    plt.show()

# Main function
def main():
    # Create Taxi-v3 environment
    env = gym.make('Taxi-v3', render_mode="ansi")
    
    # Create passenger option
    passenger_option = PassengerOption(env)
    options = [passenger_option]
    
    print("Running SMDP Q-Learning...")
    smdp_agent = SMDPQLearner(env, options, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
    smdp_rewards = smdp_agent.train(NUM_EPISODES)
    plot_rewards(smdp_rewards, "SMDP Q-Learning: Reward Plot for Passenger Option Set")
    
    print("Running Intra-Option Q-Learning...")
    intra_agent = IntraOptionQLearner(env, options, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
    intra_rewards = intra_agent.train(NUM_EPISODES)
    plot_rewards(intra_rewards, "Intra-Option Q-Learning: Rewards Plot for Passenger Option Set")
    
    # Compare the reward curves for both algorithms
    compare_methods(smdp_rewards, intra_rewards)

if __name__ == "__main__":
    main()

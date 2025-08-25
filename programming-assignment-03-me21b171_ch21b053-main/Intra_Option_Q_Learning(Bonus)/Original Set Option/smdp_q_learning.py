# smdp_q_learning.py

import numpy as np
import random
import gym
from tqdm import tqdm

# Define constants
NUM_ACTIONS = 6  # Number of possible actions in Taxi-v3

def smdp_q_learning(env, num_episodes, alpha, gamma, epsilon):
    # Initialize Q-values table (state x action)
    Q = np.zeros((env.observation_space.n, NUM_ACTIONS))  # (state x action)
    rewards = []

    for episode in tqdm(range(num_episodes), desc="Running SMDP Q-Learning", unit="episode"):
        state = env.reset()  # This could return a tuple (state, info), we need the first element
        state = state[0] if isinstance(state, tuple) else state  # Ensure state is an integer
        total_reward = 0
        done = False

        while not done:
            # Choose action based on epsilon-greedy strategy
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                action = np.argmax(Q[state])  # Choose best action based on Q-values

            # Take the action and observe the next state and reward
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            # Update Q-values
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

            # Transition to the next state
            state = next_state

        rewards.append(total_reward)

    return Q, rewards

# Optional: functions for plotting rewards and Q-values
def plot_rewards(rewards, title="SMDP Reward Plot for Original Option Set"):
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

def plot_q_values(Q, title="Q-values"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.heatmap(Q, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title(title)
    plt.show()

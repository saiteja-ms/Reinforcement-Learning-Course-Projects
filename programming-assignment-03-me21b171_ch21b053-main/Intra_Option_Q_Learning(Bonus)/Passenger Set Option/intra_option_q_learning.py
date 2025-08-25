import numpy as np
import random
import gym
from tqdm import tqdm
from smdp_q_learning import PassengerOption

class IntraOptionQLearner:
    def __init__(self, env, options, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.options = options
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.num_primitives = 6  # Number of primitive actions in Taxi environment
        self.num_options = len(options)
        self.Q = np.zeros((500, self.num_primitives + self.num_options))  # Initialize Q-table
        
    def get_available(self, state):
        available = list(range(self.num_primitives))
        taxi_row, taxi_col, _, _ = self.env.unwrapped.decode(state)
        
        for i, opt in enumerate(self.options):
            if hasattr(opt, 'is_available') and opt.is_available(state):
                available.append(self.num_primitives + i)
        
        return available

    def choose_action(self, state, available):
        if np.random.random() < self.epsilon:
            return np.random.choice(available)
        else:
            return available[np.argmax(self.Q[state, available])]
    
    def intra_option_update(self, state, action, next_state, reward):
        # Update Q-value for the primitive action
        future_max = np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (reward + self.gamma * future_max - self.Q[state, action])
        
        # Intra-option updates
        for i, option in enumerate(self.options):
            # Check if option would have selected this action
            option_action = option.get_action(state)
            if option_action == action:
                option_idx = self.num_primitives + i
                
                # Check if option would terminate in next state
                if option.is_terminated(next_state):
                    target = reward + self.gamma * np.max(self.Q[next_state])
                else:
                    # Continue with option
                    target = reward + self.gamma * self.Q[next_state, option_idx]
                
                # Update option's Q-value
                self.Q[state, option_idx] += self.alpha * (target - self.Q[state, option_idx])
    
    def train(self, episodes):
        rewards = []
        for ep in tqdm(range(episodes), desc="Intra-Option Q-Learning"):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get available actions for the current state
                available = self.get_available(state)
                action_idx = self.choose_action(state, available)
                
                if action_idx < self.num_primitives:
                    # Execute primitive action
                    action = action_idx
                else:
                    # Get action from option policy
                    option = self.options[action_idx - self.num_primitives]
                    action = option.get_action(state)
                
                # Execute the action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Perform intra-option updates
                self.intra_option_update(state, action, next_state, reward)
                
                state = next_state
            
            rewards.append(episode_reward)
            if (ep+1) % 100 == 0:
                print(f"Episode {ep+1}, Avg Reward: {np.mean(rewards[-100:]):.2f}")
        
        return rewards

import numpy as np
import random
import gym
from tqdm import tqdm

# Define passenger option
class PassengerOption:
    def __init__(self, env):
        self.env = env
        
    def get_action(self, state):
        taxi_row, taxi_col, pass_loc, dest_idx = self.env.unwrapped.decode(state)
        
        # Determine target based on passenger location
        if pass_loc == 4:  # Passenger in taxi
            target = self.env.unwrapped.locs[dest_idx]
        else:
            # Check for pickup
            if (taxi_row, taxi_col) == self.env.unwrapped.locs[pass_loc]:
                return 4  # PICKUP
            target = self.env.unwrapped.locs[pass_loc]
            
        # Navigation logic
        if taxi_row > target[0]:
            return 1  # North
        elif taxi_row < target[0]:
            return 0  # South
        elif taxi_col > target[1]:
            return 3  # West
        elif taxi_col < target[1]:
            return 2  # East
        
        # We're at the destination with passenger, drop off
        if pass_loc == 4 and (taxi_row, taxi_col) == self.env.unwrapped.locs[dest_idx]:
            return 5  # DROPOFF
            
        return 0  # Default
    
    def is_terminated(self, state):
        _, _, pass_loc, dest_idx = self.env.unwrapped.decode(state)
        return pass_loc == dest_idx  # Passenger at destination
    
    def is_available(self, state):
        # Passenger option is always available
        return True

class SMDPQLearner:
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
    
    def execute_option(self, action_idx, state):
        if action_idx < self.num_primitives:
            # Primitive action
            next_state, reward, terminated, truncated, _ = self.env.step(action_idx)
            return reward, next_state, 1, terminated or truncated
        else:
            # Option
            option = self.options[action_idx - self.num_primitives]
            total_reward = 0
            discount = 1.0
            steps = 0
            current_state = state
            done = False
            
            while True:
                if option.is_terminated(current_state):
                    break
                    
                action = option.get_action(current_state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                total_reward += discount * reward
                discount *= self.gamma
                steps += 1
                current_state = next_state
                
                if done:
                    break
            
            return total_reward, current_state, steps, done
    
    def update(self, state, action_idx, reward, next_state, steps):
        available = self.get_available(next_state)
        if len(available) > 0:
            future_max = np.max(self.Q[next_state, available])
            target = reward + (self.gamma ** steps) * future_max
            self.Q[state, action_idx] += self.alpha * (target - self.Q[state, action_idx])
    
    def train(self, episodes):
        rewards = []
        for ep in tqdm(range(episodes), desc="SMDP Q-Learning"):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get available actions for the current state
                available = self.get_available(state)
                action_idx = self.choose_action(state, available)
                
                # Execute the chosen action or option
                reward, next_state, steps, done = self.execute_option(action_idx, state)
                self.update(state, action_idx, reward, next_state, steps)
                
                episode_reward += reward
                state = next_state
            
            rewards.append(episode_reward)
            if (ep+1) % 100 == 0:
                print(f"Episode {ep+1}, Avg Reward: {np.mean(rewards[-100:]):.2f}")
        
        return rewards

import numpy as np

class SMDPQLearner:
    def __init__(self, env, options, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.options = options
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.num_primitives = 6 # Number of primitive actions in the Taxi environment
        self.num_options = len(options)
        self.Q = np.zeros((500, self.num_primitives + self.num_options)) # Initialize Q-table
        
    def get_available(self, state):
        available = list(range(self.num_primitives))
        taxi_row, taxi_col, _, _ = self.env.unwrapped.decode(state)
        
        for i, opt in enumerate(self.options):
            # Handle different option types
            if hasattr(opt, 'target_pos'):  # Original navigation options
                if (taxi_row, taxi_col) != opt.target_pos:
                    available.append(self.num_primitives + i)
            elif hasattr(opt, 'is_available'):  # Passenger-focused option
                if opt.is_available(state):
                    available.append(self.num_primitives + i)
        
        return available

    def choose_action(self, state, available):
        if np.random.random() < self.epsilon:
            return np.random.choice(available)
        else:
            return available[np.argmax(self.Q[state, available])]
    
    def execute_option(self, action_idx, state):
        if action_idx < self.num_primitives:
            # Primitive action handling remains the same
            next_state, reward, terminated, truncated, _ = self.env.step(action_idx)
            return reward, next_state, 1, terminated or truncated
        else:
            # Option execution (modified termination check)
            option = self.options[action_idx - self.num_primitives]
            total_reward = 0
            discount = 1.0
            steps = 0
            current_state = state
            terminated = False
            
            while True:
                # Check termination using option's method
                if option.is_terminated(current_state):
                    break
                    
                # Get action and execute step
                action = option.get_action(current_state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # Update accumulated values
                total_reward += discount * reward
                discount *= self.gamma
                steps += 1
                current_state = next_state
                
                if terminated or truncated:
                    break
            
            return total_reward, current_state, steps, terminated or truncated

    
    def update(self, state, action_idx, reward, next_state, steps):
        future_max = np.max(self.Q[next_state, self.get_available(next_state)]) 
        target = reward + (self.gamma ** steps) * future_max
        self.Q[state, action_idx] += self.alpha * (target - self.Q[state, action_idx])
    
    def train(self, episodes):
        rewards = []
        for ep in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get available actions for the current state
                available = self.get_available(state)
                action_idx = self.choose_action(state, available)
                
                # Execute the chosen action (primitive or option)
                reward, next_state, steps, done = self.execute_option(action_idx, state)
                self.update(state, action_idx, reward, next_state, steps)
                
                episode_reward += reward
                state = next_state
            
            rewards.append(episode_reward)
            if (ep+1) % 100 == 0: # Print every 100 episodes
                print(f"Episode {ep+1}, Avg Reward: {np.mean(rewards[-100:])}")
        
        return rewards

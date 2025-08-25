import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        # Neural network mapping states to action probabilities
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # Map state to hidden representation
            nn.ReLU(),                         # Non-linear activation
            nn.Linear(hidden_dim, hidden_dim), # Additional hidden layer
            nn.ReLU(),                         # Non-linear activation
            nn.Linear(hidden_dim, action_dim), # Map hidden representation to action scores
            nn.Softmax(dim=-1)                 # Convert scores to probabilities
        )

    def forward(self, state):
        # Forward pass that returns a categorical distribution over actions
        action_probs = self.policy(state)
        return Categorical(action_probs)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        # Value network to estimate state values
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # Input mapping
            nn.ReLU(),                         # Non-linear activation
            nn.Linear(hidden_dim, hidden_dim), # Hidden layer
            nn.ReLU(),                         # Non-linear activation
            nn.Linear(hidden_dim, 1)           # Output a single state value
        )

    def forward(self, state):
        # Forward pass for state value estimation
        return self.value(state)

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, use_baseline=False,
                 lr_policy=1e-3, lr_value=1e-3, gamma=0.99,
                 hidden_dim=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_baseline = use_baseline
        self.gamma = gamma

        # Create the policy network and its optimizer
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        
        # Create the value network and its optimizer if baseline is used
        if use_baseline:
            self.value_net = ValueNetwork(state_dim, hidden_dim)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)

        # Move networks to the appropriate device (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net.to(self.device)
        if use_baseline:
            self.value_net.to(self.device)

    def select_action(self, state):
        # Convert state to tensor and compute action distribution
        state = torch.FloatTensor(state).to(self.device)
        dist = self.policy_net(state)
        action = dist.sample()  # Sample an action from the distribution
        return action.item(), dist.log_prob(action)  # Return action index and its log probability

    def calculate_returns(self, rewards):
        # Calculate discounted returns for the episode
        returns = []
        R = 0
        # Process rewards in reverse for cumulative discounting
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, device=self.device)

    def _update_baseline_td0(self, state, reward, next_state, done):
        """
        Update value network using TD(0) at each step.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        # Compute the next state value; if done, there is no next value.
        with torch.no_grad():
            next_value = 0 if done else self.value_net(next_state_tensor)
        
        reward_tensor = torch.tensor([reward], device=self.device, dtype=torch.float32)

        # TD target: immediate reward plus discounted next state value
        td_target = reward_tensor + self.gamma * next_value
        current_value = self.value_net(state_tensor)
        value_loss = F.mse_loss(current_value, td_target)
        
        # Optimize the value network parameters
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def _update_policy(self, states, rewards, log_probs):
        """
        Update policy network via policy gradient at the end of each episode.
        """
        returns = self.calculate_returns(rewards)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        policy_loss = 0

        # Compute advantages using the baseline if available
        if self.use_baseline:
            with torch.no_grad():
                state_values = self.value_net(states).squeeze(-1)
            advantages = returns - state_values
        else:
            advantages = returns

        # Normalize advantages for stability when possible
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute policy loss: negative log probability weighted by advantage
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss -= log_prob * advantage

        # Update the policy network parameters
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def train(self, env, num_episodes, max_steps=500):
        # List to track total returns per episode
        episode_returns = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()  # Reset environment for new episode
            states, rewards, log_probs = [], [], []
            episode_return = 0

            for step in range(max_steps):
                # Select an action using current policy
                action, log_prob = self.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Record current state, reward, and log probability
                states.append(state)
                rewards.append(reward)
                log_probs.append(log_prob)
                
                # If using baseline, update value network incrementally using TD(0)
                if self.use_baseline:
                    self._update_baseline_td0(state, reward, next_state, done or truncated)
                
                # Update state and accumulate reward
                state = next_state
                episode_return += reward
                
                if done or truncated:
                    break

            # After the episode, update policy using Monte Carlo returns
            self._update_policy(states, rewards, log_probs)
            
            episode_returns.append(episode_return)
            print(f"Episode {episode+1}, Return: {episode_return}")

        return episode_returns

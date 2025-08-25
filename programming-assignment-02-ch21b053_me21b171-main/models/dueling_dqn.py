import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

class ReplayBuffer:
    """Experience replay buffer to store and sample transitions."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

class DuelingDQN(nn.Module):
    """
    Dueling DQN Network architecture separating state value and action advantage streams.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        
        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream (estimates state value V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # Advantage stream (estimates advantages A(s,a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, action_dim)
        )
        
    def forward(self, state, aggregation_type=1):
        """
        Forward pass implementing either Type-1 or Type-2 Q-value aggregation.
        
        Args:
            state: Input state tensor
            aggregation_type: 1 for mean aggregation, 2 for max aggregation
        """
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        if aggregation_type == 1:
            # Type-1: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
            return value + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            # Type-2: Q(s,a) = V(s) + (A(s,a) - max(A(s,a')))
            return value + (advantages - advantages.max(dim=1, keepdim=True)[0])

class DuelingDQNAgent:
    """Agent implementing Dueling DQN with both Type-1 and Type-2 update rules."""
    def __init__(self, state_dim, action_dim, aggregation_type=1, 
                 lr=0.001, gamma=0.99, epsilon_start=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995, 
                 buffer_size=10000, batch_size=64, 
                 target_update=10, hidden_dim=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.aggregation_type = aggregation_type
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_count = 0
        
        # Initialize networks
        self.policy_net = DuelingDQN(state_dim, action_dim, hidden_dim)
        self.target_net = DuelingDQN(state_dim, action_dim, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and np.random.rand() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.action_dim)
        else:
            # Exploitation: greedy action
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state, self.aggregation_type)
            return q_values.argmax().item()
    
    def update(self):
        """Update the policy network using a batch of experiences."""
        if len(self.buffer) < self.batch_size:
            return 0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        q_values = self.policy_net(states, self.aggregation_type).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states, self.aggregation_type).max(1)[0]
        
        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        return loss.item()
    
    def train(self, env, num_episodes, max_steps=500):
        """Train the agent over multiple episodes."""
        episode_returns = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_return = 0
            
            for step in range(max_steps):
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Store transition in replay buffer
                self.buffer.push(state, action, reward, next_state, done)
                
                # Update state and episode return
                state = next_state
                episode_return += reward
                
                # Update the network
                loss = self.update()
                
                if done or truncated:
                    break
            
            episode_returns.append(episode_return)
            print(f"Episode {episode+1}, Return: {episode_return}, Epsilon: {self.epsilon:.4f}")
        
        return episode_returns

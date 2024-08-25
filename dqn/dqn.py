# This file stores all the code for the DQN algorithm, including the class definition for the DQN model and the replay memory buffer.

import torch
import torch.backends
import torch.nn as nn
import torch.nn.functional as F

import math
import random
import numpy as np

from collections import deque

# Buffer for storing transitions for experience replay
class ReplayMemory(object):
        
        def __init__(self, capacity):
            self.memory = deque([], maxlen=capacity)
        
        # Pushes a Transition object into the memory buffer,
        # where the Transition object is a named tuple representing a single transition in the environment. It maps (state, action) pairs to their (nextState, reward) result.
        def push(self, transition, *args):
            self.memory.append(transition(*args))
            
        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)
        
        def __len__(self):
            return len(self.memory)

# Deep Q-Network
class DQN(nn.Module):
    
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier Initialization
                nn.init.xavier_uniform_(m.weight)
                # If using Kaiming Initialization (for ReLU):
                # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# Epsilon-greedy policy
class EpsilonGreedyPolicy():
    # Initializes the epsilon-greedy policy with the given epsilon values.
    def __init__(self, epsilon_end, epsilon_start, epsilon_decay):
        self.epsilon_end = epsilon_end
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
    
    # Selects an action based on the epsilon-greedy policy.
    def select_action(self, step_index, policy_net, state, n_actions, device):
        sample = random.random()
        threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * step_index / self.epsilon_decay)
        
        if sample > threshold:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[np.random.randint(n_actions)]], device=device, dtype=torch.long)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque
import random

import environment
from DDPGNetworks import Actor, Critic

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Buffer for storing transitions for experience replay
class ReplayMemory(object):
        
        def __init__(self, capacity):
            self.memory = deque([], maxlen=capacity)
            
        def push(self, *args):
            self.memory.append(transition(*args))
            
        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)
        
        def __len__(self):
            return len(self.memory)

class DDPG():
    def __init__(self, n_observations, n_actions, max_action):
        self.actor = Actor(n_observations, n_actions, max_action)
        self.actor_target = Actor(n_observations, n_actions, max_action)
        
        self.critic = Critic(n_observations, n_actions)
        self.critic_target = Critic(n_observations, n_actions)
        
        # Hyperparameters
        self.BATCH_SIZE = 64 # Number of samples to take from the replay memory.
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005 # Update rate of the target networks.
        self.LEARNING_RATE = 1e-4 # Learning rate of the optimizer.
        
        self.n_actions = n_actions
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.LEARNING_RATE)
        
        self.memory = ReplayMemory(10000)
        
        
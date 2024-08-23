import torch
import torch.backends
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import DQN.environmentV2 as environmentV2, simulation.importer as importer, qPolicies
from tf_agents.trajectories import time_step as ts

# Set up matplotlib
is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

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
        
# Deep Q-Network
class DQN(nn.Module):
    
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Soft update function
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
                
# Segment the data temps, in hours, into 6-minute inverval.
def SegmentHourTempsInto6MinuteTemps(temps):
    # Segment the data temps, in hours, into 6-minute inverval.
    tempsMinutes = np.zeros(len(temps) * 10)
    for i in range(0, len(temps) - 1):
        for j in range(0, 10):
            # Lerp the temperature
            tempsMinutes[i * 10 + j] = temps[i] + (temps[i + 1] - temps[i]) * j / 10
    return tempsMinutes
    

# Hyperparameters
BATCH_SIZE = 128 # Number of transitions to sample from the replay memory
GAMMA = 0.99 # Discount factor
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005 # Update rate of the target network
LR = 1e-4 # Learning rate of the optimizer

# Setup the environment
house = environmentV2.House()
house.RWall = 5
house.RWindow = 1
house.Length = 30
house.Width = 10
house.Height = 4
house.RoofPitch = np.radians(40)
house.WindowWidth = 1
house.WindowHeight = 1
house.NumWindows = 6
heater = environmentV2.Heater()
heater.MaximumPower = 1200
heater.NumHeaters = 5
params = {
    "ThermalResistance": house.GetThermalResistance(),
    "AirMass": house.GetTotalAirMass(1.225),
    "AirHeatCapacity": 1005.4,
    "MaximumHeatingPower": heater.GetTotalMaximumPower(),
    "NumActions": heater.NumHeaters,
    "CostPerJoule": 1.1015/3.6e6,
    "Discount": 0.9,
    "Hysteresis": 2,
    "Dt": 360
}

print(params)
    
env = environmentV2.HouseEnvironmentV2(params, SegmentHourTempsInto6MinuteTemps(importer.ImportHuskvarna()))

state = env.reset()
n_actions = env.action_spec().maximum + 1
n_observations = len(state.observation)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[np.random.randint(n_actions)]], device=device, dtype=torch.long)

episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    
    # Transpose the batch. This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = transition(*zip(*transitions))
    
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()   
    
# Main training loop
    
minTempDifference = -2
maxTempDifference = 2
tempDifferenceStep = 0.25

num_episodes = 100
    
for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state = env.reset()
    
    # Calculate the initial state.
    tempDifferenceA = state.observation[0]
    tempDifferenceB = state.observation[1]

    num_steps = 0
    while not state.is_last() and abs(tempDifferenceA) <= maxTempDifference:
        
        state_tensor = torch.tensor(state.observation, dtype=torch.float32, device=device).unsqueeze(0)

        action = select_action(state_tensor)
        next_state = env.step(action.item())
        reward_tensor = torch.tensor(np.ones(1)*next_state.reward, device=device)
        
        if next_state.step_type == ts.StepType.LAST:
            next_state_tensor = None
        else:
            next_state_tensor = torch.tensor(next_state.observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Ensure action is a PyTorch tensor before storing it
        action_tensor = torch.tensor(action, device=device, dtype=torch.int64)
        
        # Store the transition in memory
        memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)
        
        #print(state_tensor, " : ", action_tensor)
        
        # Move to the next state
        state = next_state
        
        # Perform one step of the optimization (on the policy network)
        optimize_model()

        
        #Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        
        num_steps += 1
    
    episode_durations.append(num_steps)
    plot_durations()

torch.save(policy_net.state_dict(), 'policy_net.pth')

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
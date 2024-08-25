# Module for training the DQN model(s)

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

import environmentV2 as environmentV2, dqn
import sys
sys.path.append('simulation')
import importer

from tf_agents.trajectories import time_step as ts
from IPython import display

# Helper function for Segmenting the data temps, in hours, into 6-minute invervals.
def SegmentHourTempsInto6MinuteTemps(temps):
    # Segment the data temps, in hours, into 6-minute inverval.
    tempsMinutes = np.zeros(len(temps) * 10)
    for i in range(0, len(temps) - 1):
        for j in range(0, 10):
            # Lerp the temperature
            tempsMinutes[i * 10 + j] = temps[i] + (temps[i + 1] - temps[i]) * j / 10
    return tempsMinutes

# Helper function to plot the durations of the episodes.
def plot_durations(episode_durations, is_ipython, show_result=False):
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
            
# Helper function for optimizing the model.
def optimize_model(replay_buffer, batch_size, transition, optimizer, policy_net, target_net, gamma, device):
    
    # Early out if the replay buffer is not full enough.
    if len(replay_buffer) < batch_size:
        return
    
    transitions = replay_buffer.sample(batch_size)
    
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
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
# Train the DQN model.
def Train(name, params, outdoor_temperatures, num_episodes, batch_size, device, env, policy_net, target_net, optimizer):
    
    # Set the device to use for plotting using matplotlib.
    is_ipython = 'inline' in plt.get_backend()
    plt.ion()
    
    # Get the parameters.
    gamma = params['Gamma']
    epsilon_start = params['EpsilonStart']
    epsilon_end = params['EpsilonEnd']
    epsilon_decay = params['EpsilonDecay']
    tau = params['Tau']
    learning_rate = params['LearningRate']
    
    # Setup the transition named tuple.
    transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    
    state = env.reset()
    num_actions = params["NumActions"]
    num_observations = len(state.observation)
    
    # Setup the replay buffer and policy.
    replay_buffer = dqn.ReplayMemory(10000)
    policy = dqn.EpsilonGreedyPolicy(epsilon_end, epsilon_start, epsilon_decay)
    
    episode_durations = []
    
    # Training loop.
    for episode_index in range(num_episodes):
        
        # Reset the environment and get the initial state.
        state = env.reset()
        
        num_steps = 0
        while not state.is_last():
            
            state_tensor = torch.tensor(state.observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Select an action based on the epsilon-greedy policy.
            action = policy.select_action(episode_index, policy_net, state_tensor, num_actions, device)
            action_tensor = torch.tensor(action, device=device, dtype=torch.int64)
            
            # Step the environment.
            next_state = env.step(action.item())
            
            if next_state.step_type == ts.StepType.LAST:
                next_state_tensor = None
            else:
                next_state_tensor = torch.tensor(next_state.observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            reward_tensor = torch.tensor(np.ones(1)*next_state.reward, device=device)
            
            # Store the transition in the replay buffer based on the order of the named tuple.
            replay_buffer.push(transition, state_tensor, action_tensor, next_state_tensor, reward_tensor)

            # Perform one step of the optimization (on the policy network).
            optimize_model(replay_buffer, batch_size, transition, optimizer, policy_net, target_net, gamma, device)
            
            # Soft update of the target network's weights.
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            target_net.load_state_dict(target_net_state_dict)
            
            # Move to the next state.
            state = next_state
            num_steps += 1
        
        episode_durations.append(num_steps)
        plot_durations(episode_durations, is_ipython, False)
    
    plt.ioff()
    # Save the model.
    torch.save(policy_net.state_dict(), name)
    
    return episode_durations

# Load the data.
outdoor_temperatures = SegmentHourTempsInto6MinuteTemps(importer.ImportHuskvarna())

# Set the parameters.
r_wall = 5
r_window = 1
house_length = 30
house_width = 10
house_height = 4
roof_pitch = np.radians(40)
window_width = 1
window_height = 1
num_windows = 6
heater_max_power = 1200 * 5 # 5 heaters
total_window_area = environmentV2.calculate_total_window_area(window_width, window_height, num_windows)
total_wall_area = environmentV2.calculate_total_wall_area(house_width, house_height, house_length, roof_pitch, total_window_area)
params = {
    "ThermalResistance": environmentV2.calculate_thermal_resistance(r_wall, r_window, total_wall_area, total_window_area),
    "AirMass": environmentV2.calculate_total_air_mass(house_width, house_height, house_length, roof_pitch, 1.225),
    "AirHeatCapacity": 1005.4,
    "MaximumHeatingPower": heater_max_power,
    "NumActions": 5,
    "CostPerJoule": 1.1015/3.6e6,
    "Discount": 0.9,
    "Hysteresis": 2,
    "Dt": 360,
    "Gamma": 0.9,
    "EpsilonStart": 0.9,
    "EpsilonEnd": 0.05,
    "EpsilonDecay": 200,
    "Tau": 0.005,
    "LearningRate": 1e-4
}

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Setup the environment.
env = environmentV2.HouseEnvironmentV2(params, outdoor_temperatures, device)
state = env.reset()

num_actions = params["NumActions"]
num_observations = len(state.observation)

# Setup the nets and optimizer.
policy_net = dqn.DQN(num_observations, num_actions).to(device)
target_net = dqn.DQN(num_observations, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=params['LearningRate'], amsgrad=True)

durations = Train("dqn_model_b.pth", params, outdoor_temperatures, 300, 128, device, env, policy_net, target_net, optimizer)
is_ipython = 'inline' in plt.get_backend()
plot_durations(durations, is_ipython, True)

thresholds = []
for i in range(0, 500):
    threshold = params["EpsilonEnd"] + (params["EpsilonStart"] - params["EpsilonEnd"]) * math.exp(-1. * i / params["EpsilonDecay"])
    thresholds.append(threshold)
plt.figure()
plt.plot(thresholds)
plt.show()
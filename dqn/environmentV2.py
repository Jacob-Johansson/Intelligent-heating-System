from collections import defaultdict
from typing import Optional

import numpy as np
import tensorflow as tf
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils


import sys
sys.path.append('simulation')
import simulation

# Returns the total window area of the house
def calculate_total_window_area(window_width, window_height, num_windows):
    return num_windows*window_height*window_width

# Returns the total wall area of the house
def calculate_total_wall_area(house_width, house_height, house_length, roof_pitch, total_window_area):
    return 2*house_length*house_height + 2*house_width*house_height + 2*(1/np.cos(roof_pitch/2))*house_width*house_length + np.tan(roof_pitch)*house_width - total_window_area

# Returns the equivalent thermal resistance of the whole house
def calculate_thermal_resistance(r_wall, r_window, total_wall_area, total_window_area):
    tr_wall = r_wall / total_wall_area # K/W
    tr_window = r_window / total_window_area # K/W
    return tr_wall * tr_window / (tr_wall + tr_window)

# Returns the total air mass in the house, where airDensity is the density of air in kg/m^3
def calculate_total_air_mass(house_width, house_height, house_length, roof_pitch, air_density):
    return (house_length * house_width * house_height + np.tan(roof_pitch) * house_width * house_length) * air_density

class House(object):
    
    Length = 0
    
    Width = 0
    
    Height = 0
    
    RoofPitch = 0
    
    NumWindows = 0
    
    WindowHeight = 0
    
    WindowWidth = 0
    
    # R-Value for the walls, measured in R_SI units (m^2-K/W)
    RWall = 0
    
    # R-Value for the windows, measured in R_SI units (m^2-K/W)
    RWindow = 0
    
    def GetTotalWindowArea(self):
        return self.NumWindows*self.WindowHeight*self.WindowWidth
    
    def GetTotalWallArea(self):
        totalWindowArea = self.GetTotalWindowArea()
        return 2*self.Length*self.Height + 2*self.Width*self.Height + 2*(1/np.cos(self.RoofPitch/2))*self.Width*self.Length + np.tan(self.RoofPitch)*self.Width - totalWindowArea
    
    def GetThermalResistance(self):
        
        TRWall = self.RWall / self.GetTotalWallArea() # K/W
        TRWindow = self.RWindow / self.GetTotalWindowArea() # K/W
        
        # Returns the equivalent thermal resistance of the whole house
        return TRWall * TRWindow / (TRWall + TRWindow)
    
    # Returns the total air mass in the house, where airDensity is the density of air in kg/m^3
    def GetTotalAirMass(self, airDensity):
        return (self.Length * self.Width * self.Height + np.tan(self.RoofPitch) * self.Width * self.Length) * airDensity
    
class Heater(object):
    
    MaximumPower = 0
    
    NumHeaters = 0
    
    def GetTotalMaximumPower(self):
        return self.MaximumPower * self.NumHeaters

# Dictionary storing the parameters:
# - ThermalResistance: The thermal resistance of the house
# - AirMass: The total air mass in the house
# - AirHeatCapacity: The specific heat capacity of the air
# - MaximumHeatingPower: The (total) maximum heating power of the heater(s)
# - NumActions: The number of actions that can be taken
# - CostPerJoule: The cost per joule of heating
# - Discount: The discount factor
# - Hysteresis: The hysteresis of the temperature
# - Dt: The time to integrate over at each step

class HouseEnvironmentV2(py_environment.PyEnvironment):
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def __init__(self, params, outdoorTemperatures, device="cpu"):
        super().__init__()
        
        # Define the action and observation space
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=params["NumActions"]-1, name="action")
        self._observation_spec = array_spec.BoundedArraySpec(shape=(2,), dtype=np.float32, name="observation")
        
        # Set the parameters
        self._params = params
        
        # Define the temperatures
        self._outdoorTemperatures = outdoorTemperatures
        self._indoorTemperatures = np.zeros(len(outdoorTemperatures))
        self._indoorTemperatures[0] = 20
        self._targetTemperature = 21
        self._state = 0
    
    def _reset(self):
        self._state = 0
        self._episode_ended = False
        
        self._indoorTemperatures = np.zeros(len(self._outdoorTemperatures))
        self._indoorTemperatures[0] = 20 # Todo: change this to a random value
        self._targetTemperature = 21 # Todo: change this to a random value
        
        return ts.restart(np.array([self._targetTemperature-self._indoorTemperatures[0], 0], dtype=np.float32))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        # Get the parameters
        airHeatCapacity = self._params["AirHeatCapacity"]
        airMass = self._params["AirMass"]
        thermalResistance = self._params["ThermalResistance"]
        maximumHeatingPower = self._params["MaximumHeatingPower"]
        numActions = self._params["NumActions"]
        costPerJoule = self._params["CostPerJoule"]
        discount = self._params["Discount"]
        hysteresis = self._params["Hysteresis"]
        dt = self._params["Dt"]
        
        # Get the current state
        indoorTemp = self._indoorTemperatures[self._state]
        outdoorTemp = self._outdoorTemperatures[self._state]
        heaterState = action / (numActions-1)
        
        [newIndoorTemp, newHeatGain, newCostGain] = simulation.Step(outdoorTemp, indoorTemp, airMass, airHeatCapacity, thermalResistance, maximumHeatingPower, heaterState, costPerJoule, dt)
        self._state += 1
        
        self._indoorTemperatures[self._state] = newIndoorTemp
        
        # Calculate the reward
        reward = 1/(1+abs(self._targetTemperature-newIndoorTemp))
        
        # Calculate next observations
        nextObservations = np.array([self._targetTemperature-newIndoorTemp, newIndoorTemp-indoorTemp], dtype=np.float32)
        
        # Terminate the episode if the indoor temperature is outside the bounds.
        if abs(self._targetTemperature - newIndoorTemp) > hysteresis:
            self._episode_ended = True
            return ts.termination(nextObservations, reward=-1)
        
        # Terminate the episode if the current outdoor state is the last state.
        elif self._state >= len(self._outdoorTemperatures) - 2:
            self._episode_ended = True
            return ts.termination(nextObservations, reward=reward)
        
        # Else, transition to the next state.
        return ts.transition(nextObservations, reward=reward, discount=discount)
        
    
    def _current_time_step(self):
        return self._time_step
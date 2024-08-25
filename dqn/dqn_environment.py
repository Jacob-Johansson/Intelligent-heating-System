import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

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

class DQNEnvironment(py_environment.PyEnvironment):
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def __init__(self, params, outdoorTemperatures, device="cpu"):
        super().__init__()
        
        # Define the action and observation space
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=params["NumActions"]-1, name="action")
        self._observation_spec = array_spec.BoundedArraySpec(shape=(3,), dtype=np.float32, name="observation")
        
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
        
        hysteresis = self._params["Hysteresis"]
        self._indoorTemperatures = np.zeros(len(self._outdoorTemperatures))
        self._targetTemperature = 21 + np.random.randint(-hysteresis, hysteresis)
        self._indoorTemperatures[0] = self._targetTemperature - 0.5*hysteresis + np.random.random()*hysteresis
        
        return ts.restart(np.array([self._targetTemperature-self._indoorTemperatures[0], 0, self._outdoorTemperatures[10] - self._outdoorTemperatures[0]], dtype=np.float32))

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
        nextObservations = np.array([self._targetTemperature-newIndoorTemp, newIndoorTemp-indoorTemp, self._outdoorTemperatures[self._state + 1] - outdoorTemp], dtype=np.float32)
        
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
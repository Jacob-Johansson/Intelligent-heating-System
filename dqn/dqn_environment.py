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
    
    def __init__(self, params, outdoor_temperatures, device="cpu"):
        super().__init__()
        
        # Define the action and observation space
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=params["NumActions"]-1, name="action")
        self._observation_spec = array_spec.BoundedArraySpec(shape=(3,), dtype=np.float32, name="observation")
        
        # Set the parameters
        self._params = params
        
        # Define the temperatures
        self._outdoor_temperatures = outdoor_temperatures
        self._indoor_temperatures = np.zeros(len(outdoor_temperatures))
        self._indoor_temperatures[0] = 20
        self._target_temperature = 21
        self._state = 0
    
    def _reset(self):
        self._state = 0
        self._episode_ended = False
        
        hysteresis = self._params["Hysteresis"]
        self._indoor_temperatures = np.zeros(len(self._outdoor_temperatures))
        self._target_temperature = 21 + np.random.randint(-hysteresis, hysteresis)
        self._indoor_temperatures[0] = self._target_temperature - 0.5*hysteresis + np.random.random()*hysteresis
        
        return ts.restart(np.array([self._target_temperature-self._indoor_temperatures[0], 0, self._outdoor_temperatures[10] - self._outdoor_temperatures[0]], dtype=np.float32))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        # Get the parameters
        airheatcapacity = self._params["AirHeatCapacity"]
        airmass = self._params["AirMass"]
        thermalresistance = self._params["ThermalResistance"]
        maximumheatingpower = self._params["MaximumHeatingPower"]
        numactions = self._params["NumActions"]
        costperjoule = self._params["CostPerJoule"]
        discount = self._params["Discount"]
        hysteresis = self._params["Hysteresis"]
        dt = self._params["Dt"]
        
        # Get the current state
        indoortemperature = self._indoor_temperatures[self._state]
        outdoortemperature = self._outdoor_temperatures[self._state]
        heaterstate = action / (numactions-1)
        
        [new_indoortemperature, new_heatgain, new_costgain] = simulation.Step(outdoortemperature, indoortemperature, airmass, airheatcapacity, thermalresistance, maximumheatingpower, heaterstate, costperjoule, dt)
        self._state += 1
        
        self._indoor_temperatures[self._state] = new_indoortemperature
        
        # Calculate the reward
        reward = 1/(1+abs(self._target_temperature-new_indoortemperature))
        
        # Calculate next observations
        next_observations = np.array([self._target_temperature-new_indoortemperature, new_indoortemperature-indoortemperature, self._outdoor_temperatures[self._state + 1] - outdoortemperature], dtype=np.float32)
        
        # Terminate the episode if the indoor temperature is outside the bounds.
        if abs(self._target_temperature - new_indoortemperature) > hysteresis:
            self._episode_ended = True
            return ts.termination(next_observations, reward=-1)
        
        # Terminate the episode if the current outdoor state is the last state.
        elif self._state >= len(self._outdoor_temperatures) - 2:
            self._episode_ended = True
            return ts.termination(next_observations, reward=reward)
        
        # Else, transition to the next state.
        return ts.transition(next_observations, reward=reward, discount=discount)
        
    
    def _current_time_step(self):
        return self._time_step
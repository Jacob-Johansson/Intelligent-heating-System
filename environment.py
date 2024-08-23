import numpy as np
import simulation.importer as importer, simulation.simulation as simulation

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class Heater:
    
    HeatingPower = 0
    
    NumStates = 0
    
    NumHeaters = 0
    
    def GetTotalPower(self):
        return self.HeatingPower * self.NumHeaters

class HouseEnvironment(py_environment.PyEnvironment):
    
    OutdoorTemps = []
    
    TargetTemps = []
    
    Heater = Heater()
    
    TempDifferenceHysteresis = 0
    
    def GetHouseParameters(self):
        # Define the coefficients and variables.
        #-------------------------------
        #Define the house geometry
        #-------------------------------
        #House length = 30 m
        lenHouse = 30
        # House width = 4 m
        widHouse = 10
        # House height = 4 m
        htHouse = 4
        # Roof pitch = 40 deg
        pitRoof = np.radians(40)
        # Number of windows = 6
        numWindows = 6
        # Height of windows = 1 m
        htWindows = 1
        # Width of windows = 1 m
        widWindows = 1
        windowArea = numWindows*htWindows*widWindows
        wallArea = 2*lenHouse*htHouse + 2*widHouse*htHouse + 2*(1/np.cos(pitRoof/2))*widHouse*lenHouse + np.tan(pitRoof)*widHouse - windowArea
        
        return [lenHouse, widHouse, htHouse, pitRoof, wallArea, windowArea]
    
    def GetThermalResistance(self):
        
        # Get the house parameters
        [lenHouse, widHouse, htHouse, pitRoof, wallArea, windowArea] = self.GetHouseParameters()

        # -------------------------------
        # Define the type of insulation used, measured in R_SI units (m^2-K/W)
        # -------------------------------
        # Walls and windows are set to minimum recommended values in Sweden
        RWall = 5 # m^2-K/W
        RWindow = 1 # m^2-K/W
        TRWall = RWall / wallArea # K/W
        TRWindow = RWindow / windowArea # K/W

        # -------------------------------
        # Determine the equivalent thermal resistance for the whole building
        # -------------------------------
        thermalResistance = TRWall*TRWindow/(TRWall + TRWindow)
        return thermalResistance
    
    def GetAirMass(self):
        # -------------------------------
        # Determine total internal air mass
        # -------------------------------
        # Density of air at sea level = 1.2250 kg/m^3
        densAir = 1.2250
        [lenHouse, widHouse, htHouse, pitRoof, wallArea, windowArea] = self.GetHouseParameters()
        airMass = (lenHouse*widHouse*htHouse+np.tan(pitRoof)*widHouse*lenHouse)*densAir
        return airMass

    def GetAirHeatCapacity(self):
        # Specific heat capacity of air (273 K) = 1005.4 J/kg-K
        airHeatCapacity = 1005.4
        return airHeatCapacity
    
    def GetHeatingCost(self):
        # -------------------------------
        # Enter the cost of electricity and initial internal temperature
        # -------------------------------
        # Assume the cost of electricity is 110.15 ören per kilowatt/hour
        # Assume all electric energy is transformed to heat energy
        # 1 kW-hr = 3.6e6 J
        # cost = 110.15 ören per 3.6e6 J
        costPerJoule = 1.1015/3.6e6
        return costPerJoule
    
    def IsValidState(self, currentTempIndex, numTempStates):
        return currentTempIndex >= 0 and currentTempIndex < numTempStates
    
    def CalculateReward(self, state):
        return 1/(1+abs(state[0]))
    
    def GetStateSpaceIndex(self, currentTempDiff, minTempDiff, tempDiffStep):
        return int((currentTempDiff - minTempDiff)/tempDiffStep)
    
    def __init__(self):
        # Initialize heater parameters.
        self.Heater.HeatingPower = 1200
        self.Heater.NumStates = 5
        self.Heater.NumHeaters = 6
        
        self._action_spec = array_spec.BoundedArraySpec(shape=(self.Heater.NumStates,), dtype=np.int32, minimum=0, maximum=4, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(2,), dtype=np.float32, minimum=0, name='observation')
        self._state = np.zeros(4)
        self._episode_ended = False
        self.PREVIOUS_HEAT_STATE = 0

        
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        initialIndoorTemp = np.random.uniform(21 - self.TempDifferenceHysteresis * 0.9, 21 + self.TempDifferenceHysteresis * 0.9)
        self._state = [initialIndoorTemp, 0, 0, 0]
        self._episode_ended = False
        self.PREVIOUS_HEAT_STATE = 0
        return ts.restart([self.TargetTemps[0] - initialIndoorTemp, 0])
    
    def _step(self, action):
        
        # The action is the heater state.
        heaterState = action / (self.Heater.NumStates - 1)
        
        # The last action ended the episode.
        if self._episode_ended:
            return self.reset()
        
        t = 360
        
        [currentIndoorTemp, currentHeatGain, currentCost, currentTempIndex] = self._state
        currentOutdoorTemp = self.OutdoorTemps[int(currentTempIndex)]
        
        prevIndoorTemp = currentIndoorTemp
        nextTargetTemp = self.TargetTemps[int(currentTempIndex)+1]
        
        [newIndoorTemp, newHeatGain, newCost] = simulation.Step(currentOutdoorTemp, currentIndoorTemp, self.GetAirMass(), self.GetAirHeatCapacity(), self.GetThermalResistance(), self.Heater.GetTotalPower(), heaterState, self.GetHeatingCost(), t)
        currentIndoorTemp = newIndoorTemp
            
        self._state = [newIndoorTemp, newHeatGain, newCost, int(currentTempIndex)+1]
        
        nextState = [nextTargetTemp - newIndoorTemp, newIndoorTemp - prevIndoorTemp]
        
        # Calculate reward as the root mean square of the temperature difference.
        tempDifference = nextTargetTemp - newIndoorTemp
        r1 = 1/(1 + abs(tempDifference))
        #r2 = 1/(1 + abs(heaterState - self.PREVIOUS_HEAT_STATE))
        #r3 = 1/(1 + 10*abs(newCost))
        reward = r1 #np.sqrt((1/2) * (r1 + r3))
        
        self.PREVIOUS_HEAT_STATE = heaterState
        
        # Terminate the episode if the temperature difference is greater than the hysteresis.
        if abs(tempDifference) > np.abs(self.TempDifferenceHysteresis):
            self._episode_ended = True
            return ts.termination(nextState, reward=-1)
        
        # Terminate the episode if the current outdoor state is the last state.
        elif currentTempIndex >= len(self.OutdoorTemps) - 2:
            self._episode_ended = True
            return ts.termination(nextState, reward=reward)
        
        # Else, transition to the next state.
        else:
            return ts.transition(nextState, reward=reward, discount=0.9)


print(HouseEnvironment().GetThermalResistance())
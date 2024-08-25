import torch
import torch.backends
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import dqn
import environmentV2
import simulation
import numpy as np
import importer

# Segment the data temps, in hours, into 6-minute inverval.
def SegmentHourTempsInto6MinuteTemps(temps):
    # Segment the data temps, in hours, into 6-minute inverval.
    tempsMinutes = np.zeros(len(temps) * 10)
    for i in range(0, len(temps) - 1):
        for j in range(0, 10):
            # Lerp the temperature
            tempsMinutes[i * 10 + j] = temps[i] + (temps[i + 1] - temps[i]) * j / 10
    return tempsMinutes

# Define the parameters.
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
    "EpsilonDecay": 100,
    "Tau": 0.005,
    "LearningRate": 1e-4
}

# Setup the environment
outdoorTemps = SegmentHourTempsInto6MinuteTemps(importer.ImportHuskvarna2022())
targetTemps = np.ones(len(outdoorTemps))*21

env = environmentV2.HouseEnvironmentV2(params, outdoorTemps, "cpu")

targetTempsDayCycle = np.zeros(len(targetTemps))
for i in range(len(targetTemps)):
    time = i % (24*10)
    if time < 6 or time > 22:
        targetTempsDayCycle[i] = 21
    else:
        targetTempsDayCycle[i] = 18
        
houseTemps = np.zeros(len(outdoorTemps))
heaterStates = np.zeros(len(outdoorTemps))
costResult = np.zeros(len(outdoorTemps))
houseTemp = 20 # Initialize the house temperature.
prevHouseTemp = houseTemp
prevHeatGain = 0
tempHysteresis = params["Hysteresis"]

# Load model and weights.
num_actions = params["NumActions"]
num_observations = 3
model = dqn.DQN(num_observations, num_actions)
model.load_state_dict(torch.load('dqn/models/dqn_model_b.pth'))
model.eval()

totalCost = 0
for i in range(0, len(outdoorTemps)):
    outsideTemp = outdoorTemps[i]
    targetTemp = targetTempsDayCycle[i]
    
    # Get the parameters.
    air_mass = params["AirMass"]
    air_heat_capacity = params["AirHeatCapacity"]
    thermal_resistance = params["ThermalResistance"]
    maximum_heating_power = params["MaximumHeatingPower"]
    num_actions = params["NumActions"]
    cost_per_joule = params["CostPerJoule"]
    dt = params["Dt"]
    
    # Samples are taken every 6 minute.
    for j in range(0, 1):
        index = i
        houseTemps[index] = houseTemp
        
        tempDifferenceA = targetTemp - houseTemp
        tempDifferenceB = houseTemp - prevHouseTemp
        nextOutsideTempHour = outdoorTemps[i + 1] if i + 1 < len(outdoorTemps) else outdoorTemps[len(outdoorTemps) - 1]
        
        result = model.forward(torch.tensor([tempDifferenceA, tempDifferenceB, nextOutsideTempHour - outsideTemp], dtype=torch.float32))
        action = torch.argmax(result).item()
        heaterState = action / (num_actions - 1)
        
        simulationResult = simulation.Step(outsideTemp, houseTemp, air_mass, air_heat_capacity, thermal_resistance, maximum_heating_power, heaterState, cost_per_joule, dt)
        
        cost = simulation.heater.CalculateCost(maximum_heating_power, heaterState, cost_per_joule, dt)
        totalCost += cost
        
        prevHouseTemp = houseTemp
        prevHeatGain = simulationResult[1]
        houseTemp = simulationResult[0]
        heaterStates[index] = heaterState
        costResult[index] = totalCost

# Run the base simulation.
# Get the parameters.
air_mass = params["AirMass"]
air_heat_capacity = params["AirHeatCapacity"]
thermal_resistance = params["ThermalResistance"]
maximum_heating_power = params["MaximumHeatingPower"]
num_actions = params["NumActions"]
cost_per_joule = params["CostPerJoule"]
dt = params["Dt"]
[baseHouseTempResult, baseTargetTempResult, baseOutdoorTempResult, baseCostResult, baseHeatGainResult, baseHeaterStates] = simulation.Simulate(outdoorTemps, targetTemps, air_mass, air_heat_capacity, thermal_resistance, maximum_heating_power, targetTemp, cost_per_joule, houseTemp, tempHysteresis)

# Plot the results.
import matplotlib.pyplot as plt
plt.figure()
plt.plot(houseTemps)
plt.plot(baseHouseTempResult)
plt.plot(targetTempsDayCycle)
plt.plot(outdoorTemps)
plt.plot(heaterStates)
plt.plot(baseHeaterStates)
plt.plot(np.ones(len(targetTempsDayCycle)) * (targetTempsDayCycle + tempHysteresis), 'k--')
plt.plot(np.ones(len(targetTempsDayCycle)) * (targetTempsDayCycle - tempHysteresis), 'k--')
plt.legend(['DQN Indoor Temperature', 'Base Indoor Temperature', 'Target Temperature', 'Outside Temperature', 'DQN Heater States', 'Base Heater States', 'Upper Temperature Bounds', 'Lower Temperature Bounds'])
plt.grid()

plt.figure()
plt.plot(baseCostResult)
plt.plot(costResult)
plt.legend(['Base Cost', 'DQN Cost'])
plt.show()
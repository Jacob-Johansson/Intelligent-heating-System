import torch
import torch.backends
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import dqn
import dqn_environment
import simulation
import numpy as np
import importer

# Segment the data temps, in hours, into 6-minute inverval.
def segment_hour_temps_into_6minute_temps(temps):
    # Segment the data temps, in hours, into 6-minute inverval.
    tempsminutes = np.zeros(len(temps) * 10)
    for i in range(0, len(temps) - 1):
        for j in range(0, 10):
            # Lerp the temperature
            tempsminutes[i * 10 + j] = temps[i] + (temps[i + 1] - temps[i]) * j / 10
    return tempsminutes

def evaluate_dqn_model(model_path, params, outdoor_temperatures, target_temperatures):
    # Load model and weights.
    num_actions = params["NumActions"]
    num_observations = 3
    model = dqn.DQN(num_observations, num_actions)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Setup simulation.      
    indoor_temperatures = np.zeros(len(outdoor_temperatures))
    heater_states = np.zeros(len(outdoor_temperatures))
    costs = np.zeros(len(outdoor_temperatures))
    indoortemperature = 20 # Initialize the house temperature
    previousindoortemperature = indoortemperature
    previousheatgain = 0
    hysteresis = params["Hysteresis"]
    
    # Run the simulation.
    totalcost = 0
    for i in range(0, len(outdoor_temperatures)):
        outdoortemperature = outdoor_temperatures[i]
        targettemperature = target_temperatures[i]

        # Get the parameters.
        air_mass = params["AirMass"]
        air_heat_capacity = params["AirHeatCapacity"]
        thermal_resistance = params["ThermalResistance"]
        maximum_heating_power = params["MaximumHeatingPower"]
        num_actions = params["NumActions"]
        cost_per_joule = params["CostPerJoule"]
        dt = params["Dt"]

        # Step the simulation.
        indoor_temperatures[i] = indoortemperature

        targettemperaturedifference = targettemperature - indoortemperature
        indoortemperaturedifference = indoortemperature - previousindoortemperature
        outdoortemperaturedifference = outdoor_temperatures[i+1] if i + 1 < len(outdoor_temperatures) else outdoor_temperatures[len(outdoor_temperatures)-1] #target_temperatures[i + 1] if i + 1 < len(target_temperatures) else target_temperatures[len(target_temperatures) - 1]

        input = torch.tensor([
                targettemperaturedifference, 
                indoortemperaturedifference,
                outdoortemperaturedifference - outdoor_temperatures[i]
            ],
            dtype=torch.float32
        )
        
        result = model.forward(input)
        action = torch.argmax(result).item()
        heaterstate = action / (num_actions - 1)

        simulationresult = simulation.step(outdoortemperature, indoortemperature, air_mass, air_heat_capacity, thermal_resistance, maximum_heating_power, heaterstate, cost_per_joule, dt)

        cost = simulation.heater.calculate_cost(maximum_heating_power, heaterstate, cost_per_joule, dt)
        totalcost += cost

        previousindoortemperature = indoortemperature
        previousheatgain = simulationresult[1]
        indoortemperature = simulationresult[0]
        heater_states[i] = heaterstate
        costs[i] = totalcost
    
    return [indoor_temperatures, heater_states, costs]

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
total_window_area = dqn_environment.calculate_total_window_area(window_width, window_height, num_windows)
total_wall_area = dqn_environment.calculate_total_wall_area(house_width, house_height, house_length, roof_pitch, total_window_area)

params = {
    "ThermalResistance": dqn_environment.calculate_thermal_resistance(r_wall, r_window, total_wall_area, total_window_area),
    "AirMass": dqn_environment.calculate_total_air_mass(house_width, house_height, house_length, roof_pitch, 1.225),
    "AirHeatCapacity": 1005.4,
    "MaximumHeatingPower": heater_max_power,
    "NumActions": 10,
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
outdoor_temperatures = segment_hour_temps_into_6minute_temps(importer.import_huskvarna())
target_temperatures = np.zeros(len(outdoor_temperatures))
# Define the day and night temperatures
night_temperature = 18
day_temperature = 21

# Populate the target temperatures
for i in range(len(target_temperatures)):
    time = i % (24*10)
    # Night time: 10 PM to 6 AM
    # Lerp from day to night temperature
    if time > 21 * 10 and time < 22 * 10:
        target_temperatures[i] = day_temperature + (night_temperature - day_temperature) * (time - 21 * 10) / 10
    elif time >= 22 * 10 or time < 5 * 10:
        target_temperatures[i] = night_temperature
    # Lerp from night to day temperature
    elif time >= 5 * 10 and time < 6 * 10:
        target_temperatures[i] = night_temperature + (day_temperature - night_temperature) * (time - 5 * 10) / 10
    else:
        target_temperatures[i] = day_temperature

# Run the DQN model.
[indoor_temperatures, heater_states, costs] = evaluate_dqn_model('dqn/models/dqn_model_linear10actions.pth', params, outdoor_temperatures, target_temperatures)
import csv
with open('dqn/results/dqn_result_linear10actions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Indoor Temperature', 'Heater State', 'Cost'])
    for i in range(len(indoor_temperatures)):
        writer.writerow([indoor_temperatures[i], heater_states[i], costs[i]])
    file.close()

# Run the base model.
air_mass = params["AirMass"]
air_heat_capacity = params["AirHeatCapacity"]
thermal_resistance = params["ThermalResistance"]
maximum_heating_power = params["MaximumHeatingPower"]
num_actions = params["NumActions"]
cost_per_joule = params["CostPerJoule"]
hysteresis = params["Hysteresis"]
initialindoortemperature = 20
dt = params["Dt"]
[baseindoortemperatures, basetargettemperatures, baseoutdoortemperatures, basecosts, baseheatgains, baseheaterstates] = simulation.simulate(outdoor_temperatures, target_temperatures, air_mass, air_heat_capacity, thermal_resistance, maximum_heating_power, cost_per_joule, initialindoortemperature, hysteresis, dt)
import csv
with open('dqn/results/dqn_result_base.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Indoor Temperature', 'Outdoor Temperature', 'Target Temperature', 'Heater State', 'Cost'])
    for i in range(len(baseindoortemperatures)):
        writer.writerow([baseindoortemperatures[i], baseoutdoortemperatures[i], basetargettemperatures[i], baseheaterstates[i], basecosts[i]])
    file.close()
#Plot the results.
#import matplotlib.pyplot as plt
#plt.figure()
#plt.plot(indoor_temperatures)
#plt.plot(baseindoortemperatures)
#plt.plot(target_temperatures)
#plt.plot(outdoor_temperatures)
#plt.plot(heater_states)
#plt.plot(baseheaterstates)
#plt.plot(np.ones(len(target_temperatures)) * (target_temperatures + hysteresis), 'k--')
#plt.plot(np.ones(len(target_temperatures)) * (target_temperatures - hysteresis), 'k--')
#plt.legend(['DQN Indoor Temperature', 'Base Indoor Temperature', 'Target Temperature', 'Outside Temperature', 'DQN Heater States', 'Base Heater States', 'Upper Temperature Bounds', 'Lower Temperature Bounds'])
#plt.grid()
#
#plt.figure()
#plt.plot(basecosts)
#plt.plot(costs)
#plt.legend(['Base Cost', 'DQN Cost'])
#plt.show()
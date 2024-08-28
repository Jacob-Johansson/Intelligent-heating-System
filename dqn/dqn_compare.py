import numpy as np
import pandas as pd

# Returns an array of zeros and ones representing if the heater state has switched.
def calculate_heater_state_switching(heaterstates) -> []:
    output = []
    totalswitches = 0
    for i in range(1, len(heaterstates)):
        totalswitches += 1 if heaterstates[i] != heaterstates[i-1] else 0
        output.append(totalswitches)
    return output  

# Import base result.
baseresult = pd.read_csv(open('dqn/results/dqn_result_base.csv'))
base_indoortemperatures = baseresult.loc[:, 'Indoor Temperature']
base_outdoortemperatures = baseresult.loc[:, 'Outdoor Temperature']
base_targettemperatures = baseresult.loc[:, 'Target Temperature']
base_heaterstates = baseresult.loc[:, 'Heater State']
base_costs = baseresult.loc[:, 'Cost']

# Import DQN 10 actions result.
dqn10actions = pd.read_csv(open('dqn/results/dqn_result_10actions.csv'))
dqn10actions_indoortemperatures = dqn10actions.loc[:, 'Indoor Temperature']
dqn10actions_heaterstates = dqn10actions.loc[:, 'Heater State']
dqn10actions_costs = dqn10actions.loc[:, 'Cost']

# Import DQN 5 actions result.
dqn5actions = pd.read_csv(open('dqn/results/dqn_result_5actions.csv'))
dqn5actions_indoortemperatures = dqn5actions.loc[:, 'Indoor Temperature']
dqn5actions_heaterstates = dqn5actions.loc[:, 'Heater State']
dqn5actions_costs = dqn5actions.loc[:, 'Cost']

# Plot the results.
import matplotlib.pyplot as plt
plt.figure()
plt.grid()
# Indoor temperatures
plt.plot(base_indoortemperatures, label='Base Indoor Temperature')
plt.plot(dqn10actions_indoortemperatures, label='DQN 10 actions Indoor Temperature')
plt.plot(dqn5actions_indoortemperatures, label='DQN 5 actions Indoor Temperature')
# Outdoor temperatures
plt.plot(base_outdoortemperatures, label='Outdoor Temperature')
# Target temperatures
plt.plot(base_targettemperatures, label='Target Temperature')
# Hysteresis
hysteresis = 2
lower_temperature_bounds = np.zeros(len(base_targettemperatures))
upper_temperature_bounds = np.zeros(len(base_targettemperatures))
for i in range(len(base_targettemperatures)):
    lower_temperature_bounds[i] = base_targettemperatures[i] - hysteresis
    upper_temperature_bounds[i] = base_targettemperatures[i] + hysteresis
plt.plot(upper_temperature_bounds, 'k--', label='Upper Temperature Bounds')
plt.plot(lower_temperature_bounds, 'k--', label='Lower Temperature Bounds')

plt.legend()

# Heater states
base_heater_switching = calculate_heater_state_switching(base_heaterstates)
dqn10actions_heater_switching = calculate_heater_state_switching(dqn10actions_heaterstates)
dqn5actions_heater_switching = calculate_heater_state_switching(dqn5actions_heaterstates)
plt.figure()
plt.grid()
plt.plot(base_heater_switching, label='Base Heater State')
plt.plot(dqn10actions_heater_switching, label='DQN 10 actions Heater State')
plt.plot(dqn5actions_heater_switching, label='DQN 5 actions Heater State')
plt.legend()

# Costs
plt.figure()
plt.plot(base_costs, label='Base Cost')
plt.plot(dqn10actions_costs, label='DQN 10 actions Cost')
plt.plot(dqn5actions_costs, label='DQN 5 actions Cost')
plt.legend()

plt.show()
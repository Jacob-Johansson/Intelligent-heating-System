import simulation.importer as importer, simulation.simulation as simulation, qstar
import matplotlib.pyplot as plt
import numpy as np

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

# -------------------------------
# Define the heater parameters
# -------------------------------
heaterPower = 6 * 1200 # 7.2 kW total

# -------------------------------
# Determine total internal air mass
# -------------------------------
# Density of air at sea level = 1.2250 kg/m^3
densAir = 1.2250
airMass = (lenHouse*widHouse*htHouse+np.tan(pitRoof)*widHouse*lenHouse)*densAir
print(airMass)

# Specific heat capacity of air (273 K) = 1005.4 J/kg-K
airHeatCapacity = 1005.4

# Target temperature (targetTemp)
targetTemp = 21 # degree Celsius

hysteresis = 2

# -------------------------------
# Enter the cost of electricity and initial internal temperature
# -------------------------------
# Assume the cost of electricity is 110.15 ören per kilowatt/hour
# Assume all electric energy is transformed to heat energy
# 1 kW-hr = 3.6e6 J
# cost = 110.15 ören per 3.6e6 J
costPerJoule = 1.1015/3.6e6

# Define the initial values.
initialHouseTemp = 20

temps = importer.ImportHuskvarna()

outTemps = np.zeros(48*10)
for i in range(0, len(outTemps)):
    outTemps[i] = 5 * np.sin(2 * np.pi * i / len(outTemps)) + 0

# Load the q-table.
qTable = np.load('qTable.npy')
qTableDay = np.load('qTableDayOrNight.npy')
qTableNight = np.load('qTableNight.npy')

# Load test data.
temps = importer.ImportHuskvarna2022()

# Get first week of data.
temps = temps[0:24*7]

outTemps = np.zeros(len(temps) * 10)
for i in range(0, len(temps) - 1):
    for j in range(0, 10):
        # Lerp the temperature
        outTemps[i * 10 + j] = temps[i] + (temps[i + 1] - temps[i]) * j / 10

# Define target temps
targetTemps = np.zeros(len(outTemps))
for i in range(len(targetTemps)):
    time = i % (24*10)
    if time < 6 or time > 22:
        targetTemps[i] = 21
    else:
        targetTemps[i] = 18
        


# Run the base simulation.
[roomTempResult, targetTempResult, outTempResult, costResult, heatGainResult, states] = simulation.Simulate(outTemps, targetTemps, airMass, airHeatCapacity, thermalResistance, heaterPower, targetTemp, costPerJoule, initialHouseTemp, hysteresis)

# Run the q-learning simulation.
[roomTempResultQ, targetTempResultQ, outTempResultQ, costResultQ, heatGainResultQ, statesQ] = qstar.Simulate(outTemps, targetTemps, airMass, airHeatCapacity, thermalResistance, heaterPower, costPerJoule, initialHouseTemp, qTableDay)

# Plot the results.
plt.figure()
plt.plot(roomTempResult)
plt.plot(roomTempResultQ)
plt.plot(targetTempResult)
plt.plot(outTempResult)
plt.plot(np.ones(len(targetTemps)) * (targetTemps + hysteresis), 'k--')
plt.plot(np.ones(len(targetTemps)) * (targetTemps - hysteresis), 'k--')
plt.legend(['Indoor Temperature Base', 'Indoor Temperature Q-Learning', 'Target Temperature', 'Outside Temperature', 'Upper Temperature Bounds', 'Lower Temperature Bounds'])
plt.grid()

plt.figure()
plt.plot(costResult)
plt.plot(costResultQ)
plt.xlabel('Time (hours)')
plt.ylabel('Cost (SEK)')
plt.legend(['Cost Base', 'Cost Q-Learning'])

plt.figure()
plt.plot(heatGainResult / 3.6e6)
plt.plot(heatGainResultQ / 3.6e6)
plt.legend(['Heat Gain Base (KWh)', 'Heat Gain Q-Learning (KWh)'])

plt.figure()
plt.plot(states)
plt.plot(statesQ)
plt.legend(['State Base', 'State Q-Learning'])

plt.show()
